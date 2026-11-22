#!/usr/bin/env python3
"""
The card timelines' `history` branch of TTSCardMaker (docs/Future/CARD_TIMELINE.md step 6; owner D10). By hand, on the
Mac, after `discord_preview build` and `check`:

    python scripts/history_branch.py [--root <archive>] [--store <bare repo>] [--remote <url>] [--push]

One commit of the archive's derived/preview/history (index.json, the card folders, _morgue/: timeline.json and webps,
nothing else) on top of the branch's head on the remote (no branch yet: an orphan root). Made in a bare repo of its own
(STORE, never a card repo clone: it fetches refs/heads/history alone, depth 1) with a temporary index and the folder as
its work tree. It refuses when the folder fails a check (layout, JSON keys, links/mentions/snowflakes, the left-out
channels' pictures, webp metadata and width, sizes) and prints what it would push; --push pushes that commit to
refs/heads/history only, never forced (a branch moved since the fetch refuses: run it again).
The branch runs no Action (every workflow's push trigger is main's) and Card sync never reads it; the message carries
`Card-sweep: history` (changelog.SWEEP_TRAILER) all the same.
"""
import sys
from argparse import ArgumentParser
from json import loads
from os import environ, getpid, listdir, remove, walk
from os.path import abspath, dirname, exists, expanduser, getsize, isdir, islink, join, relpath
from re import compile as re_compile, sub
from subprocess import PIPE, run
from typing import Optional

sys.path.insert(0, dirname(dirname(abspath(__file__))))          # AutoCroissant, for commands.* and global_config
from commands import discord_match as dm                          # noqa: E402
from commands.changelog import SWEEP_TRAILER                      # noqa: E402
from global_config import CARD_REPO                               # noqa: E402

# ========================
# Configuration
# ========================
BRANCH = "refs/heads/history"
FOLDER = ("derived", "preview", "history")                        # discord_preview's OUT
STORE = "~/Documents/Tools/history.git"                           # this script's own bare repo
REMOTE = f"git@github.com:{CARD_REPO}.git"
AUTHOR = "MichaelJSr"
MORGUE = "_morgue"
WEBP_W = 720                                                      # discord_preview.WEBP_W
LIST_W, LIST_SUFFIX = 360, "-list.avif"                           # discord_preview's: a Morgue entry's list copy
FILE_MAX = 50_000_000                                             # GitHub warns past 50 MB a file
TOTAL_MAX = 700_000_000                                           # the site copies every webp; Pages' 1 GB is the whole
                                                                  # site's (161 MB without it, 11-17): room to grow
INDEX_KEYS = {"version", "generated", "about", "cards", "morgue", "mdw", "created"}
TIMELINE_KEYS = {"card", "key", "name", "entries", "kind", "stop", "created", "git_first", "morgue", "first", "kind_of",
                 "git_file", "live", "list"}
ENTRY_KEYS = {"id", "kind", "date", "day", "badge", "image", "w", "h", "fields", "changes", "minor", "author", "subject",
              "channel", "repo", "posts", "era", "git_look", "change", "file", "confirmed", "confirmed_by", "as",
              "meme_copy", "also"}                                # a new key: add it once it is known to hold no message text
# also: a Morgue picture's other posts (date, day, channel; owner 11-21)
RAW_KEYS = {"message", "message_id", "messages", "content", "poster", "author_id", "jump_url", "url", "link", "links",
            "mentions", "attachment", "attachments", "note", "notes", "why", "review", "conf", "box", "channel_id",
            "guild_id"}                                           # the archive's and the review page's: never published
LINKS = re_compile(r"https?://|www\.|discord(?:app)?\.(?:com|gg)|discordapp|<[@#][!&]?\d|@everyone|@here|\d{17,}")
# 17+ digits anywhere: a snowflake glued to letters or '_' too (today's longest digit run: 12)


def log(*args) -> None:
    print(*args, flush=True)


def shown(url: str) -> str:
    """A remote as printed: never its credentials."""
    return sub(r"(://)[^/@]*@", r"\1", url)


# ========================
# Checks
# ========================
def left_out_ids(root: str) -> set:
    """The left-out channels' pictures (discord_match.LEFT_OUT: the physical card photos, owner Q6): sha256[:12], the
    picture ids' part."""
    path = join(root, "attachments.jsonl")
    if not exists(path):
        raise SystemExit(f"refused: {path} missing: the left-out pictures can't be checked")
    ids = dm.channel_ids(root, dm.LEFT_OUT)
    if sorted(ids.values()) != sorted(dm.LEFT_OUT):
        raise SystemExit(f"refused: {join(root, 'channels.json')} must name each of {', '.join(dm.LEFT_OUT)} once")
    return {row["sha256"][:12] for row in dm.read_jsonl(path) if row.get("container_id") in ids and row.get("sha256")}


def keys_of(obj) -> set:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in keys_of(v)}
    return {k for v in obj for k in keys_of(v)} if isinstance(obj, list) else set()


def webp_problem(path: str, fmt: str = "WEBP", width: int = WEBP_W) -> Optional[str]:
    """A picture's problem (a list copy: fmt AVIF, LIST_W), None when it has none."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            if im.format != fmt:
                return f"not a {fmt.lower()} ({im.format})"
            if im.width > width:
                return f"{im.width} px wide (at most {width})"
            extra = [k for k in ("exif", "xmp") if im.info.get(k)]
            if extra or getattr(im, "n_frames", 1) != 1:
                return f"metadata {', '.join(extra) or 'frames'}"
            return None
    except Exception as error:
        return f"unreadable ({type(error).__name__})"


def check(folder: str, photos: set) -> tuple:
    """(problems, counts) of the folder as the branch would hold it."""
    problems, counts = [], {"cards": 0, "morgue": 0, "timelines": 0, "webps": 0, "files": 0, "bytes": 0}
    sizes = []
    if not exists(join(folder, "index.json")):
        return [f"{folder}: no index.json"], counts
    for top, dirs, files in walk(folder):
        rel = relpath(top, folder)
        depth = 0 if rel == "." else rel.count("/") + 1
        for name in dirs + files:
            path = join(top, name)
            if name.startswith(".") or islink(path):
                problems.append(f"{relpath(path, folder)}: a hidden file or link")
            if LINKS.search(name):
                problems.append(f"{relpath(path, folder)}: a link or snowflake in its name")
        for name in dirs:
            if depth == 1 and rel != MORGUE or depth > 1:
                problems.append(f"{relpath(join(top, name), folder)}/: a folder inside a card's")
        for name in files:
            path = join(top, name)
            size = getsize(path)
            counts["files"] += 1
            counts["bytes"] += size
            sizes.append((size, relpath(path, folder)))
            ok = (name == "index.json" if depth == 0 else
                  False if rel == MORGUE else name == "timeline.json" or name.endswith(".webp")
                  or name.endswith(LIST_SUFFIX) and rel.startswith(MORGUE + "/"))
            if not ok:
                problems.append(f"{relpath(path, folder)}: not the history's (index.json, timeline.json, webps, "
                                "a Morgue entry's list copy only)")
            if size > FILE_MAX:
                problems.append(f"{relpath(path, folder)}: {size / 1e6:.1f} MB")
            if any(k in name for k in photos):
                problems.append(f"{relpath(path, folder)}: a left-out channel's picture")
    if counts["bytes"] > TOTAL_MAX:
        problems.append(f"{counts['bytes'] / 1e6:.0f} MB in all (at most {TOTAL_MAX / 1e6:.0f}: "
                        "Pages' 1 GB less the site)")

    def text_of(path):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        hits = {m.group(0)[:12] for m in LINKS.finditer(text)}
        if hits:
            problems.append(f"{relpath(path, folder)}: {len(hits)} link, mention or snowflake")
        hits = {k for k in photos if k in text}
        if hits:
            problems.append(f"{relpath(path, folder)}: {len(hits)} left-out channel pictures named")
        try:
            data = loads(text)
        except ValueError:
            problems.append(f"{relpath(path, folder)}: not JSON")
            return None
        raw = keys_of(data) & RAW_KEYS
        if raw:
            problems.append(f"{relpath(path, folder)}: keys {', '.join(sorted(raw))}")
        return data

    index = text_of(join(folder, "index.json")) or {}
    if not isinstance(index, dict) or index.get("version") != 1:
        problems.append("index.json: not version 1")
        index = {}
    if set(index) - INDEX_KEYS:
        problems.append(f"index.json: keys {', '.join(sorted(set(index) - INDEX_KEYS))}")
    named = [(row or {}).get("dir") or sub_dir for sub_dir, row in (index.get("cards") or {}).items()]
    counts["cards"] = len(named)
    extra = [row.get("dir") for row in (index.get("morgue") or []) + (index.get("mdw") or []) if isinstance(row, dict)]
    counts["morgue"] = len(extra)
    named += extra
    on_disk = {d for d in listdir(folder) if isdir(join(folder, d)) and d != MORGUE}
    if isdir(join(folder, MORGUE)):
        on_disk |= {f"{MORGUE}/{d}" for d in listdir(join(folder, MORGUE)) if isdir(join(folder, MORGUE, d))}
    for d in sorted(on_disk - set(named)):
        problems.append(f"{d}/: no index.json row")
    for d in named:
        if not isinstance(d, str) or ".." in d.split("/") or not isdir(join(folder, d)):
            problems.append(f"index.json: {d}: no such folder")
            continue
        path = join(folder, d, "timeline.json")
        if not exists(path):
            problems.append(f"{d}/: no timeline.json")
            continue
        counts["timelines"] += 1
        t = text_of(path)
        if not isinstance(t, dict) or not isinstance(t.get("entries"), list):
            problems.append(f"{d}/timeline.json: no entries")
            continue
        if set(t) - TIMELINE_KEYS:
            problems.append(f"{d}/timeline.json: keys {', '.join(sorted(set(t) - TIMELINE_KEYS))}")
        pictures = {f for f in listdir(join(folder, d)) if f.endswith(".webp")}
        copies = {f for f in listdir(join(folder, d)) if f.endswith(LIST_SUFFIX)}
        used = set()
        for e in t["entries"]:
            if not isinstance(e, dict) or set(e) - ENTRY_KEYS:
                problems.append(f"{d}/timeline.json: entry keys {', '.join(sorted(set(e) - ENTRY_KEYS)) if isinstance(e, dict) else e}")
                continue
            if e.get("image"):
                used.add(e["image"])
                if e["image"] not in pictures:
                    problems.append(f"{d}/{e['image']}: missing")
        for f in sorted(pictures - used):
            problems.append(f"{d}/{f}: no entry shows it")
        for f in sorted(pictures):
            counts["webps"] += 1
            why = webp_problem(join(folder, d, f))
            if why:
                problems.append(f"{d}/{f}: {why}")
        mine = t.get("list")                  # the list copy of one of its pictures
        if mine is not None and not (isinstance(mine, str) and mine in copies and d.startswith(MORGUE + "/")
                                     and mine[:-len(LIST_SUFFIX)] + ".webp" in used):
            problems.append(f"{d}/timeline.json: list {str(mine)[:40]}: no list copy of a picture it shows")
        for f in sorted(copies):
            counts["lists"] = counts.get("lists", 0) + 1
            why = "no timeline names it" if f != mine else webp_problem(join(folder, d, f), "AVIF", LIST_W)
            if why:
                problems.append(f"{d}/{f}: {why}")
    counts["largest"] = sorted(sizes, reverse=True)[:3]
    return problems, counts


# ========================
# The branch
# ========================
class Refused(Exception):
    pass


def git(store: str, *args, env: Optional[dict] = None, data: Optional[bytes] = None, cwd: Optional[str] = None) -> str:
    done = run(["git", *args], input=data, stdout=PIPE, stderr=PIPE, cwd=cwd,
               env={**environ, "GIT_DIR": store, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0", **(env or {})})
    if done.returncode:
        verb = next((a for a in args if not a.startswith("-") and "=" not in a), "?")
        raise Refused(f"git {verb} failed: {shown(done.stderr.decode('utf-8', 'replace').strip())}")
    return done.stdout.decode("utf-8")


def open_store(store: str) -> None:
    if not exists(join(store, "HEAD")):
        run(["git", "init", "-q", "--bare", store], check=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
        git(store, "config", "gc.auto", "0")
        git(store, "config", "maintenance.auto", "false")


def remote_head(store: str, remote: str) -> str:
    """The branch's head on the remote, fetched alone and shallow; "" when there is none."""
    listed = git(store, "ls-remote", remote, BRANCH).split()
    if not listed:
        return ""
    git(store, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", remote,
        f"+{BRANCH}:refs/remotes/origin/history")
    head = git(store, "rev-parse", "--verify", "refs/remotes/origin/history^{commit}").strip()
    if head != listed[0]:
        raise Refused(f"fetched {head[:12]}, the remote lists {listed[0][:12]}: run it again")
    return head


def make_commit(store: str, folder: str, parent: str, counts: dict) -> tuple:
    """(commit, tree); commit "" when the tree is the parent's."""
    index = join(store, f"history-index-{getpid()}")
    env = {"GIT_INDEX_FILE": index, "GIT_WORK_TREE": folder}
    try:
        git(store, "-c", "core.fileMode=false", "-c", "core.excludesFile=/dev/null", "add", "-A", "-f", "--", ".",
            env=env, cwd=folder)
        tree = git(store, "write-tree", env=env).strip()
    finally:
        if exists(index):
            remove(index)
    files = git(store, "ls-tree", "-r", "--name-only", tree).count("\n")
    if files != counts["files"]:
        raise Refused(f"the tree holds {files} files, the folder {counts['files']}")
    if parent and git(store, "rev-parse", f"{parent}^{{tree}}").strip() == tree:
        return "", tree
    lists = f", {counts['lists']} list copies" if counts.get("lists") else ""
    message = f"History: {counts['cards']} cards, {counts['webps']} pictures{lists}\n\n{SWEEP_TRAILER}: history\n"
    who = {"GIT_AUTHOR_NAME": AUTHOR, "GIT_COMMITTER_NAME": AUTHOR}
    commit = git(store, "commit-tree", tree, *(["-p", parent] if parent else []), env=who, data=message.encode()).strip()
    return commit, tree


def report(store: str, commit: str, parent: str) -> None:
    """What the push sends: files against the parent and the new objects' size."""
    if parent:
        status = [line.split("\t")[0] for line in git(store, "diff-tree", "-r", "--no-renames", "--name-status",
                                                      parent, commit).splitlines()]
        log(f"against {parent[:12]}: {status.count('A')} added, {status.count('M')} changed, {status.count('D')} deleted")
    ids = git(store, "rev-list", "--objects", commit, *(["--not", parent] if parent else [])).split("\n")
    ids = "\n".join(line.split(" ")[0] for line in ids if line) + "\n"
    sizes = git(store, "cat-file", "--batch-check=%(objectsize)", data=ids.encode()).split()
    log(f"new objects: {len(sizes)}, {sum(int(s) for s in sizes) / 1e6:.1f} MB")


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python scripts/history_branch.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
    parser.add_argument("--store", default=STORE)
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--push", action="store_true", help="push the commit to the branch (else a dry run)")
    args = parser.parse_args(argv)
    root, store = expanduser(args.root), expanduser(args.store)
    folder = join(root, *FOLDER)
    problems, counts = check(folder, left_out_ids(root))
    log(f"{folder}: {counts['cards']} cards, {counts['morgue']} Morgue folders, {counts['timelines']} timelines, "
        f"{counts['webps']} webps, {counts.get('lists', 0)} list copies, {counts['files']} files, "
        f"{counts['bytes'] / 1e6:.1f} MB (at most {TOTAL_MAX / 1e6:.0f}); largest "
        + ", ".join(f"{n} {s / 1e6:.2f} MB" for s, n in counts.get("largest", [])))
    if problems:
        for p in problems[:30]:
            log(f"  {p}")
        log(f"refused: {len(problems)} problems{' (first 30 shown)' if len(problems) > 30 else ''}")
        return 1
    log("checks: layout, JSON keys, links/mentions/snowflakes, left-out pictures, webps, list copies, sizes: ok")
    try:
        open_store(store)
        parent = remote_head(store, args.remote)
        commit, tree = make_commit(store, folder, parent, counts)
        if not commit:
            log(f"unchanged: the branch's tree is {tree[:12]} already, nothing to push")
            return 0
        log(f"commit {commit} (tree {tree[:12]}), parent {parent or 'none: an orphan root'}")
        report(store, commit, parent)
        if not args.push:
            git(store, "update-ref", "refs/dry/history", commit)
            log(f"dry run: --push sends it to {shown(args.remote)} {BRANCH}")
            return 0
        git(store, "-c", "pack.window=0", "push", "-q", args.remote, f"{commit}:{BRANCH}")
        if git(store, "ls-remote", args.remote, BRANCH).split()[:1] != [commit]:
            raise Refused("pushed, but the remote lists another head")
        git(store, "update-ref", BRANCH, commit)
        git(store, "update-ref", "refs/remotes/origin/history", commit)
        log(f"pushed {commit[:12]} to {shown(args.remote)} {BRANCH}")
        return 0
    except Refused as error:
        log(f"refused: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
