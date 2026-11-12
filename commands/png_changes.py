"""
Picture changes of the card PNGs, without Discord or psd-tools: db/png_changes.json on the data branch holds, for each
change of a card's PNG (its blob before and after), the share of the card's pixels that visibly changed (RULE). The
website lists an image version at VISIBLE or more (or with a data change) and folds the rest as minor; a changelog art
event under it is minor (no Discord post). A pair the record lacks counts as visible (owner 11-09).

Card sync measures the changes its run brings (card_data.build, in a child process beside its PSD reads: Beside); the
backfill measures history's, in batches that are fetched into scratch stores and dropped (never the whole history on
disk), and resumes where the record stops.

CLI: python -m commands.png_changes backfill --checkout <clone with main's history> --db <db dir> [--minutes N] [--events]
"""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps, loads
from os import environ, replace
from os.path import abspath, basename, dirname, isfile, join, splitext
from re import sub
from subprocess import DEVNULL, PIPE, STDOUT, Popen, TimeoutExpired, run
from sys import executable
from tempfile import TemporaryDirectory
from time import time
from typing import Optional
from weakref import finalize

from commands.card_models import RENAMED, rename_trailer      # stdlib only: the backfill's sparse copy is this and it

# ========================
# Configuration
# ========================
RULE = 1                     # the per-pixel rule's version (a pair measured by another is measured again)
TOLERANCE = 2                # a pixel visibly changed: a channel (colour over black, or alpha) off by more than this of
                             # 255 (the owner's "near identical" bar, 11-07), ...
PATCH = 7                    # ... inside a PATCH x PATCH square of such pixels (re-drawn text strokes, outlines and
                             # badge rims are no changed area; calibration 11-09)
VISIBLE = 0.05               # share of the card's pixels (alpha in either picture): a visible change (build_site PNG_VISIBLE;
                             # owner 11-09 "about 5%": the Alt template's bars are 2-7% of a card)
RECORD = "png_changes.json"
BATCH = 30                   # pairs whose blobs one scratch store holds
FETCHES = 4                  # batches fetched at a time, ahead of the one compared (the guards' FETCH_POOL)
WORKERS = 4                  # pictures compared at a time
JOIN_WAIT = 600              # s card_data waits for Beside at most (then measured in the build, as before)
RUN_MAX = 300                # pairs a Card sync run measures beside its PSD reads, and of its own events after
                             # them; the rest wait for the next run or the backfill
EXIT_LEFT = 3                # backfill: pairs left when the time ran out
NO_LAZY = {"GIT_NO_LAZY_FETCH": "1"}
BOT = "AutoCroissant"        # git_commit.AUTHOR: its Recompressed trailers name PNGs with the same pixels
RECOMPRESSED = "Recompressed"
ROOT = dirname(dirname(abspath(__file__)))    # Beside's child: python -m commands.png_changes here
CARD_PLACES = ("Creatures/", "Items/", "Field/", "N.M.E/", "Auxiliary/", "Auxillary/", "MDW/The Morgue/",
               "MDW/Unfinished/")    # card PNGs, and where a card's picture went and came back from


# ========================
# The record
# ========================
def key(old: str, new: str) -> str:
    return f"{old}..{new}"


def read_record(db_dir: str) -> dict:
    """{"rule": RULE, "pairs": {"<old>..<new>": [share or None (unreadable), rule]}, "pending": [[old, new], ...] (a Card
    sync run's pairs over RUN_MAX: the next run's first)}; empty without the file."""
    path = join(db_dir, RECORD)
    data = {}
    if isfile(path):
        with open(path, encoding="utf-8") as fh:
            data = loads(fh.read())
    return {"rule": RULE, "pairs": dict(data.get("pairs") or {}), "pending": [tuple(p) for p in data.get("pending") or []]}


def _measured(record: dict, pair: tuple) -> bool:
    return pair[0] == pair[1] or (record["pairs"].get(key(*pair)) or [0, None])[1] == RULE


def record_text(record: dict) -> str:
    pending = [list(p) for p in dict.fromkeys(record.get("pending") or []) if not _measured(record, p)]
    return dumps({"rule": RULE, "pairs": record["pairs"], "pending": pending}, indent=0, sort_keys=True) + "\n"


def write_record(db_dir: str, record: dict) -> None:
    """Whole or not at all (a temporary file replaced)."""
    path = join(db_dir, RECORD)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(record_text(record))
    replace(path + ".tmp", path)


def share_of(record: dict, old: str, new: str) -> Optional[float]:
    """The pair's share by this RULE; None when not measured (or unreadable)."""
    if old == new:
        return 0.0
    found = record["pairs"].get(key(old, new))
    return found[0] if found and found[1] == RULE else None


def visible(record: dict, old: Optional[str], new: Optional[str]) -> bool:
    """Whether a change from blob old to new shows: measured at VISIBLE or more, or not measured."""
    if not old or not new:
        return bool(old or new)
    share = share_of(record, old, new)
    return share is None or share >= VISIBLE


# ========================
# The rule
# ========================
def _pixels(data: bytes):
    from numpy import asarray, uint16
    from PIL import Image
    with Image.open(BytesIO(data)) as im:
        return asarray(im.convert("RGBA"), dtype=uint16)


def _runs(mask, op):
    """op (logical_and or logical_or) of the PATCH pixels centred on each along axis 1 (outside: unset), transposed:
    twice is the PATCH x PATCH square."""
    from numpy import pad
    out, span = pad(mask, ((0, 0), (PATCH // 2, PATCH // 2))), 1
    while span < PATCH:                         # out[:, i]: op of the span pixels from i
        step = min(span, PATCH - span)
        out, span = op(out[:, :-step], out[:, step:]), span + step
    return out.T


def changed_area(a, b):
    """The visibly changed pixels of two RGBA uint16 pictures of one size (bool): colour over black or alpha moved by
    more than TOLERANCE in a channel, inside a PATCH x PATCH square of such pixels (a morphological opening)."""
    from numpy import abs as np_abs, int16, logical_and, logical_or, zeros
    if a[..., 3].min() == 255 and b[..., 3].min() == 255:      # opaque: colour over black is the colour
        moved = zeros(a.shape[:2], bool)
        for c in range(3):
            d = a[..., c].view(int16) - b[..., c].view(int16)
            moved |= (d > TOLERANCE) | (d < -TOLERANCE)
    else:
        ca, cb = (a[..., :3] * a[..., 3:] // 255).astype(int16), (b[..., :3] * b[..., 3:] // 255).astype(int16)
        moved = (np_abs(ca - cb).max(axis=2) > TOLERANCE) | (np_abs(a[..., 3].astype(int16) - b[..., 3].astype(int16)) > TOLERANCE)
    eroded = _runs(_runs(moved, logical_and), logical_and)
    return _runs(_runs(eroded, logical_or), logical_or)


def changed_share(old: bytes, new: bytes) -> float:
    """The share of the card's pixels (alpha in either) visibly changed (changed_area); 1.0 when the size changed."""
    a, b = _pixels(old), _pixels(new)
    if a.shape != b.shape:
        return 1.0
    area = changed_area(a, b)
    card = (a[..., 3] > 0) | (b[..., 3] > 0)
    n = int(card.sum())
    return round(float((area & card).sum()) / n, 5) if n else 0.0


# ========================
# Pairs from git
# ========================
def _git(repo: str, *args: str) -> str:
    return run(["git", "-C", repo, *args], capture_output=True, text=True, check=True, env={**environ, **NO_LAZY}).stdout


def _stem(path: str) -> str:
    return sub(r"[^0-9a-z]", "", splitext(basename(path))[0].lower())


def card_pairs(checkout: str, head: str, base: Optional[str] = None) -> list:
    """(old blob, new blob) of each card PNG change in base..head (head's whole history without base), oldest first, as
    the website's image versions follow them: a PNG modified (not one the bot only recompressed: its Recompressed
    trailer), or deleted and added in one commit under the same file name (alone) or by a Renamed trailer. Trees only."""
    span = f"{base}..{head}" if base else head
    text = _git(checkout, "-c", "core.quotePath=false", "log", "--raw", "--no-renames", "--no-abbrev", "--reverse",
                f"--format=%x01%an%x02%(trailers:key={RENAMED},valueonly,separator=%x1f)"
                f"%x02%(trailers:key={RECOMPRESSED},valueonly,separator=%x1f)", span, "--", "*.png", "*.PNG")
    out = []
    for chunk in text.split("\x01")[1:]:
        head_line, _, body = chunk.partition("\n")
        author, renamed, same = (head_line.split("\x02") + ["", ""])[:3]
        told = dict(filter(None, map(rename_trailer, renamed.split("\x1f"))))
        same = {p.strip() for p in same.split("\x1f") if p.strip()} if author == BOT else set()
        adds, dels = {}, {}
        for line in body.splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) < 5 or not path.startswith(CARD_PLACES) or "Markers" in path.split("/"):
                continue
            status, old, new = parts[4][:1], parts[2], parts[3]
            if status == "M" and path not in same:
                out.append((old, new))
            elif status == "A":
                adds[path] = new
            elif status == "D":
                dels[path] = old
        moved = set(adds.values()) & set(dels.values())        # exact renames: no change
        adds = {p: b for p, b in adds.items() if b not in moved}
        for old_path, new_path in told.items():
            if old_path in dels and new_path in adds:
                out.append((dels.pop(old_path), adds.pop(new_path)))
        for stem in {_stem(p) for p in adds}:
            added = [p for p in adds if _stem(p) == stem]
            gone = [p for p in dels if _stem(p) == stem]
            if len(added) == 1 and len(gone) == 1:
                out.append((dels[gone[0]], adds[added[0]]))
    return list(dict.fromkeys(out))


# ========================
# Measuring
# ========================
class _Reader:
    """One `git cat-file --batch` of a store: blob bytes by sha, None when missing."""
    def __init__(self, store: str):
        self.proc = Popen(["git", "-C", store, "cat-file", "--batch"], stdin=PIPE, stdout=PIPE, env={**environ, **NO_LAZY})

    def read(self, sha: str) -> Optional[bytes]:
        self.proc.stdin.write(sha.encode() + b"\n")
        self.proc.stdin.flush()
        head = self.proc.stdout.readline().split()
        if len(head) != 3 or head[1] != b"blob":
            return None
        data = self.proc.stdout.read(int(head[2]))
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def _have(store: str, shas: list) -> set:
    done = run(["git", "-C", store, "cat-file", "--batch-check"], input="\n".join(shas).encode() + b"\n", capture_output=True,
               env={**environ, **NO_LAZY})
    return {line.split()[0] for line in done.stdout.decode().splitlines() if line.endswith(tuple("0123456789")) and " blob " in line}


def _auth(checkout: str) -> dict:
    """The checkout's http extraheaders (actions/checkout's token) as git's config environment: never on a command line."""
    done = run(["git", "-C", checkout, "config", "--get-regexp", r"^http\..*extraheader$"], capture_output=True, text=True)
    found = [line.split(" ", 1) for line in done.stdout.splitlines() if " " in line]
    env = {"GIT_CONFIG_COUNT": str(len(found))}
    for i, (name, value) in enumerate(found):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = name, value
    return env if found else {}


def _fetch(checkout: str, store: str, shas: list) -> None:
    """The blobs into a scratch store, from the checkout's origin (one fetch); a failure leaves them out."""
    run(["git", "init", "-q", "--bare", store], check=True, capture_output=True)
    url = run(["git", "-C", checkout, "remote", "get-url", "origin"], capture_output=True, text=True).stdout.strip()
    if not url:
        return
    run(["git", "-C", store, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags", "--no-write-fetch-head",
         "--recurse-submodules=no", "--filter=blob:none", url, *shas],
        capture_output=True, env={**environ, **NO_LAZY, **_auth(checkout)})


def measure(checkout: str, pairs: list, record: dict, limit: Optional[int] = None, deadline: Optional[float] = None,
            log=print) -> tuple:
    """
    Measure the pairs the record lacks (or has by another RULE), BATCH at a time: blobs from the checkout's store, else
    fetched into a scratch store dropped after its batch (FETCHES batches ahead). A picture that can't be read is
    recorded None (visible, not tried again); one not fetched stays unmeasured.

    Returns:
        (pairs measured, pairs left: over limit, past deadline or not fetched)
    """
    todo = [p for p in dict.fromkeys(map(tuple, pairs)) if not _measured(record, p)]
    left = todo[limit:] if limit is not None else []
    todo = todo[:limit] if limit is not None else todo
    if not todo:
        return 0, left
    done, missed = 0, 0
    chunks = iter([todo[i:i + BATCH] for i in range(0, len(todo), BATCH)])
    ahead = deque()
    fetches = ThreadPoolExecutor(FETCHES)

    def start(chunk: list) -> tuple:
        shas = sorted({b for p in chunk for b in p})
        here = _have(checkout, shas)
        tmp = TemporaryDirectory(prefix="png-changes-")
        store = join(tmp.name, "store.git")
        task = fetches.submit(_fetch, checkout, store, [s for s in shas if s not in here]) if len(here) < len(shas) else None
        return chunk, shas, here, tmp, store, task

    local_reader = _Reader(checkout)
    try:
        for i in range(0, len(todo), BATCH):
            if deadline is not None and time() > deadline:
                left = todo[i:] + left
                break
            while len(ahead) < FETCHES and (following := next(chunks, None)) is not None:
                ahead.append(start(following))
            chunk, shas, here, tmp, store, task = ahead.popleft()
            try:
                if task is not None:
                    task.result()
                fetched = _Reader(store) if isfile(join(store, "HEAD")) else None
                try:
                    data = {s: local_reader.read(s) if s in here else fetched.read(s) if fetched else None for s in shas}
                finally:
                    if fetched is not None:
                        fetched.close()
            finally:
                tmp.cleanup()

            def one(pair):
                a, b = data.get(pair[0]), data.get(pair[1])
                if a is None or b is None:
                    return pair, False
                try:
                    return pair, changed_share(a, b)
                except Exception as e:                  # an unreadable picture: visible
                    log(f"{key(*pair)}: not read ({type(e).__name__}: {e})")
                    return pair, None
            with ThreadPoolExecutor(WORKERS) as pool:
                for pair, share in pool.map(one, chunk):
                    if share is False:
                        missed += 1
                        left.append(pair)
                        continue
                    record["pairs"][key(*pair)] = [share, RULE]
                    done += 1
            data.clear()
            log(f"png changes: {done} measured, {missed} not fetched, {len(todo) - i - len(chunk)} to go in this run")
    finally:
        fetches.shutdown(cancel_futures=True)
        for *_, tmp, _store, _task in ahead:
            tmp.cleanup()
        local_reader.close()
    return done, left


def _stop(proc: Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()


class Beside:
    """
    measure() in a child process (python -m commands.png_changes measure), started before card_data's PSD reads and
    joined after them; a thread would turn the build's forked parsers off (card_data._forks). Nothing to measure: no
    process.
    """
    def __init__(self, checkout: str, pairs: list, record: dict, limit: Optional[int] = None):
        self.checkout, self.limit = checkout, limit
        self.todo = [p for p in dict.fromkeys(map(tuple, pairs)) if not _measured(record, p)]
        self.proc = self._end = self.tmp = None
        self.why = ""
        if not self.todo:
            return
        try:
            self.tmp = TemporaryDirectory(prefix="png-beside-")
            given, self.out, self.log = (join(self.tmp.name, n) for n in ("pairs.json", "out.json", "log.txt"))
            with open(given, "w", encoding="utf-8") as fh:
                fh.write(dumps([list(p) for p in self.todo]))
            args = [executable, "-m", "commands.png_changes", "measure", "--checkout", abspath(checkout), "--pairs",
                    given, "--out", self.out]
            with open(self.log, "wb") as fh:
                self.proc = Popen(args + (["--limit", str(limit)] if limit is not None else []), cwd=ROOT,
                                  stdin=DEVNULL, stdout=fh, stderr=STDOUT)
        except OSError as exc:                              # no fork (EAGAIN) or no disk: join measures them here
            self.why = f"not started beside the PSD reads ({exc.__class__.__name__})"
            if self.tmp is not None:
                self.tmp.cleanup()
            return
        self._end = finalize(self, _stop, self.proc)        # the build stopped early: so does it

    def stop(self) -> None:
        if self._end is not None:
            self._end()

    def join(self, record: dict, more: list = (), timeout: float = JOIN_WAIT, log=print) -> tuple:
        """
        Its measures into the record (timeout s waited at most), then the pairs of `more` (the run's own events) it did
        not cover measured here, first and with the whole limit, and its own after them when it failed, did not start
        or ran out of time (as before it ran beside).

        Returns:
            (pairs measured, pairs left), as measure()
        """
        got, own = None, []
        if self.proc is not None:
            waited = time()
            try:
                self.proc.wait(timeout)
            except TimeoutExpired:
                pass
            self.stop()
            with open(self.log, encoding="utf-8", errors="replace") as fh:
                for line in fh.read().splitlines():
                    log(line)
            if self.proc.returncode == 0 and isfile(self.out):
                with open(self.out, encoding="utf-8") as fh:
                    got = loads(fh.read())
                record["pairs"].update(got["pairs"])
                log(f"png changes: {got['done']} measured beside the PSD reads in {got['seconds']:.1f} s, "
                    f"{time() - waited:.1f} s waited for")
            else:
                own = self.todo
                log(f"png changes: not measured beside the PSD reads (exit {self.proc.returncode}): measured here")
            self.tmp.cleanup()
        elif self.todo:
            own = self.todo
            log(f"png changes: {self.why}: measured here")
        done, left = (got["done"], [tuple(p) for p in got["left"]]) if got else (0, [])
        more_done, more_left = measure(self.checkout, list(more) + own, record, limit=self.limit, log=log)
        left = [p for p in dict.fromkeys(left + more_left) if not _measured(record, p)]
        return done + more_done, left


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.png_changes", description="The card PNGs' picture changes.")
    sub_parsers = parser.add_subparsers(dest="command", required=True)
    back = sub_parsers.add_parser("backfill", help="measure every card PNG change in head's history the record lacks")
    back.add_argument("--checkout", required=True, help="a clone with main's history (blobless is enough)")
    back.add_argument("--db", required=True)
    back.add_argument("--head", default="HEAD")
    back.add_argument("--minutes", type=float, default=0, help="stop after this long (0: no limit); exit 3 with pairs left")
    back.add_argument("--events", action="store_true", help="also db/changelog's past events: their pairs first, then "
                      "marked minor as Card sync marks new ones (psd-tools: card_data)")
    side = sub_parsers.add_parser("measure", help="measure the pairs of a JSON file (card_data's Beside)")
    side.add_argument("--checkout", required=True)
    side.add_argument("--pairs", required=True, help="a JSON file: [[old blob, new blob], ...]")
    side.add_argument("--out", required=True, help="written when done: {done, left, pairs, seconds}")
    side.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "measure":
        with open(args.pairs, encoding="utf-8") as fh:
            pairs = loads(fh.read())
        record, began = {"rule": RULE, "pairs": {}, "pending": []}, time()
        done, left = measure(args.checkout, pairs, record, limit=args.limit)
        with open(args.out + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(dumps({"done": done, "left": [list(p) for p in left], "pairs": record["pairs"], "seconds": time() - began}))
        replace(args.out + ".tmp", args.out)
        return 0
    record = read_record(args.db)
    events, shown = [], {}
    if args.events:
        from commands.card_data import mark_past, past_pairs
        events, shown = past_pairs(args.checkout, args.db, args.head)
    pairs = card_pairs(args.checkout, args.head)
    deadline = time() + args.minutes * 60 if args.minutes else None
    done, left = 0, []
    try:
        done, left = measure(args.checkout, list(shown.values()) + pairs, record, deadline=deadline)
    finally:
        write_record(args.db, record)                   # what was measured stays, whatever stopped the run
    print(f"png changes: {len(pairs)} pairs in history, {done} measured now, {len(left)} left, {len(record['pairs'])} recorded.")
    if args.events:
        print(f"past events: {mark_past(args.db, events, shown, record)} of {len(events)} unmarked marked minor.")
    return EXIT_LEFT if left else 0


# ========================
# Module Exports
# ========================
__all__ = [
    'Beside',
    'RECORD',
    'RULE',
    'RUN_MAX',
    'PATCH',
    'TOLERANCE',
    'VISIBLE',
    'card_pairs',
    'changed_area',
    'changed_share',
    'key',
    'main',
    'measure',
    'read_record',
    'record_text',
    'share_of',
    'visible',
    'write_record',
]


if __name__ == "__main__":
    raise SystemExit(main())
