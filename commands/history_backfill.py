"""
One-time stat history backfill and archive completion (docs/NEXT_SESSION_PLAN.md step 1b). Dead code once
its merge has run in production: delete this module, history_backfill.json, its call in card_data.update() and
tests/test_history_backfill.py.

- build (the Mac, by hand): every stat-card PSD version in the card repo's git read by today's parser, the stat bars of
  consecutive versions compared on the PSDs' stored previews (a reading that changed while the bars printed the same is
  a misread, never a change), the cards that left the card folders with their fate and author; one reviewable file.
- merged (/update_stats, once, after recover): that file into old_stats, on a trial copy.
- rehearse: the same merge on copies of the pickles.

CLI (copies only, the fork venv):
  python -m commands.history_backfill build --git <full clone> --head origin/main --cache <jsonl> --out history_backfill.json
  python -m commands.history_backfill rehearse --file history_backfill.json --git <full clone> --pickles <dir> --out <dir>
"""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import BytesIO
from json import dumps, loads
from os import environ, makedirs
from os.path import abspath, dirname, exists, isfile, join, samefile
from posixpath import basename as path_base, dirname as path_dir, splitext
from re import compile as re_compile
from subprocess import run
from time import localtime, strftime, time
from typing import Optional

from global_config import AUTHORS_PATH, EDITS_PATH, ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR
from commands.card_db import card as from_record, record
from commands.card_edits import CardAuthors, CardEdits
from commands.card_history import (BOT_CHECKOUT, HIGH_STATS_LISTED, RETIRED, STAT_CAP, STAT_MAX, GitIndex, Refined,
                                   close, collapse, commits_of, date_archive, meta, recovered_before, start)
from commands.card_models import CardInfo, CardStats, card_name

# ========================
# Configuration
# ========================
FILE_NAME = "history_backfill.json"
FILE = join(BOT_CHECKOUT, FILE_NAME)          # what /update_stats merges once
FILE_ID = "backfill-1"
MARK = "backfill_id"                          # on the version records and archive tags the merge wrote: merged
RULES = 1                                     # cache lines read by other rules are read again
PX_DIFF, PX_SAME = 24, 1000                   # bars print the same: at most PX_SAME pixels moved by more than PX_DIFF
OFF_SCALE = 40                                # analytics: a total above it is no buff or nerf
STAT_PATH = re_compile(r"^(Creatures|Aux[il]+ary/(Minions|Creatures))/")
CARD_TOPS = ("Creatures", "Items", "Field", "N.M.E", "Auxiliary", "Auxillary")
NO_CARDS = ("Rulebook", "Rules", "Handbook", "Markers", "Field Cards", "MDW")
SPELLINGS = (("Auxillary/", "Auxiliary/"),)   # an old folder spelling, read as today's
# the pre-2025 template's four rows (bad64921 renamed them)
TEMPLATE = {"hp dark copy": "Spd Dark", "hp dark copy 2": "Atk Dark", "hp dark copy 3": "Def Dark"}
# owner-confirmed renames (2026-10-23) git can't tell: last path -> the card it became
CONFIRMED = {"Items/Support/2 Stars/Plan_Ahead.psd": "Tomorrow I'll",
             "Items/Heal/1 Stars/Nuts_and_Bolts.psd": "Bolts and Screws",
             "Items/Heal/2 Stars/Pizza.psd": "Pepperoni Pizza"}
KINDS = {"aux item": "item"}                  # a qualified key's kind: "Pizza (item)"
DRAWN = ("pixel", "smartobject", "group", "type", "shape")   # psd_parser.DRAWN_LAYER_KINDS
STAT_OF = {"hp": "hp", "def": "def", "atk": "atk", "spd": "spd", "health": "hp"}
NO_LAZY = {"GIT_NO_LAZY_FETCH": "1"}
WORKERS = 4
FAR = 9e12                                    # an open end


# ========================
# Git
# ========================
def _git(repo: str, *args: str, data: Optional[bytes] = None) -> bytes:
    """git -C repo ..., never fetching a blob."""
    return run(["git", "-C", repo, *args], input=data, capture_output=True, check=True, env={**environ, **NO_LAZY}).stdout


def _text(repo: str, head: str, path: str) -> Optional[str]:
    """A file at head, else None."""
    done = run(["git", "-C", repo, "show", f"{head}:{path}"], capture_output=True, env={**environ, **NO_LAZY})
    return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None


def _missing(repo: str, blobs) -> list:
    """The blobs the clone lacks."""
    wanted = sorted({b for b in blobs if b})
    out = _git(repo, "cat-file", "--batch-check", data="".join(b + "\n" for b in wanted).encode()).decode()
    return [line.split()[0] for line in out.splitlines() if line.endswith(" missing")]


def today(path: str) -> str:
    """A path in today's folder spelling (Auxillary/ -> Auxiliary/)."""
    for old, new in SPELLINGS:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def is_card(path: str) -> bool:
    """A card PSD: under a card folder, not a rule page, marker, template or MDW."""
    parts = path.split("/")
    return path.lower().endswith(".psd") and parts[0] in CARD_TOPS and not any(p in NO_CARDS for p in parts)


def is_stat(path: str) -> bool:
    return is_card(path) and bool(STAT_PATH.match(path))


def retired(path: str) -> bool:
    return any(folder in RETIRED for folder in path.split("/"))


def key(name: str) -> str:
    return CardAuthors.key(name)


# ========================
# Card repo history
# ========================
@dataclass
class History:
    """GitIndex plus each path's events (oldest first: (Commit, blob or None for a deletion)) and each commit's adds
    (paths it put where none was)."""
    index: GitIndex
    events: dict = field(default_factory=dict)
    added: dict = field(default_factory=dict)

    @classmethod
    def of(cls, index: GitIndex) -> "History":
        out = cls(index)
        for path in set(index.at_path) | set(index.gone_at):
            got = [(c, b) for c, b in index.at_path.get(path, [])] + [(c, None) for c in index.gone_at.get(path, [])]
            got.sort(key=lambda e: (-e[0].order, e[1] is not None))    # oldest first; a delete before a put
            out.events[path] = got
            present = False
            for commit, blob in got:
                if blob is not None and not present:
                    out.added.setdefault(commit.sha, []).append(path)
                present = blob is not None
        return out

    def alive(self, path: str) -> bool:
        got = self.events.get(path) or []
        return bool(got) and got[len(got) - 1][1] is not None

    def gone(self, path: str):
        """The commit that deleted path last (None when alive)."""
        return None if self.alive(path) else self.index.gone_at[path][0]

    def deleted_blob(self, path: str) -> Optional[str]:
        """The bytes path held when it was deleted last."""
        got = self.events.get(path) or []
        blobs = [b for _, b in got if b is not None]
        return blobs[len(blobs) - 1] if blobs else None

    def spans(self, path: str) -> list:
        """[(from, to)] times path held a file (to FAR when alive)."""
        out, begin = [], None
        for commit, blob in self.events.get(path) or []:
            if blob is not None and begin is None:
                begin = commit.at
            elif blob is None and begin is not None:
                out.append((begin, commit.at))
                begin = None
        if begin is not None:
            out.append((begin, FAR))
        return out

    def first_put(self, path: str):
        got = self.events.get(path) or []
        return got[0][0] if got else None


@dataclass
class Lineage:
    """One card's PSD paths: name (the database's; a qualified key for another card of a live name) and its versions
    [(Commit, blob, path)], oldest first, the same bytes again counted once (a move is no version)."""
    name: str
    paths: list
    title: Optional[str] = None               # the plain name of a qualified key
    alive: bool = False
    versions: list = field(default_factory=list)


def _components(hist: History, paths: list) -> list:
    """paths joined by moves (a path deleted by the commit that put another): [[path]]."""
    parent = {p: p for p in paths}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for p in paths:
        for cut in hist.index.gone_at.get(p, []):
            for q in hist.added.get(cut.sha, []):
                if q != p and q in parent:
                    parent[find(p)] = find(q)
    groups: dict = {}
    for p in paths:
        groups.setdefault(find(p), []).append(p)
    return list(groups.values())


def _overlap(hist: History, a: list, b: list) -> bool:
    return any(x0 < y1 and y0 < x1 for p in a for x0, x1 in hist.spans(p) for q in b for y0, y1 in hist.spans(q))


def _versions(hist: History, paths: list) -> list:
    puts = sorted(((c, b, p) for p in paths for c, b in hist.index.at_path.get(p, [])), key=lambda x: -x[0].order)
    out, last = [], None
    for commit, blob, path in puts:
        if blob != last:
            out.append((commit, blob, path))
        last = blob
    return out


def lineages(hist: History) -> tuple:
    """
    Every card name's lineage: its card-folder paths by name (any case, _ and spaces alike), joined by moves. A path
    group that coexisted with the name's main one (the live one, else the last gone) is another card: kept apart under
    a qualified key ("Pizza (item)"); the others are the same card at other times.

    Returns:
        ({name: Lineage} main lineages, [Lineage] other cards)
    """
    by: dict = {}
    for path in hist.events:
        if is_card(path):
            by.setdefault(key(card_name(path)), []).append(path)
    mains, others = {}, []
    for paths in by.values():
        groups = _components(hist, sorted(paths))
        live = [g for g in groups if any(hist.alive(p) for p in g)]
        if live:
            main = max(live, key=lambda g: max(hist.first_put(p).at for p in g if hist.alive(p)))
        else:
            main = max(groups, key=lambda g: max(hist.gone(p).at for p in g))
        apart = [g for g in groups if g is not main and _overlap(hist, main, g)]
        joined = [p for g in groups if g is main or not _overlap(hist, main, g) for p in g]
        alive = any(hist.alive(p) for p in joined)
        newest = max(joined, key=lambda p: (hist.alive(p), max(c.at for c, _ in hist.events[p])))
        name = card_name(newest)
        mains[name] = Lineage(name, sorted(joined), None, alive, _versions(hist, joined))
        for group in apart:
            last = max(group, key=lambda p: max(c.at for c, _ in hist.events[p]))
            others.append(Lineage(card_name(last), sorted(group), None, False, _versions(hist, group)))
    return (mains, others)


# ========================
# Fates (which cards left, and how)
# ========================
@dataclass
class Fate:
    kind: str                                 # renamed, retired, deleted
    commit: object                            # the deleting commit (index Commit)
    target: Optional[str] = None              # renamed: the current card's name; retired: the path in MDW
    how: str = ""                             # exact, same folder, same file, confirmed, ...


def _step(hist: History, path: str, live: dict) -> tuple:
    """Where path went at its last deletion: (kind, the path or name it went to, how)."""
    index = hist.index
    cut = hist.gone(path)
    if cut is None:
        return ("live", path, "")
    moved = index.renames.get(path)
    if moved is not None and moved[1] is cut:
        return ("retired" if retired(moved[0]) else "renamed", moved[0], "exact")
    adds = hist.added.get(cut.sha, [])
    same = [p for p in adds if retired(p) and path_base(p) == path_base(path)]
    if same:
        return ("retired", same[0], "same file")
    if path in CONFIRMED:
        return ("renamed", CONFIRMED[path], "confirmed")
    folder = path_dir(path)
    targets = {r[0] for r in index.renames.values() if r[1] is cut}
    new = [p for p in adds if is_card(p) and path_dir(p) == folder and p not in targets
           and not any(key(card_name(q)) == key(card_name(p)) and hist.first_put(q).order > cut.order
                       for q in hist.events if q != p and is_card(q))]
    moved = {path_base(p) for p in adds if retired(p)}
    gone = [p for p, cuts in index.gone_at.items() if is_card(p) and path_dir(p) == folder and any(c is cut for c in cuts)
            and not (index.renames.get(p) and index.renames[p][1] is cut) and path_base(p) not in moved]
    if new and len(new) == len(gone):
        pairs, left = {}, list(new)
        for old in sorted(gone, key=lambda g: -max(SequenceMatcher(None, card_name(g).lower(),
                                                                   card_name(n).lower()).ratio() for n in new)):
            best = max(left, key=lambda n: SequenceMatcher(None, card_name(old).lower(), card_name(n).lower()).ratio())
            pairs[old] = best
            left.remove(best)
        return ("renamed", pairs[path], "same folder")
    return ("deleted", None, "")


def fate(hist: History, path: str, live: dict) -> Fate:
    """
    A gone card's fate from the commit that deleted its last path: retired (moved_to its path in MDW: the commit put
    the file there, or an exact rename did), renamed (renamed_to the live card a chain of renames ends at: exact
    renames, an owner-confirmed one, or the commit adding as many new card names in that folder as it deleted, paired
    by name), else deleted.

    Args:
        live: {key: live card name}
    """
    cut = hist.gone(path)
    kind, where, how = _step(hist, path, live)
    seen = {path}
    while kind == "renamed" and how != "confirmed" and key(card_name(where)) not in live and where not in seen:
        seen.add(where)
        kind, where, step_how = _step(hist, where, live)
        how = f"{how}, {step_how}" if step_how else how
        if kind == "live":
            break
    if kind == "renamed":
        target = live.get(key(where if how.endswith("confirmed") else card_name(where)))
        return Fate("renamed", cut, target, how) if target else Fate("deleted", cut, None, how + ", ends at no card")
    if kind == "retired":
        return Fate("retired", cut, where, how)
    return Fate("deleted", cut, None, how)


# ========================
# Reading a PSD version (the workers)
# ========================
_PARSER = None


def _init(vocab: tuple) -> None:
    """A worker's parser: today's code and icon names, an empty database (no author or series carried)."""
    global _PARSER
    from commands.card_models import StatsDatabase
    from commands.psd_parser import IconVocabulary, PSDParser
    all_types, stars, manifest = vocab
    _PARSER = PSDParser(list(all_types), StatsDatabase(), IconVocabulary.build(list(all_types), list(stars), manifest))


def _open(repo: str, blob: str):
    from psd_tools import PSDImage
    return PSDImage.open(BytesIO(_git(repo, "cat-file", "blob", blob)))


def template(psd) -> bool:
    """The pre-2025 template ("Darks" holding "Hp Dark copy", "... copy 2", "... copy 3"): its rows renamed in memory to
    today's (bad64921), so the parser reads them by position. Whether it applied."""
    hit = False
    for group in psd.descendants():
        if not (group.is_group() and group.name.strip().lower() == "darks"):
            continue
        rows = [c for c in group if c.is_group() and c.name.strip().lower() in TEMPLATE]
        if len(rows) >= 2:
            for row in rows:
                row.name = TEMPLATE[row.name.strip().lower()]
            hit = True
    return hit


def hidden_rows(parser, psd) -> list:
    """Stats whose digit layers are drawn cells (pixel or smart-object layers named 1-10), all hidden: the parser's
    "every digit hidden -> 10" is meant for the darkening template only, so such a stat is unread."""
    seen: dict = {}
    for layer in psd.descendants():
        name = layer.name.strip()
        if layer.kind == "type" or not name.isdigit() or not parser._is_stat_layer(layer):
            continue
        if layer.is_group() and (len(layer) == 0 or any(child.name.isdigit() for child in layer)):
            continue
        names, cur = [], layer.parent
        while cur is not None and len(names) < 2 and hasattr(cur, "name"):
            names.append(cur.name.lower())
            cur = getattr(cur, "parent", None)
        stat = next((STAT_OF[k] for k in STAT_OF if any(k in n for n in names)), None)
        if stat:
            seen.setdefault(stat, []).append((layer.is_visible(), layer.kind))
    return sorted(s for s, found in seen.items() if not any(v for v, _ in found) and all(k in DRAWN for _, k in found))


def bar_boxes(psd) -> list:
    """The stat bars' boxes: groups named "...bars" or "...dark...", each stat digit layer and its group (never the
    canvas)."""
    out = []
    for layer in psd.descendants():
        name = layer.name.lower()
        if layer.is_group() and ("bars" in name or "dark" in name):
            out.append(layer.bbox)
        if layer.name.strip().isdigit():
            cur, names = layer.parent, []
            while len(names) < 3 and cur is not None and cur is not psd and hasattr(cur, "name"):
                names.append(cur.name.lower())
                cur = getattr(cur, "parent", None)
            if any(k in n for n in names for k in ("dark", "bars")):
                out.append(layer.bbox)
                if layer.parent is not None and layer.parent is not psd:
                    out.append(layer.parent.bbox)
    return sorted({tuple(int(v) for v in b) for b in out if b and b[2] > b[0] and b[3] > b[1]})


def _preview(psd):
    import numpy as np
    try:
        img = psd.topil() if psd.has_preview() else None
    except Exception:                         # noqa: BLE001 - no usable preview: the render compares
        img = None
    return np.asarray(img.convert("RGB")).astype(np.int16) if img is not None else None


def _region(boxes: list, shape: tuple) -> Optional[tuple]:
    if not boxes:
        return None
    x0, y0 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
    x1, y1 = min(shape[1], max(b[2] for b in boxes)), min(shape[0], max(b[3] for b in boxes))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def bars_moved(a, b, boxes: list) -> Optional[int]:
    """Pixels of the stat-bar region (boxes: both versions') that moved by more than PX_DIFF; None when the images'
    sizes differ; 0 with no region."""
    import numpy as np
    if a.shape != b.shape:
        return None
    box = _region(boxes, a.shape)
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return int((np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).max(axis=2) > PX_DIFF).sum())


def _render(psd, box: tuple):
    """A region rendered from the layers (a PSD without a stored preview)."""
    import numpy as np
    img = psd.composite(viewport=box, ignore_preview=True, apply_icc=False)
    return np.asarray(img.convert("RGB")).astype(np.int16)


def read_psd(parser, psd, path: str) -> dict:
    """One version's reading: line [hp, def, atk, spd] (creature 1's), creatures, how it was read, hidden drawn rows,
    card type and stars, the bars' boxes."""
    how = "template" if template(psd) else "parser"
    out = {"how": how, "hidden": hidden_rows(parser, psd), "boxes": [list(b) for b in bar_boxes(psd)]}
    try:
        card = parser.parse(psd, today(path))
    except ValueError as e:                   # a folder the classifier refuses
        return {**out, "read": None, "err": f"classifier: {e}"}
    s = card.stats
    out.update({"read": [s.hp, s.defense, s.attack, s.speed] if s is not None else None,
                "creatures": [[c.hp, c.defense, c.attack, c.speed] for c in card.creatures or []],
                "type": card.card_type, "stars": card.stars})
    return out


def _task_lineage(repo: str, items: list, have_reads: set, have_pairs: set) -> list:
    """A worker's lineage: cache lines for the readings and print pairs not cached (each version opened at most once,
    the one before it again only when a pair needs it)."""
    out, prev = [], None                      # prev: (blob, psd or None when not opened, preview, boxes)
    for blob, path in items:
        want_read = (blob, path) not in have_reads
        want_pair = prev is not None and (prev[0], blob) not in have_pairs
        if not want_read and not want_pair:
            prev = (blob, None, None, None)
            continue
        t = time()
        try:
            if want_pair and prev[1] is None:
                old = _open(repo, prev[0])
                template(old)
                prev = (prev[0], old, _preview(old), bar_boxes(old))
            psd = _open(repo, blob)
            reading = read_psd(_PARSER, psd, path)
            image = _preview(psd)
        except Exception as e:                # noqa: BLE001 - one unreadable version is one line
            out.append({"kind": "read", "blob": blob, "path": path, "rules": RULES, "read": None,
                        "err": f"{type(e).__name__}: {str(e)[:160]}"})
            prev = None
            continue
        if want_read:
            out.append({"kind": "read", "blob": blob, "path": path, "rules": RULES, "s": round(time() - t, 2), **reading})
        if want_pair:
            boxes = [tuple(b) for b in reading["boxes"]] + [tuple(b) for b in prev[3]]
            a, b, how = prev[2], image, "preview"
            if a is None or b is None:        # no stored preview: that region rendered from the layers
                box = _region(boxes, (psd.height, psd.width))
                a, b, how = (_render(prev[1], box), _render(psd, box), "render") if box else (None, None, "render")
            px = (0 if how == "render" and a is None else
                  bars_moved(a, b, [] if how == "render" else boxes))
            out.append({"kind": "pair", "a": prev[0], "b": blob, "rules": RULES, "px": px, "how": how})
        prev = (blob, psd, image, reading["boxes"])
    return out


def _task_card(repo: str, blob: str, path: str) -> list:
    """A worker's archive card: today's parser on the bytes git deleted, at today's folder spelling."""
    try:
        card = _PARSER.parse(_open(repo, blob), today(path))
        card.path = path
        return [{"kind": "card", "blob": blob, "path": path, "rules": RULES, "card": record(card)}]
    except Exception as e:                    # noqa: BLE001
        return [{"kind": "card", "blob": blob, "path": path, "rules": RULES, "card": None,
                 "err": f"{type(e).__name__}: {str(e)[:160]}"}]


def _run_task(task: tuple) -> list:
    kind, *args = task
    return _task_lineage(*args) if kind == "lineage" else _task_card(*args)


# ========================
# Lines, print runs, segments
# ========================
def status(reading: dict, allowed: bool) -> Optional[str]:
    """"full", "partial" (values and -1s: a bar the card lacks) or None: unread (no reading, a hidden drawn-cell row,
    a value out of range, the all-default 10/10/10/10, all -1)."""
    line = reading.get("read")
    if not line or reading.get("err") or reading.get("hidden"):
        return None
    if line == [10, 10, 10, 10] or all(v == -1 for v in line):
        return None
    if any(not isinstance(v, int) or v < -1 or v > (STAT_CAP if allowed else STAT_MAX) for v in line):
        return None
    return "partial" if -1 in line else "full"


def kind_of(a: Optional[list], b: list) -> str:
    """A stat change in analytics' words: buff, nerf or shift; partial when either line isn't full; off-scale."""
    if not a or -1 in a or -1 in b:
        return "partial"
    if sum(a) > OFF_SCALE or sum(b) > OFF_SCALE:
        return "off-scale"
    return "buff" if sum(b) > sum(a) else "nerf" if sum(b) < sum(a) else "shift"


def segments(readings: list, allowed: bool) -> list:
    """
    The stat line over time (the misread rule): consecutive versions whose bars printed the same are one print run,
    whose line is its newest reading with a line; a run without one takes the line before it (the first: the one after
    it). Consecutive runs with the same line are one segment; each later segment is a change, dated at its first
    version. So a line changes only where the PSD's printed bars changed.

    Args:
        readings: [{at, commit, blob, path, read, creatures, print, ...}] oldest first

    Returns:
        [{from, commit, blob, path, line, creatures, change?}]
    """
    runs: list = []
    for i, reading in enumerate(readings):
        if runs and reading.get("print") == "same":
            runs[len(runs) - 1].append(i)
        else:
            runs.append([i])
    lines = []
    for run_ in runs:
        known = [i for i in run_ if status(readings[i], allowed)]
        lines.append((readings[known[len(known) - 1]]["read"], readings[known[len(known) - 1]].get("creatures") or [])
                     if known else None)
    first = next((x for x in lines if x is not None), None)
    if first is None:
        return []
    segs: list = []
    prev = None
    for run_, found in zip(runs, lines):
        use = found or prev or first
        prev = use
        line, creatures = use
        if segs and segs[len(segs) - 1]["line"] == line:
            if found:
                segs[len(segs) - 1]["creatures"] = creatures
            continue
        r = readings[run_[0]]
        seg = {"from": r["at"], "commit": r["commit"], "blob": r["blob"], "path": r["path"], "line": line,
               "creatures": creatures}
        if segs:
            seg["change"] = kind_of(segs[len(segs) - 1]["line"], line)
        segs.append(seg)
    return segs


# ========================
# Build
# ========================
def vocabulary(repo: str, head: str) -> tuple:
    """(Types/ names, Types/Stars/ names, the font manifest) at head, as card_data reads the tree."""
    paths = _git(repo, "ls-tree", "-r", "--name-only", "-z", head).decode().split("\0")
    types = [splitext(path_base(p))[0].lower() for p in paths if p.startswith(TYPES_DIR) and "Stars" not in p and "." in p]
    stars = [splitext(path_base(p))[0].lower().replace(" ", "_") for p in paths
             if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
    manifest = _text(repo, head, ICON_MANIFEST_PATH)
    return (types, stars, loads(manifest) if manifest else None)


def _load_cache(path: str) -> dict:
    found = {"read": {}, "pair": {}, "card": {}}
    if not exists(path):
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = loads(line)
            except ValueError:
                continue                      # a line cut by an interruption
            if rec.get("rules") != RULES:
                continue
            k = (rec["a"], rec["b"]) if rec["kind"] == "pair" else (rec["blob"], rec["path"])
            found[rec["kind"]][k] = rec
    return found


def allows(name: str, live: bool, edits: Optional[CardEdits]) -> bool:
    """card_history.allows_high from git: no live card, allow_high_stats in Card Edits.json, the retired list."""
    entry = (edits.cards.get(key(name)) or {}) if edits is not None else {}
    return not live or bool(entry.get("allow_high_stats")) or name in HIGH_STATS_LISTED


def _print_of(cache: dict, prev: Optional[dict], reading: dict) -> tuple:
    """(verdict, px) of a version against the one before it."""
    if prev is None or prev.get("err"):
        return ("first", None)
    pair = cache["pair"].get((prev["blob"], reading["blob"]))
    if pair is None or pair.get("px") is None:
        return ("resized" if pair is not None else "unknown", None)
    pre = "rendered-" if pair.get("how") == "render" else ""
    return (pre + ("same" if pair["px"] <= PX_SAME else "changed"), pair["px"])


def readings_of(line: Lineage, cache: dict) -> list:
    out, prev = [], None
    for commit, blob, path in line.versions:
        rec = cache["read"].get((blob, path)) or {"read": None, "err": "not read"}
        verdict, px = _print_of(cache, prev, {"blob": blob})
        r = {"at": commit.at, "commit": commit.sha, "blob": blob, "path": path, "read": rec.get("read"),
             "how": rec.get("how", "parser"), "print": verdict}
        if px is not None and verdict != "first":
            r["px"] = px
        for k in ("creatures", "hidden", "err", "type", "stars"):
            if rec.get(k):
                r[k] = rec[k]
        if verdict == "rendered-same":
            r["print"] = "same"
            r["rendered"] = True
        out.append(r)
        prev = {"blob": blob, "err": rec.get("err")}
    return out


def credit(hist: History, authors: Optional[CardAuthors], name: str, paths: list, fate_: Optional[Fate]) -> list:
    """[author, rule] of an archived name: the card it became (renamed), its own Card Authors.txt line, else the first
    author of the oldest path of its exact renames (.mailmap'd), each through [people]."""
    if authors is None:
        authors = CardAuthors()
    if fate_ is not None and fate_.kind == "renamed" and authors.cards.get(key(fate_.target)):
        return [authors.resolve(fate_.target, None), "the card it became"]
    if authors.cards.get(key(name)):
        return [authors.resolve(name, None), "Card Authors.txt"]
    oldest = max(paths, key=lambda p: hist.first_put(p).order)
    origin = hist.index.origin(oldest)
    first = hist.index.first.get(origin) or hist.index.first.get(oldest)
    who = first.author if first else None
    return [authors.people.get((who or "").lower(), who) if who else None, f"first git author of {origin}"]


def archive_candidates(hist: History, mains: dict, others: list) -> list:
    """
    The cards that left the card folders: (name, title, Lineage, Fate). Left out: a name that only passed through (an
    exact rename in and out, one blob: the chain's archive covers it, Pandeity), a stray copy (deleted, every blob also
    another card's: Soul_Master - Copy), another card of a live name unless renamed or retired (Slugcat's 3 Stars copy).
    """
    live = {key(n): n for n, line in mains.items() if line.alive}
    blobs_of: dict = {}
    for name, line in mains.items():
        for _, blob, _ in line.versions:
            blobs_of.setdefault(blob, set()).add(name)
    out = []
    for line, other in [(line, False) for line in mains.values() if not line.alive] + [(line, True) for line in others]:
        last = max(line.paths, key=lambda p: hist.gone(p).at if hist.gone(p) else FAR)
        found = fate(hist, last, live)
        index = hist.index
        if not other and index.renamed_from.get(last) and found.how.startswith("exact") \
                and len({b for c, b in index.at_path.get(last, [])}) == 1 and len(line.paths) == 1:
            continue                          # passed through
        if found.kind == "deleted" and line.versions and blobs_of.get(line.versions[0][1], set()) - {line.name}:
            continue                          # a stray copy: added with another card's bytes
        if other and found.kind == "deleted":
            continue
        name, title = line.name, None
        if other:
            from commands.psd_parser import CardClassifier
            kind = CardClassifier.classify(today(last)).get("type", "card")
            name, title = f"{line.name} ({KINDS.get(kind, kind)})", line.name
        out.append((name, title, line, found, last))
    return out


def _archive_record(hist: History, name: str, title: Optional[str], last: str, found: Fate, card: dict,
                    live: dict) -> dict:
    """The archived record of a candidate: today's parse of the deleted bytes, its version from their intro, closed at
    the tag; the tag dated as date_archive would (so later cleanups leave it)."""
    index = hist.index
    info = from_record(card)
    info.name = name
    if title:
        info.display_name = title
    blob = hist.deleted_blob(last)
    intro = index.blob_since(blob, last) or index.blob_intro(blob)
    info.blob_sha, info.path = blob, last
    info.timestamp = intro.at if intro else found.commit.at
    info.version = {**meta(intro, info.timestamp, "psd"), MARK: FILE_ID}
    tag = {"at": found.commit.at, "path": last, "blob": blob, "commit": found.commit.sha}
    if found.kind == "renamed":
        tag["renamed_to"] = found.target
    elif found.kind == "retired":
        tag["moved_to"] = found.target
    info.archived = tag
    date_archive(dict.fromkeys(live.values()), {name: [info]}, index, Refined(), backfill=False)
    close(info, index, tag["at"], index.by_sha.get(tag["commit"]))
    tag[MARK] = FILE_ID
    return record(info)


def build(repo: str, head: str, cache_path: str, out: str, workers: int = WORKERS) -> int:
    """The backfill file from git alone (no pickle, no network). Returns the exit code."""
    t0 = time()
    head_sha = _git(repo, "rev-parse", head).decode().strip()
    index = GitIndex.from_git(repo, head_sha)
    hist = History.of(index)
    mains, others = lineages(hist)
    live = {key(n): n for n, line in mains.items() if line.alive}
    authors_text, edits_text = _text(repo, head_sha, AUTHORS_PATH), _text(repo, head_sha, EDITS_PATH)
    authors = CardAuthors.parse(authors_text) if authors_text else None
    edits = CardEdits.parse(edits_text) if edits_text else None
    stat = {n: line for n, line in mains.items() if any(is_stat(p) for p in line.paths)}
    candidates = archive_candidates(hist, mains, others)
    for name, title, line, found, last in candidates:
        if title and any(is_stat(p) for p in line.paths):
            stat[name] = Lineage(name, line.paths, title, False, line.versions)
    wanted = [b for line in stat.values() for _, b, _ in line.versions]
    wanted += [hist.deleted_blob(last) for _, _, _, _, last in candidates]
    missing = _missing(repo, wanted)
    if missing:
        where = join(dirname(abspath(cache_path)), "missing.txt")
        with open(where, "w", encoding="utf-8") as fh:
            fh.write("".join(b + "\n" for b in missing))
        print(f"{len(missing)} blobs are not in {repo} (ids in {where}). Fetch them, then build again:\n"
              f"  split -l 200 {where} batch_ && for b in batch_*; do git -C {repo} -c fetch.negotiationAlgorithm=noop "
              f"fetch origin --no-tags --no-write-fetch-head --recurse-submodules=no --filter=blob:none --stdin < $b; done")
        return 1
    cache = _load_cache(cache_path)
    tasks = []
    for line in stat.values():
        items = [(b, p) for _, b, p in line.versions]
        pairs = list(zip(items, items[1:]))
        if any(i not in cache["read"] for i in items) or any((a[0], b[0]) not in cache["pair"] for a, b in pairs):
            tasks.append(("lineage", repo, items, set(cache["read"]), set(cache["pair"])))
    for _, _, _, _, last in candidates:
        blob = hist.deleted_blob(last)
        if (blob, last) not in cache["card"]:
            tasks.append(("card", repo, blob, last))
    if tasks:
        print(f"Reading {len(tasks)} lineages and cards with {workers} workers ...", flush=True)
        done = 0
        with open(cache_path, "a", encoding="utf-8") as fh, \
                ProcessPoolExecutor(workers, initializer=_init, initargs=(vocabulary(repo, head_sha),)) as pool:
            for lines in pool.map(_run_task, tasks):
                for rec in lines:
                    fh.write(dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)} ({time() - t0:.0f} s)", flush=True)
        cache = _load_cache(cache_path)
    data = assemble(hist, stat, candidates, cache, authors, edits, live)
    bot_head = run(["git", "-C", BOT_CHECKOUT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    data["made"].update({"card_head": head_sha, "bot_head": bot_head})
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(file_text(data))
    for line in summary(data):
        print(line)
    print(f"Wrote {out} ({time() - t0:.0f} s).")
    return 0


def assemble(hist: History, stat: dict, candidates: list, cache: dict, authors: Optional[CardAuthors],
             edits: Optional[CardEdits], live: dict) -> dict:
    """The file's data from the lineages, the cache and the candidates."""
    cards, changes, counts = {}, [], Counter()
    for name in sorted(stat):
        line = stat[name]
        allowed = allows(name, line.alive, edits)
        readings = readings_of(line, cache)
        segs = segments(readings, allowed)
        for r in readings:
            counts["readings"] += 1
            counts[status(r, allowed) or "unread"] += 1
            counts["template"] += r["how"] == "template"
        for a, b in zip(readings, readings[1:]):              # the template read checks itself
            if a["how"] == "template" and b["how"] == "parser" and b["print"] == "same":
                same = status(a, allowed) and a["read"] == b["read"]
                counts["template agrees" if same else "template disagrees"] += 1
            if b["print"] == "same" and a.get("read") != b.get("read") and status(a, allowed) and status(b, allowed):
                counts["misread pairs"] += 1
        entry = {"paths": line.paths, "allowed_high": allowed, "readings": readings, "segments": segs}
        if line.title:
            entry["title"] = line.title
        cards[name] = entry
        for seg in segs[1:]:
            changes.append([strftime("%Y-%m-%d", localtime(seg["from"])), name,
                            next(s["line"] for s in reversed(segs) if s["from"] < seg["from"]), seg["line"],
                            seg["change"]])
    changes.sort()
    records, credits = {}, {}
    for name, title, line, found, last in candidates:
        got = cache["card"].get((hist.deleted_blob(last), last)) or {}
        entry = {"fate": found.kind, "how": found.how, "path": last}
        if got.get("card"):
            entry["record"] = _archive_record(hist, name, title, last, found, got["card"], live)
        else:
            entry["error"] = got.get("err", "not read")
        records[name] = entry
        credits[name] = credit(hist, authors, title or name, line.paths, found)
    counts["cards"] = len(cards)
    return {"id": FILE_ID, "made": {"parser": _parser_version(), "px": PX_SAME, "rules": RULES, **dict(counts)},
            "cards": cards, "changes": changes, "archive": {"cards": records, "authors": credits}}


def _parser_version() -> int:
    from commands.psd_parser import PARSER_VERSION
    return PARSER_VERSION


def summary(data: dict) -> list:
    made = data["made"]
    kinds = Counter(c[4] for c in data["changes"])
    fates = Counter(e["fate"] for e in data["archive"]["cards"].values())
    return [f"{made.get('readings', 0)} PSD versions of {made.get('cards', 0)} cards: {made.get('full', 0)} full lines, "
            f"{made.get('partial', 0)} partial, {made.get('unread', 0)} unread; {made.get('template', 0)} by the old "
            f"template's positions (agrees {made.get('template agrees', 0)}, disagrees {made.get('template disagrees', 0)})",
            f"Misread pairs (read differently, printed the same): {made.get('misread pairs', 0)}",
            f"Stat changes: {len(data['changes'])} ({', '.join(f'{n} {k}' for k, n in sorted(kinds.items()))}) on "
            f"{len({c[1] for c in data['changes']})} cards",
            f"Archive candidates: {len(data['archive']['cards'])} ({', '.join(f'{n} {k}' for k, n in sorted(fates.items()))})"]


def file_text(data: dict) -> str:
    """The file, one reading, segment, change and archive record per line (keys sorted)."""
    def one(value) -> str:
        return dumps(value, sort_keys=True, ensure_ascii=False)
    out = ["{", '"archive": {', '"authors": {']
    names = sorted(data["archive"]["authors"])
    out += [f"{one(n)}: {one(data['archive']['authors'][n])}{',' if i + 1 < len(names) else ''}" for i, n in enumerate(names)]
    out += ["},", '"cards": {']
    names = sorted(data["archive"]["cards"])
    out += [f"{one(n)}: {one(data['archive']['cards'][n])}{',' if i + 1 < len(names) else ''}" for i, n in enumerate(names)]
    out += ["}", "},", '"cards": {']
    names = sorted(data["cards"])
    for i, name in enumerate(names):
        entry = data["cards"][name]
        head = {k: v for k, v in entry.items() if k not in ("readings", "segments")}
        out.append(f"{one(name)}: {one(head)[:-1]}, \"readings\": [")
        out += [one(r) + ("," if j + 1 < len(entry["readings"]) else "") for j, r in enumerate(entry["readings"])]
        out.append('], "segments": [')
        out += [one(s) + ("," if j + 1 < len(entry["segments"]) else "") for j, s in enumerate(entry["segments"])]
        out.append("]}" + ("," if i + 1 < len(names) else ""))
    out += ["},", '"changes": [']
    out += [one(c) + ("," if i + 1 < len(data["changes"]) else "") for i, c in enumerate(data["changes"])]
    out += ["],", f'"id": {one(data["id"])},', f'"made": {one(data["made"])}', "}"]
    return "\n".join(out) + "\n"


# ========================
# Merge
# ========================
@dataclass
class Merged:
    """What the merge changed."""
    made: dict = field(default_factory=dict)
    corrected: list = field(default_factory=list)         # names, one per version corrected in place
    neutral: list = field(default_factory=list)           # names of unmatched versions aligned to a neighbour
    unmatched: list = field(default_factory=list)         # (name, start) of versions no reading matched
    added: list = field(default_factory=list)             # (name, start, line) of stat versions added
    folded: list = field(default_factory=list)            # names of versions collapse folded
    earlier: list = field(default_factory=list)           # (name, current?) starts moved earlier
    later: list = field(default_factory=list)             # (name, current?) starts moved later
    current: list = field(default_factory=list)           # (name, stored line, the PSD's) current cards that differ
    tags_moved: list = field(default_factory=list)        # archived records whose tag went to a newer line
    archived: list = field(default_factory=list)          # (name, fate) records added
    retagged: list = field(default_factory=list)          # (name, renamed_to or moved_to) retagged
    authors: int = 0                                      # archived records with an author
    authors_set: list = field(default_factory=list)       # names whose author was set or changed
    balance: Counter = field(default_factory=Counter)     # stat changes after it (analytics' rule)
    balance_cards: int = 0

    def changed(self) -> bool:
        return bool(self.corrected or self.neutral or self.added or self.folded or self.earlier or self.later
                    or self.tags_moved or self.archived or self.retagged or self.authors_set)

    def line(self) -> str:
        def names(items):
            return ", ".join(sorted(set(items)))
        made = self.made
        fates = Counter(f for _, f in self.archived)
        cur_e = sum(1 for _, c in self.earlier if c)
        parts = [f"{made.get('readings', 0):,} PSD versions of {made.get('cards', 0)} cards read "
                 f"({made.get('template', 0)} by the old template's positions, {made.get('unread', 0)} unread)",
                 f"{len(self.corrected)} version{'' if len(self.corrected) == 1 else 's'} corrected"
                 + (f" ({names(self.corrected)})" if self.corrected else "")]
        if self.neutral:
            parts.append(f"{len(self.neutral)} unmatched aligned to a neighbour ({names(self.neutral)})")
        parts.append(f"{len(self.added)} stat version{'' if len(self.added) == 1 else 's'} added, {len(self.folded)} "
                     f"folded" + (f" ({names(self.folded)})" if self.folded else "")
                     + f", {len(self.earlier)} start{'' if len(self.earlier) == 1 else 's'} moved earlier "
                     f"({cur_e} current card{'' if cur_e == 1 else 's'})"
                     + (f", {len(self.later)} later" if self.later else ""))
        b = self.balance
        parts.append(f"stat changes {b['buff'] + b['nerf'] + b['shift']} ({b['buff']} buffs, {b['nerf']} nerfs, "
                     f"{b['shift']} moved) and {b['uncounted']} uncounted, on {self.balance_cards} cards")
        parts.append(f"{len(self.archived)} card{'' if len(self.archived) == 1 else 's'} archived ("
                     + ", ".join(f"{fates[k]} {k}" for k in ("renamed", "retired", "deleted") if fates[k]) + ")"
                     + (f", {len(self.retagged)} retagged ({names(n for n, _ in self.retagged)})" if self.retagged else "")
                     + f"; authors on {self.authors}")
        return "Backfill: " + "; ".join(parts) + "."


def _line(card) -> Optional[list]:
    s = getattr(card, "stats", None)
    return [s.hp, s.defense, s.attack, s.speed] if s is not None else None


def _edited(card) -> bool:
    return (getattr(card, "version", None) or {}).get("cause") in ("edit", "authors")


def _info(card) -> dict:
    """card's version record (a legacy one when it has none, as carry_to makes)."""
    if getattr(card, "version", None) is None:
        card.version = meta(None, card.timestamp, "legacy", legacy=not getattr(card, "blob_sha", None))
    return card.version


def _set_line(card, seg: dict) -> None:
    """A version's stats corrected to its segment's (stats and creatures only); what it had kept in backfill_was."""
    was = {"stats": _line(card), "creatures": [[c.hp, c.defense, c.attack, c.speed] for c in card.creatures or []]}
    problems = [p for p in getattr(card, "problems", None) or [] if str(p).startswith("OUT OF RANGE: stats")]
    if problems:
        was["problems"] = problems
        card.problems = [p for p in card.problems if p not in problems]
    card.stats = CardStats(*seg["line"])
    card.creatures = [CardStats(*c) for c in seg.get("creatures") or []] or None
    info = _info(card)
    info["backfill_was"], info[MARK] = was, FILE_ID


def _seg_at(segs: list, at: float) -> int:
    k = 0
    for j, seg in enumerate(segs):
        if seg["from"] <= at:
            k = j
    return k


def match(card, readings: list, segs: list, paths: set, index: GitIndex) -> Optional[int]:
    """A stored version's segment: by its blob (a blob met twice: the meeting at or before its start), else by the
    commit at its path with its time (or at the path an exact rename brought it from), else by its start when its path
    is the lineage's; None: unmatched."""
    when = max(start(card)[0] or card.timestamp, card.timestamp)
    blob = getattr(card, "blob_sha", None)
    if not blob:
        hit = index.blob_at(card.path, card.timestamp) or index.blob_at(index.origin(card.path), card.timestamp)
        blob = hit[1] if hit else None
    found = [i for i, r in enumerate(readings) if blob and r["blob"] == blob]
    if found:
        i = max((i for i in found if readings[i]["at"] <= when), default=found[0])
        return _seg_at(segs, readings[i]["at"])
    if card.path in paths:
        return _seg_at(segs, start(card)[0] or card.timestamp)
    return None


def _copy(card, seg: dict, at: float, commit: Optional[str]) -> CardInfo:
    out = deepcopy(card)
    out.archived = None
    vars(out).pop("created", None)
    out.stats = CardStats(*seg["line"])
    out.creatures = [CardStats(*c) for c in seg.get("creatures") or []] or None
    out.blob_sha, out.path, out.timestamp = seg["blob"], seg["path"], at
    out.version = {**meta(None, at, "psd"), "from_commit": commit, MARK: FILE_ID}
    return out


def _move(card, at: float, commit: Optional[str], report: Merged, name: str, current: bool) -> None:
    info = _info(card)
    (report.earlier if at < (info.get("from") if info.get("from") is not None else card.timestamp)
     else report.later).append((name, current))
    info["from"], info["from_commit"], info[MARK] = at, commit, FILE_ID


def merge_card(name: str, entry: dict, stats: dict, old_stats: dict, index: GitIndex, report: Merged) -> None:
    """
    One name's stat history brought to its segments: matched versions corrected in place (an edit's or the current
    card's never), unmatched ones aligned to a neighbour, a change inside a version's span split into a copy, a change
    before the first version reaching back to the name's first PSD commit; then collapse. Never the current card's data.
    """
    segs, readings = entry["segments"], entry["readings"]
    holder = stats.get(name)
    if not segs or (holder is not None and getattr(holder, "duplicate_of", None)):
        return
    stored = list(old_stats.get(name) or [])
    chain = [*stored, holder] if holder is not None else stored
    if not chain:
        return
    paths = set(entry["paths"])
    own = {}
    for v in chain:
        k = match(v, readings, segs, paths, index)
        if k is None:
            report.unmatched.append((name, start(v)[0]))
        else:
            own[id(v)] = k
    for v in chain:                                               # corrections in place
        if id(v) not in own or _edited(v):
            continue
        seg = segs[own[id(v)]]
        if _line(v) == seg["line"]:
            continue
        if v is holder:
            report.current.append((name, _line(v), seg["line"]))
            continue
        _set_line(v, seg)
        report.corrected.append(name)
    for i, v in enumerate(chain):                                 # unmatched: never a change of their own
        if id(v) in own or v is holder or _edited(v):
            continue
        before = chain[i - 1] if i else None
        after = chain[i + 1] if i + 1 < len(chain) else None
        if any(n is not None and _line(n) == _line(v) for n in (before, after)):
            continue
        near = after if after is not None else before
        if near is not None and _line(near):
            _set_line(v, {"line": _line(near), "creatures": [[c.hp, c.defense, c.attack, c.speed]
                                                               for c in near.creatures or []]})
            report.neutral.append(name)
    copies = []
    for i, v in enumerate(chain):                                 # splits
        if id(v) not in own or _edited(v):
            continue
        k = own[id(v)]
        a = start(v)[0] if start(v)[0] is not None else v.timestamp
        nxt = chain[i + 1] if i + 1 < len(chain) else None
        b = (start(nxt)[0] or nxt.timestamp) if nxt is not None else \
            ((getattr(v, "archived", None) or {}).get("at") or FAR)
        if i == 0 and any(seg["from"] < a for seg in segs[1:]):
            a = segs[0]["from"]                                   # a change before the first version
        mine, earlier = None, False
        for j, seg in enumerate(segs):
            end = segs[j + 1]["from"] if j + 1 < len(segs) else FAR
            if not (seg["from"] < b and end > a):
                continue
            at = max(a, seg["from"])
            commit = seg["commit"] if at == seg["from"] else (getattr(v, "version", None) or {}).get("from_commit")
            if j == k:
                mine = (at, commit) if earlier else None      # moves only when an earlier line takes the time before
                continue
            earlier = earlier or j < k
            if j > k and v is holder:
                report.current.append((name, _line(v), seg["line"]))
                continue
            if any(start(w)[0] == at for w in chain if w is not v) or any(start(c)[0] == at for c in copies):
                continue
            copy = _copy(v, seg, at, commit)
            if j > k and getattr(v, "archived", None):            # the PSD changed after the record's reading
                copy.archived, v.archived = v.archived, None
                report.tags_moved.append(name)
            copies.append(copy)
            report.added.append((name, at, seg["line"]))
        if mine is not None and mine[0] != (start(v)[0] if start(v)[0] is not None else v.timestamp):
            _move(v, mine[0], mine[1], report, name, v is holder)
    hist = sorted([v for v in chain if v is not holder] + copies,
                  key=lambda v: (bool(getattr(v, "archived", None)), start(v)[0] or v.timestamp))
    kept, dropped = collapse(hist, holder)
    report.folded += [name] * len(dropped)
    for x, y in zip(kept, kept[1:] + ([holder] if holder is not None else [None])):
        if MARK not in (getattr(x, "version", None) or {}) and MARK not in (getattr(y, "version", None) or {}):
            continue
        if y is not None:
            close(x, index, start(y)[0], index.by_sha.get(start(y)[1] or ""))
        elif getattr(x, "archived", None):
            close(x, index, x.archived.get("at"), index.by_sha.get(x.archived.get("commit") or ""))
    if kept:
        old_stats[name] = kept
    elif name in old_stats:
        del old_stats[name]


def _author(name: str, tag: dict, data: dict, stats: dict, last) -> Optional[str]:
    """An archived record's author: the card it became (its credit in the database), its own line or first author
    (the file), else the record's own."""
    target = tag.get("renamed_to")
    if target and target in stats and getattr(stats[target], "author", None):
        return stats[target].author
    found = data["archive"]["authors"].get(name) or data["archive"]["authors"].get(
        next((n for n in data["archive"]["authors"] if key(n) == key(name)), ""))
    return (found[0] if found and found[0] else None) or getattr(last, "author", None)


def merge(stats: dict, old_stats: dict, data: dict, index: GitIndex) -> Merged:
    """
    The backfill file into the database, in place: the archive records git has and the database lacks, the fate of a
    record tagged deleted that git calls a rename or a retirement, every archived record's author; then each name's
    stat history (merge_card). Idempotent: a second call changes nothing.
    """
    report = Merged(made=data.get("made") or {})
    archived = {key(n) for n, vs in old_stats.items() if n not in stats and vs and getattr(vs[len(vs) - 1], "archived", None)}
    for name, entry in sorted(data["archive"]["cards"].items()):
        tag_of = old_stats.get(name) or []
        tag = getattr(tag_of[len(tag_of) - 1], "archived", None) if tag_of and name not in stats else None
        if tag is not None or key(name) in archived:
            if tag is not None and not tag.get("renamed_to") and not tag.get("moved_to") and entry["fate"] != "deleted":
                rec = (entry.get("record") or {}).get("archived") or {}
                field_ = "renamed_to" if entry["fate"] == "renamed" else "moved_to"
                if rec.get(field_):
                    tag[field_], tag[MARK] = rec[field_], FILE_ID
                    report.retagged.append((name, rec[field_]))
            continue
        if name in stats or not entry.get("record") or any(key(n) == key(name) for n in stats):
            continue
        card = from_record(entry["record"])
        old_stats.setdefault(name, []).append(card)
        report.archived.append((name, entry["fate"]))
    for name, entry in sorted(data["cards"].items()):
        if name in stats or name in old_stats:
            merge_card(name, entry, stats, old_stats, index, report)
    for name, versions in old_stats.items():
        last = versions[len(versions) - 1] if versions else None
        tag = getattr(last, "archived", None) if last is not None and name not in stats else None
        if not tag:
            continue
        who = _author(name, tag, data, stats, last)
        if who and tag.get("author") != who:
            tag["author"] = who
            report.authors_set.append(name)
        if tag.get("author"):
            report.authors += 1
    commits_of(stats, old_stats, index)                          # a record made here: its start's commit, as the cleanup would
    report.balance, report.balance_cards = balance(stats, old_stats)
    return report


def balance(stats: dict, old_stats: dict) -> tuple:
    """Stat changes as the website's analytics counts them: consecutive versions (a start read twice folds), both
    lines full and on the scale: buff, nerf or shift; else uncounted. (Counter, cards with a counted change)."""
    def when(v):
        info = getattr(v, "version", None) or {}
        return info["from"] if isinstance(info.get("from"), (int, float)) else v.timestamp

    def full(v):
        line = _line(v)
        return tuple(line) if line and all(isinstance(x, int) and x >= 0 for x in line) else None
    counts, cards = Counter(), set()
    for name in set(stats) | set(old_stats):
        current = stats.get(name)
        if current is not None and getattr(current, "duplicate_of", None):
            continue
        chain = sorted((v for v in old_stats.get(name) or [] if isinstance(v.timestamp, (int, float))), key=when)
        chain += [current] if current is not None else []
        chain = [v for v, w in zip(chain, chain[1:]) if when(v) != when(w)] + chain[len(chain) - 1:]
        for a, b in zip(chain, chain[1:]):
            la, lb = full(a), full(b)
            if la == lb or not (la or lb):
                continue
            if la and lb and sum(la) <= OFF_SCALE and sum(lb) <= OFF_SCALE:
                counts["buff" if sum(lb) > sum(la) else "nerf" if sum(lb) < sum(la) else "shift"] += 1
                cards.add(name)
            else:
                counts["uncounted"] += 1
    return (counts, len(cards))


def merged_before(stats: dict, old_stats: dict, file_id: str) -> bool:
    every = [v for versions in old_stats.values() for v in versions] + list(stats.values())
    return any((getattr(v, "version", None) or {}).get(MARK) == file_id
               or (getattr(v, "archived", None) or {}).get(MARK) == file_id for v in every)


def load_file(path: str = "") -> Optional[dict]:
    path = path or FILE
    if not isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return loads(fh.read())


def merged(db, index, path: str = "") -> tuple:
    """
    /update_stats' once-merge, on a copy of db's cards and history that goes into db only when it ran through: nothing
    without the file or the history index; "waits" until recover has run; nothing once a version or tag carries the
    file's id (merged_before).

    Returns:
        (the report line or None, whether db changed)
    """
    from commands.card_models import StatsDatabase
    if index is None:
        return (None, False)
    data = load_file(path)
    if data is None:
        return (None, False)
    if not recovered_before(db.stats, db.old_stats):
        return ("Backfill waits for the history recovery.", False)
    if merged_before(db.stats, db.old_stats, data["id"]):
        return (None, False)
    trial = StatsDatabase()
    trial.stats, trial.old_stats = deepcopy(db.stats), deepcopy(db.old_stats)
    report = merge(trial.stats, trial.old_stats, data, index)
    if not report.changed():
        return (None, False)
    db.stats.clear()
    db.stats.update(trial.stats)
    db.old_stats.clear()
    db.old_stats.update(trial.old_stats)
    return (report.line(), True)


# ========================
# Rehearse
# ========================
def rehearse(path: str, repo: str, head: str, pickles: str, out: str) -> int:
    """The bot's merge on copies of the pickles: its line, what changed per card, the second cleanup, the copies."""
    from commands.card_data import cleaned
    from commands.card_db import dump_pickle, read_pickles
    from commands.card_models import StatsDatabase
    makedirs(out, exist_ok=True)
    if samefile(abspath(out), abspath(pickles)) or samefile(abspath(out), BOT_CHECKOUT):
        print("--out must be neither the --pickles folder nor the bot's checkout.")
        return 1
    db = StatsDatabase()
    db.stats, db.old_stats = read_pickles(pickles)
    index = GitIndex.from_git(repo, head)
    before = (len(db.old_stats), sum(len(v) for v in db.old_stats.values()), len(db.archived()))
    data = load_file(path)
    report = Merged()
    if data is not None and recovered_before(db.stats, db.old_stats) and not merged_before(db.stats, db.old_stats, data["id"]):
        trial = StatsDatabase()
        trial.stats, trial.old_stats = deepcopy(db.stats), deepcopy(db.old_stats)
        report = merge(trial.stats, trial.old_stats, data, index)
    line, changed = merged(db, index, path)
    print(line or "Backfill: nothing to do.")
    for title, items in (("corrected", report.corrected), ("aligned", report.neutral), ("folded", report.folded),
                         ("tags moved to a newer line", report.tags_moved)):
        print(f"{title}: {', '.join(f'{n} {c}' if c > 1 else n for n, c in sorted(Counter(items).items()))}")
    print("added:", ", ".join(f"{n} {strftime('%Y-%m-%d', localtime(at))} {line_}" for n, at, line_ in report.added))
    print("starts earlier (current cards):", ", ".join(sorted(n for n, c in report.earlier if c)))
    print("starts later:", ", ".join(sorted(f"{n}{' (current)' if c else ''}" for n, c in report.later)))
    print("unmatched:", ", ".join(f"{n} {strftime('%Y-%m-%d', localtime(at or 0))}" for n, at in report.unmatched))
    print("current cards that differ (not written):", "; ".join(f"{n} {a} -> {b}" for n, a, b in report.current))
    print("archived:", ", ".join(f"{n} ({f})" for n, f in report.archived))
    print("retagged:", ", ".join(f"{n} -> {t}" for n, t in report.retagged))
    print("authors set:", len(report.authors_set))
    again = cleaned(db, index)
    print(f"Second cleanup: {again or 'nothing to change'}")
    again2, _ = merged(db, index, path)
    print(f"Second merge: {again2 or 'nothing'}")
    after = (len(db.old_stats), sum(len(v) for v in db.old_stats.values()), len(db.archived()))
    print(f"old_stats names {before[0]} -> {after[0]}, versions {before[1]} -> {after[1]}, archived {before[2]} -> {after[2]}")
    dump_pickle(db.old_stats, join(out, "old_stats.pkl"))
    dump_pickle(db.stats, join(out, "stats.pkl"))
    print(f"Copies written to {out}.")
    return 0


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.history_backfill", description="The one-time stat history backfill.")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("build", help="read every stat-card PSD version in git; write the backfill file")
    cmd.add_argument("--git", required=True, help="a full clone of the card repo holding every blob it reads")
    cmd.add_argument("--head", default="origin/main", help="the card repo commit (default origin/main)")
    cmd.add_argument("--cache", required=True, help="a JSON-lines file of readings (kept between runs)")
    cmd.add_argument("--out", default=FILE_NAME, help=f"the file to write (default {FILE_NAME})")
    cmd = sub.add_parser("rehearse", help="the bot's merge on copies of the pickles")
    cmd.add_argument("--file", default=FILE, help="the backfill file")
    cmd.add_argument("--git", required=True, help="a full clone of the card repo (blobless is enough)")
    cmd.add_argument("--head", default="origin/main", help="the card repo commit (default origin/main)")
    cmd.add_argument("--pickles", required=True, help="a folder holding stats.pkl and old_stats.pkl (read only)")
    cmd.add_argument("--out", required=True, help="a folder for the merged copies (never --pickles)")
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.git, args.head, args.cache, args.out)
    return rehearse(args.file, args.git, args.head, args.pickles, args.out)


# ========================
# Module Exports
# ========================
__all__ = [
    'FILE',
    'FILE_ID',
    'MARK',
    'Fate',
    'History',
    'Lineage',
    'Merged',
    'archive_candidates',
    'balance',
    'build',
    'fate',
    'lineages',
    'main',
    'match',
    'merge',
    'merge_card',
    'merged',
    'merged_before',
    'read_psd',
    'segments',
    'status',
]


if __name__ == "__main__":
    raise SystemExit(main())
