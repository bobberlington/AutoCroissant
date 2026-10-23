"""
Card history from git, without Discord, network or psd-tools: one `git log --raw` of the card repo gives the newest
commit, first author, deletions and PNG commits per path, each blob's intro (the oldest commit that added it at any
path) and exact renames. On that index: the version rules (a version only when a card's data changes; from/to dates
and a cause from git; versions without a blob marked legacy), the seed pass over old_stats, and the archive tags'
dates. Design: docs/Campaigns/DATA_BUILD_ACTION.md section 7.4.

CLI (copies only): python -m commands.card_history refine --git <full clone> --pickles <dir> --out <dir> [--bot-git <dir>]
"""
from argparse import ArgumentParser
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from os import environ, makedirs
from os.path import abspath, dirname, isdir, join, samefile, splitext
from re import findall, sub
from subprocess import PIPE, Popen, run
from typing import Optional

from commands.card_db import SeedUnpickler, card, dump_pickle, read_pickles, record
from commands.card_models import CardInfo, CardStats, CardType, card_name

# ========================
# Configuration
# ========================
# the fields a version is about (the path, blob, timestamp, parse and edit bookkeeping are not data)
DATA_FIELDS = ("card_type", "display_name", "ability", "stars", "subtype", "series", "subseries", "types", "stats",
               "creatures", "author", "note", "hidden", "duplicate_of")
CAUSES = ("psd", "reparse", "moved", "edit", "authors", "legacy")   # moved: the same blob in another folder (its data)
LOG_FORMAT = "C%x09%H%x09%ct%x09%an"                  # one header line per commit: sha, committer time, author
NO_BLOB = "0" * 40
TYPE_RENAMES = {"offense": "negation"}                # retired Types/ icon -> its name now (same bytes, 9649b4d)
# a parse problem -> the field that reading could not know (compared as unknown)
UNKNOWN_IF = (("NO STARS FOUND", "stars"), ("MISSPELT TYPE", "types"), ("NO ABILITY LAYER", "ability"))
SERIES = ("series", "subseries")
RETIRED = ("MDW", "Markers")                          # no card folders (card_data.EXCLUDE_FOLDERS): a PSD there retired
STAT_MAX, STAT_CAP, STARS_MAX = 10, 99, 7             # a stat above 10 needs an allowance (allows_high); 3 digits: none
# CardValidator.EXCESSIVE_STAT_EXCLUSIONS until parser 2 (5835139, 2026-10-21): their history's high stats were allowed
HIGH_STATS_LISTED = frozenset({"Royal Eradicator Main Cannon", "Pix", "Tainted Lazarus", "Sonic", "Twin Emperors"})
RANGE_PARSER = 2                                      # parser 2 (5835139): the validator's range check, allow_high_stats
PNG_SLACK = 6 * 3600.0                                # a push's PNG export can follow its PSD by hours
PICKLE_FILES = ("stats.pkl", "old_stats.pkl")         # the bot's committed database (PICKLE commits)
RESTORED_MARK = b"restored_from"                      # in a PICKLE's bytes: it held a restored version (recover ran)
PRINTED = ("card_type", "ability", "types", "stars", "stats")    # what a reading of the card prints
STAT_KEYS = ("hp", "defense", "attack", "speed")      # _printed's stats, each known or not on its own
PRINTS_NONE = ("",)                                   # _printed's ability when parser 2 found none printed
BOT_CHECKOUT = abspath(join(dirname(abspath(__file__)), ".."))   # this AutoCroissant checkout (its PICKLE commits)


@dataclass
class Commit:
    """One commit of the index: its place in the log (0 = newest), sha, committer time and author name."""
    order: int
    sha: str
    at: float
    author: str


@dataclass
class GitIndex:
    """The card repo's PSD and PNG history from one `git log --raw --no-renames` (newest first)."""
    commits: list = field(default_factory=list)       # Commit, newest first
    newest: dict = field(default_factory=dict)        # path -> newest Commit that touched it
    first: dict = field(default_factory=dict)         # path -> oldest Commit that touched it (its first author)
    deleted: dict = field(default_factory=dict)       # path -> newest Commit that deleted it
    intro: dict = field(default_factory=dict)         # blob -> (oldest Commit that added it at any path, that path)
    intro_at: dict = field(default_factory=dict)      # (blob, path) -> oldest Commit that added it there
    paths_of: dict = field(default_factory=dict)      # blob -> {path: oldest Commit that added it there}
    pngs: dict = field(default_factory=dict)          # PNG path -> Commits that changed it, newest first
    pngs_lower: dict = field(default_factory=dict)    # the same by lower-cased path (a PNG named in another case)
    renames: dict = field(default_factory=dict)       # old path -> (new path, Commit): the newest exact rename
    renamed_from: dict = field(default_factory=dict)  # new path -> old path of its newest exact rename into it
    by_sha: dict = field(default_factory=dict)        # commit sha -> Commit
    at_path: dict = field(default_factory=dict)       # path -> [(Commit, blob)] that put a blob there, newest first
    gone_at: dict = field(default_factory=dict)       # path -> Commits that deleted it, newest first
    lower: dict = field(default_factory=dict)         # lower-cased path -> path (at_path or gone_at), filled on first use

    @classmethod
    def parse(cls, text: str) -> "GitIndex":
        """The index of `git log <head> --format=LOG_FORMAT --raw --no-abbrev --no-renames -z -- '*.psd' '*.png'`."""
        index = cls()
        commit, meta, gone, added = None, None, {}, {}

        def close():
            for blob, old in gone.items():
                if blob in added and old not in index.renames:
                    index.renames[old] = (added[blob], commit)
                    index.renamed_from.setdefault(added[blob], old)

        for token in text.split("\0"):
            token = token.lstrip("\n")
            if meta is not None:                          # the path of the raw line before
                index._entry(commit, meta, token, gone, added)
                meta = None
            elif token.startswith(":"):
                meta = token
            elif token.startswith("C\t"):
                if commit is not None:
                    close()
                _, sha, at, author = (token.split("\t", 3) + [""])[:4]
                commit, gone, added = Commit(len(index.commits), sha, float(at), author), {}, {}
                index.commits.append(commit)
                index.by_sha[sha] = commit
        if commit is not None:
            close()
        return index

    def _entry(self, commit: Commit, meta: str, path: str, gone: dict, added: dict) -> None:
        """One raw line ":<mode> <mode> <src> <dst> <status>" at path."""
        parts = meta[1:].split()
        src, dst, status = parts[2], parts[3], parts[4][:1]
        self.newest.setdefault(path, commit)
        self.first[path] = commit                         # newest first: the last one seen is the oldest
        if status == "D":
            self.deleted.setdefault(path, commit)
            self.gone_at.setdefault(path, []).append(commit)
            gone[src] = path
        elif dst != NO_BLOB:
            self.intro[dst] = (commit, path)
            self.intro_at[(dst, path)] = commit
            self.paths_of.setdefault(dst, {})[path] = commit
            self.at_path.setdefault(path, []).append((commit, dst))
            if status == "A":
                added[dst] = path
        if path.lower().endswith(".png"):
            self.pngs.setdefault(path, []).append(commit)
            self.pngs_lower.setdefault(path.lower(), []).append(commit)

    @classmethod
    def from_git(cls, repo: str, head: str = "HEAD") -> "GitIndex":
        """
        The index of a clone (a blobless one does: no blob is read; GIT_NO_LAZY_FETCH makes a fetch fail loudly).

        Raises:
            ValueError: a shallow clone (its history is cut, so every date would be wrong)
        """
        env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
        shallow = run(["git", "-C", repo, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, env=env)
        if shallow.returncode != 0 or shallow.stdout.strip() != "false":
            raise ValueError(f"{repo}: a full (non-shallow) clone of the card repo is needed for its history")
        done = run(["git", "-C", repo, "log", head, f"--format={LOG_FORMAT}", "--raw", "--no-abbrev", "--no-renames", "-z",
                    "--", "*.psd", "*.png"], capture_output=True, env=env, check=True)
        return cls.parse(done.stdout.decode("utf-8", "replace"))

    def blob_intro(self, blob: Optional[str]) -> Optional[Commit]:
        """The oldest commit that added this blob at any path (a move or a re-commit of the same bytes is no change)."""
        found = self.intro.get(blob) if blob else None
        return found[0] if found else None

    def blob_since(self, blob: Optional[str], path: str, at: Optional[float] = None) -> Optional[Commit]:
        """When a version's bytes became its card's: the blob's intro, unless path (or the path an exact rename brought
        them from) held other bytes after it: then the newest commit that put these back, no later than at (the
        version's reading; None: the newest). A revert dates from the revert (2025-04-04), an exact rename keeps the
        date (even onto a path that held others); idempotent on its result."""
        first = self.blob_intro(blob)
        starts, seen, after = [], {path}, None        # starts newest first; after: the rename that took them from path
        while first is not None:
            puts = [(c, b) for c, b in self.at_path.get(path, []) if after is None or c.order > after.order]
            mine = [c for c, b in puts if b == blob]
            old, came = self.renamed_from.get(path), mine[len(mine) - 1] if mine else None
            renamed = came is not None and old is not None and old not in seen and \
                (self.renames.get(old) or (None, None))[1] is came
            starts += [c for i, (c, b) in enumerate(puts) if b == blob and i + 1 < len(puts) and not (renamed and c is came)
                       and puts[i + 1][1] != blob and puts[i + 1][0].order < first.order]
            if not renamed:
                break
            seen.add(old)
            path, after = old, came
        return next((c for c in starts if at is None or c.at <= at), first)

    def intro_of(self, blob: Optional[str], name: str, others=(), path: Optional[str] = None,
                 at: Optional[float] = None) -> Optional[Commit]:
        """blob_since at path (no path: blob_intro) for the card `name`, never before the oldest commit that put the
        bytes at a path of this name when they were also at a path of another name in `others` (the database's names:
        a renamed card's archive holds that time, a copy is another card)."""
        since = self.blob_since(blob, path, at) if path else self.blob_intro(blob)
        if since is None or not others:
            return since
        paths = self.paths_of.get(blob) or {}
        if not any(card_name(p) != name and card_name(p) in others for p in paths):
            return since
        mine = [c for p, c in paths.items() if card_name(p) == name]
        own = max(mine, key=lambda c: c.order) if mine else None
        return own if own is not None and own.order < since.order else since

    def arrival(self, blob: Optional[str], path: str, at: Optional[float] = None) -> Optional[Commit]:
        """The commit that put blob at path (a path in another letter case counts), else None: the oldest; with `at`,
        the one that began the stretch holding them then, since the path last held other bytes (a revert dates from
        the revert, as blob_since), else the oldest."""
        puts = self.at_path.get(self._path(path), []) if blob else []
        mine = [i for i, (_, b) in enumerate(puts) if b == blob]
        if not mine:
            return None
        i = next((i for i in mine if at is not None and puts[i][0].at <= at), mine[len(mine) - 1])
        while i + 1 < len(puts) and puts[i + 1][1] == blob:
            i += 1
        return puts[i][0]

    def _path(self, path: str) -> str:
        """path as the index has it: itself, else the one in another letter case (a PNG named so)."""
        if path in self.at_path or path in self.gone_at:
            return path
        if not self.lower:
            self.lower = {p.lower(): p for p in (*self.gone_at, *self.at_path)}
        return self.lower.get(path.lower(), path)

    def origin(self, path: str, once: bool = False) -> str:
        """The path an exact rename (or a chain of them; once: the newest only) brought path from, else path."""
        seen = {path}
        while path in self.renamed_from and self.renamed_from[path] not in seen:
            path = self.renamed_from[path]
            seen.add(path)
            if once:
                break
        return path

    def blob_at(self, path: str, at: Optional[float]) -> Optional[tuple]:
        """(Commit, blob) of the one commit at `at` that put a blob at path (a legacy version's stored timestamp is its
        commit's time), else None (none, or several at that second)."""
        found = [(commit, blob) for commit, blob in self.at_path.get(path, []) if commit.at == at]
        return found[0] if len(found) == 1 else None

    def held(self, path: str, blob: Optional[str], since: float, until: float) -> bool:
        """Whether no commit put other bytes than blob at path after `since`, up to `until` (a start carried across
        them would cross a revert's interlude)."""
        return all(b == blob for c, b in self.at_path.get(self._path(path), []) if since < c.at <= until)

    def blob_before(self, path: str, at: float) -> Optional[str]:
        """The blob at path at time `at`: the newest commit (log order) at or before it that put one there or deleted
        it (None then, or with none); a path in another letter case counts (a PNG named so)."""
        path = self._path(path)
        put = next(((c, b) for c, b in self.at_path.get(path, []) if c.at <= at), None)
        cut = next((c for c in self.gone_at.get(path, []) if c.at <= at), None)
        if put is None or (cut is not None and cut.order < put[0].order):
            return None
        return put[1]

    def png_blob(self, psd_path: str, at: float) -> Optional[str]:
        """The card's PNG blob `at` that time plus PNG_SLACK (an export pushed after its PSD counts)."""
        return self.blob_before(splitext(psd_path)[0] + ".png", at + PNG_SLACK)

    def renamed(self, path: str, stats, since: float = 0.0) -> tuple:
        """
        Where an exact rename chain from path leads (the PSD's, else its PNG's: a rename that edited the PSD), its first
        rename no older than `since` (a record's own reading): (the name of a card in stats (any container of names),
        None, the rename that reached it), or (None, the path in a folder of no cards: retired, the move into it), or
        (None, None, None). That step is the name's last day: Azamoth the Possessed King moved within its name first.
        """
        for old in (path, splitext(path)[0] + ".png"):
            if old in self.renames and self.renames[old][1].at < since:
                continue
            seen = set()
            while old in self.renames and old not in seen:
                seen.add(old)
                old, step = self.renames[old]
                if card_name(old) in stats and not any(folder in RETIRED for folder in old.split("/")):
                    return (card_name(old), None, step)
                if any(folder in RETIRED for folder in old.split("/")):
                    return (None, old, step) if old.lower().endswith(".psd") else (None, None, None)
        return (None, None, None)

    def png_commit(self, psd_path: str, before: Optional[Commit] = None, before_at: Optional[float] = None) -> Optional[str]:
        """The newest commit of the card's PNG older than `before` (else older than before_at; else the newest); a PNG
        named in another letter case counts (build_site pairs them so)."""
        png = splitext(psd_path)[0] + ".png"
        for commit in self.pngs.get(png) or self.pngs_lower.get(png.lower(), []):
            if before is not None:
                if commit.order > before.order:
                    return commit.sha
            elif before_at is None or commit.at < before_at:
                return commit.sha
        return None


# ========================
# Versions
# ========================
def data_of(card: CardInfo) -> dict:
    """A card's data fields as plain values (card_db's record), the ones a version is about; one form each for the
    same data: a title equal to the name is none, an empty ability none, a retired type icon its name now."""
    rec = record(card)
    data = {key: rec[key] for key in DATA_FIELDS}
    if data["display_name"] == rec["name"]:
        data["display_name"] = None
    data["ability"] = data["ability"] or None
    data["types"] = [TYPE_RENAMES.get(t, t) for t in data["types"] or []]
    return data


def unknown(card: CardInfo) -> set:
    """Fields a reading could not know: the title before parser 2 (it read none), a field its parse failed on."""
    found = set()
    if getattr(card, "parser_version", None) is None and not getattr(card, "display_name", None):
        found.add("display_name")
    for problem in getattr(card, "problems", None) or []:
        found.update(key for prefix, key in UNKNOWN_IF if str(problem).startswith(prefix))
    return found


def leaf(data: dict) -> Optional[str]:
    """The most specific series: the subseries, else the series (the 2026-10-20 split made a series a subseries)."""
    return data["subseries"] or data["series"]


def knows(a: CardInfo, b: CardInfo) -> bool:
    """
    Whether reading a knew all that b did (data_of): every field b could know (unknown), a knew too, with b's value; a
    series moved under a parent (the same leaf) is the same. So a reading that knew less (a parse that failed on a
    field) never stands for one that knew more. A value record() can't read counts as not known (a version is kept, as
    before 2026-10-22), so the cleanup at each /update_stats never fails on it.
    """
    try:
        x, y = data_of(a), data_of(b)
    except (TypeError, AttributeError) as error:
        print(f"History: {getattr(a, 'name', '?')} compared as changed ({error})")
        return False
    skip = unknown(b)
    if not unknown(a) <= skip:
        return False
    if leaf(x) == leaf(y):
        skip = skip | set(SERIES)
    return all(x[key] == y[key] for key in DATA_FIELDS if key not in skip)


def same_data(a: CardInfo, b: CardInfo) -> bool:
    """Whether one of two readings stands for the other (knows): the same data, a field the other could not know
    aside."""
    return knows(a, b) or knows(b, a)


def _exact(a: CardInfo, b: CardInfo) -> bool:
    """Whether two versions' data_of is equal, every field known."""
    try:
        return data_of(a) == data_of(b)
    except (TypeError, AttributeError):
        return False


def start(card: CardInfo) -> tuple:
    """(when, commit) a version starts: its `version` record's from, else its timestamp (/update_stats keeps none)."""
    info = getattr(card, "version", None)
    if info and info.get("from") is not None:
        return (info["from"], info.get("from_commit"))
    return (card.timestamp, None)


def carry(dropped: CardInfo, holder: CardInfo) -> None:
    """
    A dropped version's earlier start goes to the version that holds its data now (as_of keeps the first date); the
    cause and legacy flag stay the holder's own (a current card with a blob is no legacy version). A holder without a
    record (/update_stats) gets one, cause legacy: the seed pass dates it from git and keeps the earlier start. The
    website dates a version by this start too (TTSCardMaker site/build/analytics.py when).
    """
    at, commit = start(dropped)
    if at is not None:
        carry_to(holder, at, commit)


def carry_to(holder: CardInfo, at: float, commit: Optional[str]) -> None:
    """carry() from a start: holder's start moves to `at` (its commit `commit`) when that is earlier."""
    now = start(holder)[0]
    if now is not None and at >= now:
        return
    info = getattr(holder, "version", None)
    if info is None:
        holder.version = info = meta(None, at, "legacy", legacy=not getattr(holder, "blob_sha", None))
    info["from"], info["from_commit"] = at, commit


def _reread(older: CardInfo, newer: CardInfo) -> bool:
    """Whether older is an earlier reading of newer's PSD: the same start (one commit read twice, by an older parser
    or before an edit) and no other blob, newer read with a blob or older without one, by no older parser, knowing
    every field older knew (unknown). A version an edit or authors change made (cause edit, authors) is no re-read:
    it starts no earlier than the version it ends, so at its start when the edits commit is older."""
    if getattr(older, "archived", None) or start(older)[0] != start(newer)[0]:
        return False
    if (getattr(newer, "version", None) or {}).get("cause") in ("edit", "authors"):
        return False
    a, b = getattr(older, "blob_sha", None), getattr(newer, "blob_sha", None)
    if (a and b and a != b) or (a and not b) or not unknown(newer) <= unknown(older):
        return False
    return (getattr(newer, "parser_version", None) or 0) >= (getattr(older, "parser_version", None) or 0)


def _rereads(versions: list, holder: Optional[CardInfo]) -> tuple:
    """versions without the earlier readings (_reread) of the version after them (or the holder): (kept, dropped).
    A dropped reading's start is its successor's, so nothing is carried."""
    kept, dropped = [], []
    for version in versions:
        if kept and _reread(kept[len(kept) - 1], version):
            dropped.append(kept.pop())
        kept.append(version)
    if kept and holder is not None and _reread(kept[len(kept) - 1], holder):
        dropped.append(kept.pop())
    return (kept, dropped)


def _runs(versions: list) -> list:
    """Consecutive versions one of which knew all the others did (_knower: a field one of them could not know never
    joins two known values); an archived version joins only a run it knew all of (it holds the run) and ends it."""
    runs: list = []
    for version in versions:
        tail = runs[len(runs) - 1] if runs else None
        if tail and not getattr(tail[len(tail) - 1], "archived", None) and (
                all(knows(version, member) for member in tail) if getattr(version, "archived", None)
                else _knower([*tail, version]) is not None):
            tail.append(version)
        else:
            runs.append([version])
    return runs


def _knower(run: list) -> Optional[CardInfo]:
    """The version a run keeps when it does not go into the card after it: one that knew all the others did (knows),
    the first when all are equal (data_of), else the newest of those; None when none did (no run)."""
    found = [version for version in run if all(other is version or knows(version, other) for other in run)]
    if not found:
        return None
    first = run[0]
    if found[0] is first and all(_exact(first, version) for version in run):
        return first
    return found[len(found) - 1]


def collapse(versions: list, holder: Optional[CardInfo]) -> tuple:
    """
    One name's history without duplicates (design 7.4). An earlier reading of the same commit goes (_rereads). Of each
    run of consecutive versions with the same data (_runs: the blob, path, dates and parse bookkeeping are no data; a
    field a reading could not know is no difference) the one that knew all the others did stays (_knower); a run the
    card that follows it knew all of (knows: the current card, or the archived record ending the list) goes into that
    card. A reading that knew less never absorbs one that knew more. Either way the run's earliest start stays
    (carry). A version tagged `archived` is never dropped and ends its run (the versions after it are the card's next
    life). Repeated until nothing goes, so a second call changes nothing.

    Args:
        versions: The name's old_stats versions, oldest first (not changed)
        holder: The current card, or None

    Returns:
        (kept versions, dropped versions), each oldest first
    """
    kept, dropped = _rereads(versions, holder)
    while True:
        runs = _runs(kept)
        kept, gone = [], []
        for i, run in enumerate(runs):
            last = run[len(run) - 1]
            if i + 1 == len(runs) and holder is not None and all(knows(holder, member) for member in run):
                into = holder
            elif getattr(last, "archived", None):
                into = last
            else:
                into = _knower(run)
            for version in run:
                if version is into or getattr(version, "archived", None):
                    kept.append(version)
                else:
                    carry(version, into)
                    gone.append(version)
        if not gone:
            break
        dropped += gone
    dropped.sort(key=lambda v: next(i for i, x in enumerate(versions) if x is v))
    return (kept, dropped)


def _normalized(version: CardInfo, after: Optional[CardInfo]) -> bool:
    """One stored version brought to today's forms (normalize); whether it changed."""
    changed = False
    types = list(getattr(version, "types", None) or [])
    if any(t in TYPE_RENAMES for t in types):
        version.types = [TYPE_RENAMES.get(t, t) for t in types]
        changed = True
    if vars(version).get("created") is not None and not getattr(version, "archived", None):
        del vars(version)["created"]                  # the card's own date, kept on the card (and an archived record)
        changed = True
    if after is None or getattr(version, "archived", None):
        return changed                                # after a tag comes the card's next life
    old = (getattr(version, "series", None), getattr(version, "subseries", None))
    new = (getattr(after, "series", None), getattr(after, "subseries", None))
    if old == new or new[0] is None:
        return changed
    filled = old == (None, None) and version.card_type != CardType.CREATURE.value    # typed later (creature: folder)
    split = old[0] is not None and old[1] is None and new[1] == old[0]                # now a subseries under a parent
    if filled or split:
        version.series, version.subseries = new
        changed = True
    return changed


def normalize(stats: dict, old_stats: dict) -> list:
    """
    History's stored versions in today's forms, in place, newest first so a value reaches back through each version
    that lacked it: a series filled from the version after it when it had none (an item's series is typed in later,
    never read from its PSD; a creature's is its folder) or moved under the parent the 2026-10-20 split gave it; a
    retired type icon renamed; the card's creation date dropped from a version that is no archived record. Part of
    the one-time cleanup at each /update_stats (with none left it changes nothing).

    Returns:
        (name, timestamp) of each version changed
    """
    changed = []
    for name, versions in old_stats.items():
        holder = stats.get(name)
        for i in range(len(versions) - 1, -1, -1):
            after = versions[i + 1] if i + 1 < len(versions) else holder
            if _normalized(versions[i], after):
                changed.append((name, versions[i].timestamp))
    return changed


def dedupe(stats: dict, old_stats: dict) -> list:
    """
    collapse() over every name, in place: the one-time cleanup of the duplicates /update_stats archived before
    2026-10-22 (it runs at each /update_stats; with none left it changes nothing).

    Args:
        stats: {name: CardInfo}
        old_stats: {name: [CardInfo]}, oldest first

    Returns:
        (name, timestamp) of each version dropped
    """
    gone = []
    for name in list(old_stats):
        kept, dropped = collapse(old_stats[name], stats.get(name))
        if not dropped:
            continue
        gone += [(name, version.timestamp) for version in dropped]
        if kept:
            old_stats[name] = kept
        else:
            del old_stats[name]
    return gone


def meta(start: Optional[Commit], start_at: float, cause: str, legacy: bool = False) -> dict:
    """A version's `version` record, open-ended until close() sets its end."""
    return {"from": start.at if start else start_at, "from_commit": start.sha if start else None, "to": None,
            "to_commit": None, "cause": cause, "png_commit": None, "legacy": legacy}


def close(card: CardInfo, index: GitIndex, end_at: Optional[float], end: Optional[Commit] = None) -> None:
    """End a version at the next one's start (or its deleting commit) and pin the PNG of that moment (by log order
    when the end is in the index, else by time: an edits commit is no PSD or PNG commit)."""
    info = getattr(card, "version", None)
    if info is None:
        return
    info["to"], info["to_commit"] = (end.at if end else end_at), (end.sha if end else None)
    listed = end if end is not None and index.by_sha.get(end.sha) is end else None
    info["png_commit"] = index.png_commit(card.path, listed, info["to"]) if info["to"] is not None else None


def start_of(card: CardInfo, index: GitIndex, previous: Optional[CardInfo], others=()) -> dict:
    """
    A version's start from git: its blob's intro (psd; after a revert, the revert: blob_since; intro_of: not at another
    name of `others`); the previous version's blob again: reparse from the same intro, or moved from the commit that
    put the blob at this path (a folder's data); no blob in the index: legacy, at the stored timestamp.
    """
    blob = getattr(card, "blob_sha", None)
    intro = index.intro_of(blob, getattr(card, "name", None) or card_name(card.path or ""), others, card.path,
                           card.timestamp)
    if intro is None:
        return meta(None, card.timestamp, "legacy", legacy=True)
    if previous is not None and getattr(previous, "blob_sha", None) == blob:
        here = index.intro_at.get((blob, card.path))
        if previous.path != card.path and here is not None:
            return meta(here, here.at, "moved")
        return meta(intro, intro.at, "reparse")
    return meta(intro, intro.at, "psd")


def allows_high(name: str, holder: Optional[CardInfo]) -> bool:
    """Whether a name's history may hold stats above STAT_MAX (up to STAT_CAP): its card has allow_high_stats, it was
    on the retired exclusion list (HIGH_STATS_LISTED), or it has no current card to tell (archived: listed for the
    owner by high_values, never unknown)."""
    return holder is None or bool(getattr(holder, "allow_high_stats", False)) or name in HIGH_STATS_LISTED


def out_of_range(card: CardInfo, allowed: bool = False) -> list:
    """The values no card prints, as fields: a stat above STAT_CAP (three digits), or above STAT_MAX without an
    allowance (the version's allow_high_stats, or `allowed`: allows_high), stars outside 0..STARS_MAX."""
    found = []
    stats = getattr(card, "stats", None)
    high = allowed or getattr(card, "allow_high_stats", False)
    values = [getattr(stats, key, -1) for key in STAT_KEYS] if stats is not None else []
    if any(isinstance(v, int) and v > (STAT_CAP if high else STAT_MAX) for v in values):
        found.append("stats")
    stars = getattr(card, "stars", None)
    if isinstance(stars, int) and not 0 <= stars <= STARS_MAX:
        found.append("stars")
    return found


def unchecked(card: CardInfo) -> bool:
    """Whether a version was read before the range check (RANGE_PARSER): a value out of range then is a misparse no
    one saw. A later reading's was a problem when read; it stays, so an allowance added later keeps it."""
    return (getattr(card, "parser_version", None) or 0) < RANGE_PARSER


def unknown_values(stats: dict, old_stats: dict) -> list:
    """
    Stored versions' impossible values set to unknown (stats -1, stars None; out_of_range), in place, each noted in its
    problems: they can't be checked (no PSD of theirs is read again). Only the versions read before the range check
    (unchecked); a later one's are kept and listed (kept_values). Part of the one-time cleanup.

    Returns:
        (name, timestamp, fields) of each version changed
    """
    changed = []
    for name, versions in old_stats.items():
        allowed = allows_high(name, stats.get(name))
        for version in versions:
            fields = out_of_range(version, allowed) if unchecked(version) else []
            if not fields:
                continue
            was = []
            if "stats" in fields:
                s = version.stats
                was.append(f"stats {s.hp}/{s.defense}/{s.attack}/{s.speed}")
                version.stats = CardStats()
            if "stars" in fields:
                was.append(f"stars {version.stars}")
                version.stars = None
            version.problems = [*(getattr(version, "problems", None) or []), f"OUT OF RANGE: {', '.join(was)} (unknown)"]
            changed.append((name, version.timestamp, fields))
    return changed


def kept_values(stats: dict, old_stats: dict) -> list:
    """History values out of range that unknown_values keeps (read by RANGE_PARSER or later; an allow_high_stats entry
    allows a stat up to STAT_CAP), for the owner: the names, sorted."""
    return sorted({name for name, versions in old_stats.items() for v in versions
                   if not unchecked(v) and out_of_range(v, allows_high(name, stats.get(name)))})


def high_values(stats: dict, old_stats: dict) -> list:
    """History stats above STAT_MAX that an allowance keeps (allows_high), for the owner to check: (name, version
    timestamp, 'hp/def/atk/spd')."""
    found = []
    for name, versions in old_stats.items():
        for version in versions:
            s = getattr(version, "stats", None)
            values = [getattr(s, key, -1) for key in STAT_KEYS] if s is not None else []
            if any(isinstance(v, int) and v > STAT_MAX for v in values) and not out_of_range(version, True):
                found.append((name, version.timestamp, "/".join(str(v) for v in values)))
    return found


def _blob_of(card: CardInfo, index: GitIndex) -> Optional[str]:
    """A version's PSD blob: its own, else the one commit's at its path with its time (blob_at)."""
    own = getattr(card, "blob_sha", None)
    if own:
        return own
    found = index.blob_at(card.path, card.timestamp)
    return found[1] if found else None


def _read_at(path: str, blob: Optional[str], at: Optional[float], index: GitIndex) -> Optional[float]:
    """When a reading's PSD bytes came to its own path for the stretch holding them at `at` (the PNG beside it is that
    path's; after a revert, the revert's), else its date."""
    here = index.arrival(blob, path, at)
    return here.at if here is not None else at


def _only_types(a: CardInfo, b: CardInfo) -> bool:
    """Whether the newer version b differs from a in its types alone and knew all a knew besides (knows)."""
    try:
        if data_of(a)["types"] == data_of(b)["types"]:
            return False
    except (TypeError, AttributeError):
        return False
    other = deepcopy(a)
    other.types = list(getattr(b, "types", None) or [])
    return knows(b, other)


def _same_png(index: GitIndex, a_path: str, a_at: Optional[float], b_path: str, b_at: Optional[float]) -> bool:
    """Whether the card's PNG is the same blob at both starts, at least PNG_SLACK apart (no export between them)."""
    if a_at is None or b_at is None or b_at - a_at < PNG_SLACK:
        return False
    png = index.png_blob(a_path, a_at)
    return png is not None and png == index.png_blob(b_path, b_at)


def misreads(stats: dict, old_stats: dict, index: GitIndex) -> list:
    """
    Type misreads, in place: consecutive versions whose PSD blobs differ while the card's PNG stayed the same (the
    printed card did not change from the older one's reading to the newer one's start) and whose data differ in the type
    list alone are one version read twice: the older goes into the newer, its earlier start carried (the 2026-02-08 commits changed 208 PSDs, 66
    of them without a new PNG). Repeated until none is left, so a second call changes nothing.

    Returns:
        (name, start, types before, types after) of each version merged
    """
    merged = []
    for name in list(old_stats):
        merged += _misread(name, old_stats[name], stats.get(name), index)
        if not old_stats[name]:
            del old_stats[name]
    return merged


def _misread(name: str, versions: list, holder: Optional[CardInfo], index: GitIndex) -> list:
    """misreads() on one name's versions (changed in place) and its card."""
    merged = []
    while True:
        chain = [*versions, holder] if holder is not None else list(versions)
        found = None
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if getattr(a, "archived", None) or getattr(b, "archived", None):
                continue
            blobs = (_blob_of(a, index), _blob_of(b, index))
            if None in blobs or blobs[0] == blobs[1] or not _only_types(a, b):
                continue
            if _same_png(index, a.path, _read_at(a.path, blobs[0], a.timestamp, index), b.path, start(b)[0]):
                found = (a, b)                  # a's own reading at its path, b's data from its start (a re-read carried it)
                break
        if found is None:
            return merged
        a, b = found
        merged.append((name, start(a)[0], list(getattr(a, "types", None) or []), list(getattr(b, "types", None) or [])))
        carry(a, b)
        del versions[next(i for i, v in enumerate(versions) if v is a)]


def commits_of(stats: dict, old_stats: dict, index: GitIndex) -> list:
    """
    Starts without their commit (carried from a version /update_stats stored without one) given the one commit at the
    version's path with that time (blob_at), in place, as the seed pass does (_dated). Part of the one-time cleanup.

    Returns:
        The names of the versions and cards given one
    """
    found = []
    every = [(name, v) for name, versions in old_stats.items() for v in versions] + list(stats.items())
    for name, card in every:
        info = getattr(card, "version", None)
        if getattr(card, "duplicate_of", None) or not info or info.get("from") is None or info.get("from_commit"):
            continue
        hit = index.blob_at(card.path, info["from"])
        if hit is not None:
            info["from_commit"] = hit[0].sha
            found.append(name)
    return found


# ========================
# Recovery from the bot's PICKLE commits
# ========================
@dataclass
class Snapshot:
    """One commit of the bot's database: its sha, committer time and the pickles as they were; marked: they held a
    restored version (recover's, so it ran)."""
    sha: str
    at: float
    stats: dict
    old_stats: dict
    marked: bool = False


@dataclass
class Reading:
    """A card as one PICKLE held it, dated from git: its start and commit, its PSD blob, the fields it could not
    know (an older parser's missing ones, impossible values, a failed parse)."""
    card: CardInfo
    snapshot: str
    order: int                                    # the snapshot's place, oldest 0
    at: float = 0.0
    commit: Optional[str] = None
    blob: Optional[str] = None
    unknown: set = field(default_factory=set)
    read_at: Optional[float] = None               # when its blob came to its path (the PNG beside it: _read_at)


@dataclass
class Recovered:
    """What recover changed."""
    added: list = field(default_factory=list)     # (name, start, PICKLE sha) of each version restored
    starts: list = field(default_factory=list)    # (name, earlier start) of each start moved earlier
    archived: list = field(default_factory=list)  # names gone from both pickles, archived from their last reading
    undated: int = 0                              # readings whose PSD blob git can't tell (skipped)
    clashed: int = 0                              # readings at another card's path of the same name (skipped)
    done: bool = False                            # it ran before (recovered_before, Pickles.done): nothing read

    def changed(self) -> bool:
        return bool(self.added or self.starts or self.archived)

    def lines(self) -> list:
        if self.done:
            return ["Recovery from the bot's PICKLE commits: done before (a restored version in the database or a "
                    "PICKLE), skipped"]
        by = {}
        for _, _, sha in self.added:
            by[sha[:7]] = by.get(sha[:7], 0) + 1
        cards = len({name for name, _, _ in self.added})
        return [f"Versions recovered from the bot's PICKLE commits: {len(self.added)} on {cards} cards"
                + (f" ({', '.join(f'{sha} {n}' for sha, n in sorted(by.items()))})" if by else ""),
                f"Starts moved earlier by a PICKLE's reading: {len(self.starts)}",
                f"Cards gone before archiving, archived: {len(self.archived)}"
                + (f" ({', '.join(self.archived)})" if self.archived else ""),
                f"Readings git can't date (skipped): {self.undated}",
                f"Readings at another card's path of the same name (skipped): {self.clashed}"]


@dataclass
class Pickles:
    """What recover reads from the bot's PICKLE commits: each name's readings (_readings) and each commit's time,
    folded in one snapshot at a time (add), so no two snapshots are held at once."""
    found: dict = field(default_factory=dict)     # name -> {(path, timestamp): Reading}
    at_of: dict = field(default_factory=dict)     # PICKLE sha -> committer time
    shas: tuple = ()                              # the PICKLE commits read (or refused), oldest first
    done: bool = False                            # a PICKLE (a reverted one too) held a restored version: recover ran
    peeked: frozenset = frozenset()               # the reverted PICKLE commits looked at for that

    @property
    def readings(self) -> dict:
        """{name: [Reading]}: one per (path, timestamp), the newest PICKLE's."""
        return {name: list(by.values()) for name, by in self.found.items()}

    def add(self, snap: "Snapshot") -> None:
        """One more PICKLE commit (newer than those added): its readings replace older ones of the same record."""
        _fold(self.found, snap, len(self.shas))
        self.at_of[snap.sha] = snap.at
        self.shas = (*self.shas, snap.sha)
        self.done = self.done or snap.marked

    @classmethod
    def of(cls, snapshots) -> "Pickles":
        out = cls()
        for snap in snapshots:
            out.add(snap)
        return out


_PICKLES: dict = {}                                   # clone -> Pickles of its PICKLE commits: a new one is added to it


def _pickle_commits(clone: str) -> list:
    """(sha, committer time) of each commit of clone's HEAD history that changed stats.pkl or old_stats.pkl, oldest
    first, without the ones a later commit reverted (cca0aaf); [] when clone is no full git clone."""
    return _pickle_log(clone)[0]


def _pickle_log(clone: str) -> tuple:
    """(_pickle_commits, the reverted ones), each [(sha, committer time)] oldest first; ([], []) without a full clone."""
    env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
    if not clone or not isdir(clone):
        return ([], [])
    shallow = run(["git", "-C", clone, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, env=env)
    if shallow.returncode != 0 or shallow.stdout.strip() != "false":
        return ([], [])
    log = run(["git", "-C", clone, "log", "HEAD", "--format=%H%x09%ct%x09%B%x00", "--", *PICKLE_FILES],
              capture_output=True, env=env)
    if log.returncode != 0:
        return ([], [])
    commits, reverted = [], set()
    for chunk in log.stdout.decode("utf-8", "replace").split("\0"):
        chunk = chunk.strip("\n")
        if chunk:
            sha, at, body = (chunk.split("\t", 2) + ["", ""])[:3]
            reverted.update(findall(r"This reverts commit ([0-9a-f]{7,40})", body))
            commits.append((sha, float(at or 0)))
    gone = [(sha, at) for sha, at in reversed(commits) if any(sha.startswith(r) for r in reverted)]
    return ([c for c in reversed(commits) if c not in gone], gone)


def _snapshots(clone: str, commits: list, peek: bool = False):
    """
    A Snapshot for each (sha, time) of commits, one at a time: one git cat-file --batch process, a blob read when its
    snapshot is due, the allow-list unpickler; a pickle it refuses is one line and left out (None yielded, so the
    caller can count the commit read). Marked when the bytes hold RESTORED_MARK; peek: nothing unpickled (empty).
    """
    env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
    proc = Popen(["git", "-C", clone, "cat-file", "--batch"], stdin=PIPE, stdout=PIPE, env=env)
    try:
        for sha, at in commits:
            pickles = []
            for name in PICKLE_FILES:
                proc.stdin.write(f"{sha}:{name}\n".encode())
                proc.stdin.flush()
                header = proc.stdout.readline().split()
                data = None
                if len(header) == 3 and header[1] == b"blob":
                    data = proc.stdout.read(int(header[2]))
                    proc.stdout.read(1)
                pickles.append(data)
            marked = any(data is not None and RESTORED_MARK in data for data in pickles)
            if peek:
                yield (sha, Snapshot(sha, at, {}, {}, marked))
                continue
            try:
                loaded = [SeedUnpickler(BytesIO(data)).load() if data is not None else {} for data in pickles]
            except Exception as e:                      # noqa: BLE001 - an unreadable PICKLE is one line, never a failed run
                print(f"History: PICKLE {sha[:7]} not read ({e})")
                yield (sha, None)
                continue
            yield (sha, Snapshot(sha, at, *(p if isinstance(p, dict) else {} for p in loaded), marked))
    finally:
        proc.stdin.close()
        proc.stdout.close()
        proc.wait()


def bot_snapshots(clone: str) -> list:
    """The bot's committed database: a Snapshot for each of _pickle_commits(clone), oldest first; [] without a full
    clone (the Action's sparse checkout, a git archive). All in memory: for tools and tests (bot_pickles reads one at a
    time)."""
    commits = _pickle_commits(clone)
    return [snap for _, snap in _snapshots(clone, commits) if snap is not None] if commits else []


def bot_pickles(clone: str) -> Optional[Pickles]:
    """recover's input from clone's PICKLE commits, kept for the process: a new PICKLE commit (each /push) is the only
    one read then (the list grew at its end; else all again), each snapshot unpickled, folded in and dropped before the
    next; a reverted one only looked at for RESTORED_MARK (a reverted recovery stays undone: done); None without a full
    clone."""
    commits, reverted = _pickle_log(clone)
    if not commits:
        return None
    key, shas = abspath(clone), tuple(sha for sha, _ in commits)
    have = _PICKLES.get(key)
    if have is None or shas[:len(have.shas)] != have.shas:
        have = Pickles()
    for sha, snap in _snapshots(clone, commits[len(have.shas):]):
        if snap is not None:
            have.add(snap)
        else:
            have.shas = (*have.shas, sha)
    for sha, snap in _snapshots(clone, [c for c in reverted if c[0] not in have.peeked], peek=True):
        have.done = have.done or snap.marked
        have.peeked = have.peeked | {sha}
    _PICKLES.clear()
    _PICKLES[key] = have
    return have


def _as_card(key: str, value) -> Optional[tuple]:
    """A PICKLE's record as (a new CardInfo, the fields it lacks): a CardInfo through record (an older pickle's missing
    fields defaulted), a 2025 dict (keyed by path in April 2025) converted."""
    if isinstance(value, CardInfo):
        try:
            found = card(record(value))
        except (TypeError, AttributeError, ValueError):
            return None
        found.name = key
        return (found, {"stars"} if found.stars == 0 else set())
    if not isinstance(value, dict):
        return None
    path = value.get("path") or (key if str(key).lower().endswith(".psd") else "")
    if not path:
        return None
    lacks = {f for f, k in (("stars", "stars"), ("stats", "hp")) if k not in value}
    if value.get("stars") == 0:
        lacks.add("stars")                          # none found (no card prints 0 stars; NO STARS FOUND since)
    stats = CardStats(*(value.get(k, -1) for k in ("hp", "def", "atk", "spd"))) if "stats" not in lacks else CardStats()
    found = CardInfo(name=card_name(path), card_type=value.get("type") or "unknown", path=path,
                     timestamp=float(value.get("timestamp") or 0.0), ability=value.get("ability"),
                     stars=value.get("stars"), subtype=value.get("subtype"), series=value.get("series"),
                     types=list(value.get("types") or []), stats=stats, problems=list(value.get("problem") or []),
                     author=value.get("author"))
    return (found, lacks)


def _readings(snapshots: list) -> dict:
    """{name: [Reading]} of every card record the PICKLEs held (current and history), one per (path, timestamp): the
    newest PICKLE's (its parser read best). Left out: no real date (local mode), no card folder, duplicates, no data."""
    return Pickles.of(snapshots).readings


def _fold(found: dict, snap: Snapshot, order: int) -> None:
    """One snapshot's records into found ({name: {(path, timestamp): Reading}}; _readings), in place."""
    records = [(k, v) for k, v in snap.stats.items()]
    records += [(k, v) for k, vs in snap.old_stats.items() for v in (vs if isinstance(vs, list) else [])]
    for key, value in records:
        got = _as_card(str(key), value)
        if got is None:
            continue
        reading, lacks = got
        path = reading.path or ""
        if (reading.timestamp <= 0 or not path.lower().endswith(".psd") or getattr(reading, "duplicate_of", None)
                or any(folder in RETIRED for folder in path.split("/"))
                or (reading.card_type == CardType.UNKNOWN.value and not reading.ability)):
            continue
        reading.archived = reading.version = None
        vars(reading).pop("created", None)
        found.setdefault(reading.name, {})[(path, reading.timestamp)] = Reading(reading, snap.sha, order, unknown=lacks)


def _locate(reading: Reading, index: GitIndex, snapshot_at: float) -> bool:
    """A reading's blob and start from git: the commit at its path with its timestamp (the bot stored commit times
    since 2025-05), else the one blob its path held from that time (a file's mtime in April 2025) to its PICKLE, dated
    by the commit that put it at that path (a reading never stands for a time its file sat at another path), since
    the path last held other bytes (a revert dates from the revert: King Dedede's 2025-04-03). False when git can't
    tell."""
    found = index.blob_at(reading.card.path, reading.card.timestamp)
    if found is not None:
        reading.at, reading.commit, reading.blob = reading.card.timestamp, found[0].sha, found[1]
    else:
        when = min(reading.card.timestamp, snapshot_at)
        first = index.blob_before(reading.card.path, when)
        if first is None or first != index.blob_before(reading.card.path, snapshot_at):
            return False
        intro = index.arrival(first, reading.card.path, when)
        if intro is None:
            return False
        reading.at, reading.commit, reading.blob = intro.at, intro.sha, first
    reading.read_at = _read_at(reading.card.path, reading.blob, reading.at, index)
    return True


def _words(text: Optional[str]) -> list:
    """An ability's printed words, in any case: icon tokens ([fire], [x:tint]) and everything but letters and digits
    dropped (spacing, punctuation and capitals are format)."""
    return findall(r"[^\W_]+", sub(r"\[[^\]]*\]", " ", text or "").lower())


def _printed(card: CardInfo, lacks: set, allowed: bool) -> dict:
    """What a reading prints (PRINTED), without the fields it could not know: each stat apart (-1: not read, as
    Uumuu's 2025 DEF)."""
    stats = getattr(card, "stats", None)
    out = {"card_type": card.card_type, "ability": _words(card.ability),
           "types": sorted(TYPE_RENAMES.get(t, t) for t in getattr(card, "types", None) or []),
           "stars": getattr(card, "stars", None),
           **{key: getattr(stats, key, -1) if stats is not None else -1 for key in STAT_KEYS}}
    skip = set(lacks) | set(out_of_range(card, allowed)) | (unknown(card) - {"display_name"})
    if "stats" in skip:
        skip.update(STAT_KEYS)
    if not out["ability"]:
        if (getattr(card, "parser_version", None) or 0) >= 2 and card.ability == "" and "ability" not in skip:
            out["ability"] = PRINTS_NONE            # parser 2 reads a hidden Ability layer as nothing printed
        else:
            skip.add("ability")                     # none read (a failed parse's empty text; data_of: none)
    if out["stars"] is None:                        # none counted: no field a reading can disagree on
        skip.add("stars")
    skip.update(key for key in STAT_KEYS if out[key] == -1)
    return {k: v for k, v in out.items() if k not in skip}


def _same_printed(a: dict, b: dict) -> bool:
    """Equal on what both know; an ability parser 2 found printed nowhere matches any older reading's words (those
    parsers read hidden layers too)."""
    return all(a[k] == b[k] or (k == "ability" and PRINTS_NONE in (a[k], b[k])) for k in a if k in b)


def recovered_before(stats: dict, old_stats: dict) -> bool:
    """Whether the database holds a restored version (recover ran: it is one time only, so a version removed later on
    purpose, a fold or a revert of its PICKLE, stays removed)."""
    every = [v for versions in old_stats.values() for v in versions] + list(stats.values())
    return any((getattr(v, "version", None) or {}).get("restored_from") for v in every)


def recover(stats: dict, old_stats: dict, index: GitIndex, pickles: Pickles, once: bool = True) -> Recovered:
    """
    History the bot's own PICKLE commits held and old_stats lost (the 2026-02-07 cleanup dropped all but 208 names; the
    2025 pickles held states no backfill read; 795fd23 replaced Spawn of the Forbidden One's June reading), in place.
    Each name's readings (_readings) dated from git (_locate); one with the PSD blob of a stored version is a re-read:
    its earlier start carried (never past the version before, nor across other bytes at its path: a revert dates from
    the revert). The others, in time order between the stored versions:
    a reading printing what the version before it prints is nothing new; a run of readings printing the same goes, as
    one, into the version after it when that prints it too (its earlier start carried: a format or parser change) or
    becomes a legacy version (the old parse's ability, types, stars, stats, card type; the next version's series,
    title, author, note, hidden and creatures; restored_from the PICKLE). A type list alone that differs while the PNG
    stayed is a misread (misreads), never a version. A name gone from both pickles gets its readings as versions, the
    last archived from its deleting commit (renamed_to, moved_to by date_archive); no deleting commit, nothing.
    One time only (once): nothing when the database or a PICKLE commit (a reverted one too) held a restored version
    (recovered_before, Pickles.done). Idempotent besides: a second call finds every reading held.

    Args:
        stats, old_stats: The database (changed in place)
        index: The card repo's GitIndex
        pickles: bot_pickles() (or Pickles.of(snapshots))
        once: False: run though it ran before (tests of the rules)

    Returns:
        What changed
    """
    report = Recovered()
    if once and (pickles.done or recovered_before(stats, old_stats)):
        report.done = True
        return report
    at_of = pickles.at_of
    for name, readings in pickles.readings.items():
        holder = stats.get(name)
        if holder is not None and getattr(holder, "duplicate_of", None):
            continue
        allowed = allows_high(name, holder)
        stored = list(old_stats.get(name) or [])
        items = [*stored, holder] if holder is not None else stored
        readings, clashed = _own_path(readings, items, index)
        report.clashed += clashed
        located: dict = {}
        for reading in readings:
            if not _locate(reading, index, at_of[reading.snapshot]):
                report.undated += 1
                continue
            key = (reading.blob, reading.card.path, reading.read_at)   # the same bytes in another folder print
            have = located.get(key)                                    # another card type; back after others: a revert
            if have is None:
                located[key] = reading
            else:                                   # one blob read twice at one path: the newer parse, the earlier start
                newer = reading if reading.order > have.order else have
                newer.at, newer.commit = min((reading.at, reading.commit), (have.at, have.commit),
                                             key=lambda pair: pair[0])
                located[key] = newer
        if not items:
            _archive_gone(name, sorted(located.values(), key=lambda r: r.at), old_stats, stats, index, report)
            continue
        if getattr(items[len(items) - 1], "archived", None) is None and holder is None:
            continue                                # an untagged gone name: prune's to tag first
        keys = [(_blob_of(v, index), v.path) for v in items]
        fresh = []
        for reading in sorted(located.values(), key=lambda r: r.at):
            if (reading.blob, reading.card.path) in keys:
                k = keys.index((reading.blob, reading.card.path))
                begin = start(items[k])[0]
                if begin is not None and reading.at < begin and (k == 0 or (start(items[k - 1])[0] or 0) < reading.at) \
                        and index.held(reading.card.path, reading.blob, reading.at, begin):   # never across a revert
                    carry_to(items[k], reading.at, reading.commit)
                    report.starts.append((name, reading.at))
            else:
                fresh.append(reading)
        if fresh:
            _merge(name, fresh, items, holder, old_stats, index, allowed, report)
    return report


def _own_path(readings: list, items: list, index: GitIndex) -> tuple:
    """
    readings without those at another card's path of the same name: a PICKLE that held the name at two paths (the
    2025 ones are keyed by path: Auxiliary/Minions/Auto_Croissant.psd, now Mini Auto Croissant, beside the creature)
    keeps the path of the name's stored versions (or one an exact rename chain brought them from); neither: none.

    Returns:
        (the readings kept, how many were left out)
    """
    by: dict = {}
    for reading in readings:
        by.setdefault(reading.snapshot, set()).add(reading.card.path)
    clash = set().union(*(paths for paths in by.values() if len(paths) > 1))
    if not clash:
        return (readings, 0)
    own = set()
    for item in items:
        path, seen = item.path, set()
        while path and path not in seen:
            seen.add(path)
            own.add(path)
            path = index.renamed_from.get(path)
    foreign = clash - own if clash & own else clash
    kept = [r for r in readings if r.card.path not in foreign]
    return (kept, len(readings) - len(kept))


def _merge(name: str, fresh: list, items: list, holder: Optional[CardInfo], old_stats: dict, index: GitIndex,
           allowed: bool, report: Recovered) -> None:
    """recover's placing of one name's new readings among its stored versions (items: old_stats, then the card)."""
    starts = [start(v)[0] or 0.0 for v in items]
    if any(starts[i] > starts[i + 1] for i in range(len(starts) - 1)):
        return                                      # stored out of order: nothing is placed between them
    gaps: dict = {}
    for reading in fresh:
        k = bisect_right(starts, reading.at)
        if k >= len(items) or (k and starts[k - 1] == reading.at):
            continue                                # newer than the card, or a re-read of the version's commit
        gaps.setdefault(k, []).append(reading)
    placed: dict = {}                               # item index -> versions restored before it
    for k, readings in sorted(gaps.items()):
        before = items[k - 1] if k else None
        runs = _runs_of(readings, index, allowed, (_printed(before, set(), allowed), _read_at(
            before.path, _blob_of(before, index), before.timestamp, index), before.path) if before is not None else None)
        after = items[k]
        held = (_printed(after, set(), allowed), starts[k], after.path)
        if runs and _alike(index, _form(runs[len(runs) - 1], allowed), held) and \
                set(_form(runs[len(runs) - 1], allowed, best=True)[0]) <= set(held[0]):   # the version knows it all
            first = _first(runs.pop())                        # it prints what the version after it prints
            carry_to(after, first.at, first.commit)
            report.starts.append((name, first.at))
        if runs:
            placed[k] = [_restored(name, run_, after, report) for run_ in runs]
    if not placed:
        return
    out = []
    for k, item in enumerate(items):
        out += placed.get(k, [])
        if item is not holder:
            out.append(item)
    old_stats[name] = out


def _form(run_: list, allowed: bool, best: bool = False) -> tuple:
    """A run's printed form for _alike: (printed, commit time, path) of its first reading (best: of the one that knew
    most, _restored's)."""
    first = max(run_, key=lambda r: (-len(r.unknown), r.order)) if best else run_[0]
    return (_printed(first.card, first.unknown, allowed), first.read_at or first.at, first.card.path)


def _runs_of(readings: list, index: GitIndex, allowed: bool, before: Optional[tuple] = None) -> list:
    """Readings in time order as runs printing the same (_alike); the ones printing what `before` (a version's form)
    prints go, when it knew all they knew (a reading that knew less never stands for one that knew more)."""
    runs: list = []
    for reading in readings:
        mine = (_printed(reading.card, reading.unknown, allowed), reading.read_at or reading.at, reading.card.path)
        if runs and _alike(index, _form(runs[len(runs) - 1], allowed), mine):
            runs[len(runs) - 1].append(reading)
        elif not runs and before is not None and _alike(index, before, mine) and set(mine[0]) <= set(before[0]):
            continue
        else:
            runs.append([reading])
    return runs


def _alike(index: GitIndex, a: tuple, b: tuple) -> bool:
    """Whether two printed forms (printed, its reading's commit time, path) are the same card: equal on what both know,
    or the type list alone differs while the PNG stayed (a misread)."""
    if _same_printed(a[0], b[0]):
        return True
    if {k for k in a[0] if k in b[0] and a[0][k] != b[0][k]} != {"types"}:
        return False
    first, second = sorted((a, b), key=lambda x: x[1])
    return _same_png(index, first[2], first[1], second[2], second[1])


def _first(run_: list) -> Reading:
    """A run's earliest reading, never before a later stretch of bytes the run read at a path twice (other bytes came
    and went between: a revert dates from the revert)."""
    floor = max((r.read_at for r in run_ for o in run_ if (o.blob, o.card.path) == (r.blob, r.card.path)
                 and (o.read_at or 0) < (r.read_at or 0)), default=None)
    return min((r for r in run_ if floor is None or r.at >= floor), key=lambda r: r.at)


def _restored(name: str, run_: list, after: Optional[CardInfo], report: Recovered) -> CardInfo:
    """A run of readings as one legacy version: the reading that knew most (the newest of those), the run's earliest
    start (_first) with that reading's blob and commit (a pair git had), the next version's non-PSD fields (the old pickles
    lack them or used another series scheme)."""
    best = max(run_, key=lambda r: (-len(r.unknown), r.order))
    first = _first(run_)
    version = deepcopy(best.card)
    version.name = name
    for key in ("parsed_values", "duplicate_of", "alt_names", "archived"):
        setattr(version, key, None)
    if after is not None:
        for key in ("series", "subseries", "display_name", "author", "note", "hidden", "creatures"):
            setattr(version, key, deepcopy(getattr(after, key, None)))
    if "stars" in best.unknown:
        version.stars = None
    if "stats" in best.unknown:
        version.stats = CardStats()
    version.blob_sha = first.blob
    version.timestamp = first.at
    version.version = {**meta(None, first.at, "legacy", legacy=not first.blob), "from_commit": first.commit,
                       "restored_from": best.snapshot}
    vars(version).pop("created", None)
    report.added.append((name, first.at, best.snapshot))
    return version


def _archive_gone(name: str, located: list, old_stats: dict, stats: dict, index: GitIndex, report: Recovered) -> None:
    """A name in neither pickle: its readings as versions (runs printing the same as one), the last tagged archived
    with its deleting commit and the bytes it deleted (Chestnut's 4155a8f, edited after its last reading b7d4546);
    nothing when git has no deletion of its last path (it may be listed under another name)."""
    if not located:
        return
    path = located[len(located) - 1].card.path
    gone = index.deleted.get(path)
    if gone is None or index.newest.get(path) is not gone:
        return
    newest = max(located, key=lambda r: (r.at, r.order)).card
    versions = [_restored(name, run_, newest, report) for run_ in _runs_of(located, index, allows_high(name, None))]
    last = versions[len(versions) - 1]
    deleted = next((b for c, b in index.at_path.get(path, []) if c.order > gone.order), None)   # the bytes git deleted
    last.archived = {"at": gone.at, "path": last.path, "blob": deleted or last.blob_sha, "commit": gone.sha,
                     "backfill": True}
    old_stats[name] = versions
    report.archived.append(name)


# ========================
# The seed pass
# ========================
@dataclass
class Refined:
    """What refine changed (each count once)."""
    dropped: list = field(default_factory=list)       # (name, version timestamp) of each duplicate version dropped
    normalized: list = field(default_factory=list)    # (name, version timestamp) of each version normalize() changed
    versions_dated: int = 0                           # versions with a blob whose timestamp moved to the blob's intro
    versions_with_blob: int = 0
    legacy: int = 0                                   # versions without a blob, marked legacy (before the drop)
    blobs_found: int = 0                              # blob-less versions given the blob of their commit (blob_at)
    commits_found: int = 0                            # carried starts given their commit (blob_at)
    legacy_kept: int = 0                              # of those, still versions after it
    cards_dated: list = field(default_factory=list)   # current cards (no duplicates) re-dated
    duplicates_dated: int = 0                         # duplicates that followed their original
    cards: int = 0                                    # current cards that are no duplicate
    tagged: list = field(default_factory=list)        # archive tags added (backfill)
    tags_dated: list = field(default_factory=list)    # archive tags given git's date and commit
    renamed: list = field(default_factory=list)       # (old, new) exact renames recorded on tags
    retired: list = field(default_factory=list)       # names whose PSD moved into a folder of no cards (moved_to)
    recovered: Optional["Recovered"] = None           # recover() from the bot's PICKLE commits, when it ran
    unknown: list = field(default_factory=list)       # (name, timestamp, fields) of impossible values set to unknown
    misread: list = field(default_factory=list)       # (name, start, types, types) of type misreads merged

    def changed(self) -> bool:
        return bool(self.dropped or self.normalized or self.versions_dated or self.cards_dated or self.duplicates_dated
                    or self.tagged or self.tags_dated or self.renamed or self.retired or self.blobs_found
                    or self.commits_found or (self.recovered and self.recovered.changed()) or self.unknown
                    or self.misread)

    def lines(self) -> list:
        def names(items):
            return f" ({', '.join(items)})" if items else ""
        return [f"Duplicate versions dropped: {len(self.dropped)}{names([n for n, _ in self.dropped])}",
                f"Versions normalized (series, type names): {len(self.normalized)}",
                f"Versions re-dated: {self.versions_dated} of {self.versions_with_blob} with a blob",
                f"Versions marked legacy (no blob): {self.legacy}, {self.legacy_kept} after the drop",
                f"Legacy versions given their commit's blob: {self.blobs_found}; carried starts given their commit: "
                f"{self.commits_found}",
                f"Cards re-dated: {len(self.cards_dated)} of {self.cards} (and {self.duplicates_dated} duplicates follow)",
                f"Archive tags added: {len(self.tagged)}{names(self.tagged)}",
                f"Archive tags dated from git: {len(self.tags_dated)}{names(self.tags_dated)}",
                f"Renames recorded: {len(self.renamed)}{names([f'{a} -> {b}' for a, b in self.renamed])}",
                f"Moves into MDW recorded: {len(self.retired)}{names(self.retired)}",
                f"Impossible values set to unknown: {len(self.unknown)}{names(sorted({n for n, _, _ in self.unknown}))}",
                f"Type misreads merged (the PNG stayed): {len(self.misread)}{names([n for n, _, _, _ in self.misread])}",
                *(self.recovered.lines() if self.recovered is not None else ["Recovery from the bot's PICKLE commits: "
                                                                            "skipped (no clone of the bot's repo)"])]


def date_cards(stats: dict, index: GitIndex, report: Refined, others=()) -> None:
    """Each current card's timestamp becomes its blob's intro (blob_since, intro_of `others`); a duplicate takes its
    original's."""
    for name, card in stats.items():
        if getattr(card, "duplicate_of", None):
            continue
        report.cards += 1
        intro = index.intro_of(getattr(card, "blob_sha", None), name, others, card.path, card.timestamp)
        if intro is not None and intro.at != card.timestamp:
            card.timestamp = intro.at
            report.cards_dated.append(name)
    for card in stats.values():
        original = stats.get(getattr(card, "duplicate_of", None) or "")
        if original is not None and card.timestamp != original.timestamp:
            card.timestamp = original.timestamp
            report.duplicates_dated += 1


def date_archive(stats: dict, old_stats: dict, index: GitIndex, report: Refined, backfill: bool = True) -> None:
    """Names gone from stats: the last version tagged archived (backfill) if untagged (not with backfill False:
    /update_stats leaves that to its guarded prune); every tag's at and commit from its path's deleting commit (an
    archived duplicate's from its PNG's, or its original's when the PNG stayed), renamed_to from an exact rename chain
    that ends at a listed card (else its PNG's: a rename that also edited the PSD), moved_to from one into a folder of
    no cards (MDW/The Morgue: retired, not deleted), either dated by the chain's step that reached it (Pandiety's move
    into MDW, not its rename to Pandeity 44 min before; the new name starts there too: intro_of). A deletion or rename older than the record's own reading, or an
    index that lacks the tag's commit (a clone not fetched since: a path deleted, added back and deleted again),
    changes nothing."""
    for name, versions in sorted(old_stats.items(), key=lambda item: bool(getattr(
            item[1][len(item[1]) - 1], "duplicate_of", None) if item[1] else False)):     # originals before their copies
        if not versions or name in stats:
            continue
        last = versions[len(versions) - 1]
        original = getattr(last, "duplicate_of", None)
        if original and not getattr(last, "archived", None):
            continue
        if not getattr(last, "archived", None):
            if not backfill:
                continue
            last.archived = {"at": None, "path": last.path, "blob": getattr(last, "blob_sha", None) or None,
                             "commit": None, "backfill": True}
            report.tagged.append(name)
        tag = last.archived
        read = getattr(last, "timestamp", None) or 0.0
        stale = bool(tag.get("commit")) and tag["commit"] not in index.by_sha      # the index predates the tag's commit
        gone = index.deleted.get(tag.get("path") or "")
        if original and (gone is None or index.newest.get(tag.get("path") or "") is not gone):
            mine = old_stats.get(original) or []                # its PNG is there: it left with its original's PSD
            theirs = getattr(mine[len(mine) - 1], "archived", None) if mine and original not in stats else None
            gone = index.by_sha.get((theirs or {}).get("commit") or "")
        new, retired, step = (None, None, None) if original or stale else index.renamed(tag.get("path") or "", stats, read)
        if (tag.get("renamed_to") or tag.get("moved_to")) and (new, retired) != (tag.get("renamed_to"), tag.get("moved_to")):
            step = None                                         # linked otherwise (a commit page, a new blob)
        if step is not None and (gone is None or step.at >= gone.at):
            gone = step                                         # the chain's step that reached the card or MDW
        if gone is not None and not stale and gone.at >= read and (tag.get("at"), tag.get("commit")) != (gone.at, gone.sha):
            tag["at"], tag["commit"] = gone.at, gone.sha
            report.tags_dated.append(name)
        if original or stale or tag.get("renamed_to") or tag.get("moved_to"):
            continue
        if new:
            tag["renamed_to"] = new
            report.renamed.append((name, new))
        elif retired:
            tag["moved_to"] = retired
            report.retired.append(name)


def date_versions(name: str, stats: dict, old_stats: dict, index: GitIndex, report: Refined) -> None:
    """
    One name's versions: each one's `version` (kept when the seed, a build or recover decided it; dated again from
    git when /update_stats made it) and, with a blob, its timestamp from the blob's intro (intro_of the database's
    names; a restored version keeps its own); duplicates dropped (collapse), their starts carried; then each end set
    to the next start, the deleting commit, or open for the current card's.
    """
    versions = old_stats[name]
    current = stats.get(name)
    others = set(stats) | set(old_stats)
    previous = None
    for version in versions:
        if not getattr(version, "blob_sha", None):
            found = index.blob_at(version.path, version.timestamp)
            if found is not None:                   # its commit is known: its blob too (no longer approximate)
                version.blob_sha = found[1]
                report.blobs_found += 1
        version.version = _dated(version, index, previous, report, others)
        info = version.version
        if info["legacy"] and not getattr(version, "blob_sha", None):
            report.legacy += 1
        elif getattr(version, "blob_sha", None):
            report.versions_with_blob += 1
            intro = None if info.get("restored_from") else index.intro_of(version.blob_sha, name, others, version.path,
                                                                          version.timestamp)
            if intro is not None and version.timestamp != intro.at:
                version.timestamp = intro.at
                report.versions_dated += 1
        previous = version
    if current is not None:
        current.version = _dated(current, index, versions[len(versions) - 1] if versions else None, report, others)

    report.misread += _misread(name, versions, current, index)
    kept, dropped = collapse(versions, current)
    report.dropped += [(name, version.timestamp) for version in dropped]
    report.legacy_kept += sum(1 for v in kept if v.version["legacy"] and not getattr(v, "blob_sha", None))
    for i, version in enumerate(kept):
        if i + 1 < len(kept):
            end = kept[i + 1].version
            close(version, index, end["from"], index.by_sha.get(end["from_commit"] or ""))
        elif current is not None:
            close(version, index, current.version["from"], index.by_sha.get(current.version["from_commit"] or ""))
        elif getattr(version, "archived", None):
            tag = version.archived
            close(version, index, tag.get("at"), index.by_sha.get(tag.get("commit") or ""))
    if kept:
        old_stats[name] = kept
    else:
        del old_stats[name]


def _dated(card: CardInfo, index: GitIndex, previous: Optional[CardInfo], report: Optional[Refined] = None,
           others=()) -> dict:
    """card's `version` for the seed: its own when the seed, a build or recover (restored_from: dated from git where
    the PICKLE read it) decided it; else start_of(), keeping an earlier start that /update_stats carried (its records
    are `legacy`, made without git), with that start's commit when one commit at the card's path has its time
    (blob_at)."""
    info = getattr(card, "version", None)
    if info is not None and (info.get("cause") != "legacy" or (info.get("restored_from") and info.get("from_commit"))):
        return info
    fresh = start_of(card, index, previous, others)
    if info and info.get("restored_from"):
        fresh["restored_from"] = info["restored_from"]
    if info and info.get("from") is not None and (fresh["from"] is None or info["from"] < fresh["from"]):
        fresh["from"], fresh["from_commit"] = info["from"], info.get("from_commit")
        found = index.blob_at(card.path, fresh["from"])
        if found is not None and fresh["from_commit"] in (None, found[0].sha):
            if fresh["from_commit"] is None:
                fresh["from_commit"] = found[0].sha
                if report is not None:
                    report.commits_found += 1
            intro = index.intro_of(found[1], getattr(card, "name", None) or card_name(card.path or ""), others,
                                   card.path, fresh["from"])
            if intro is not None and intro.at < fresh["from"] and getattr(previous, "blob_sha", None) != found[1]:
                fresh["from"], fresh["from_commit"] = intro.at, intro.sha   # the carried version's bytes, dated as
    return fresh                                                            # start_of dates a version (no move)


def refine(stats: dict, old_stats: dict, index: GitIndex, pickles: Optional[Pickles] = None) -> Refined:
    """
    The seed pass, in place: history the bot's PICKLE commits held recovered (with pickles), stored versions
    normalized and their impossible values unknown, cards and versions dated from git, type misreads and duplicate
    versions dropped (collapse; never an archived one), blob-less versions marked legacy, archive tags added and dated,
    every version and current card given its `version` record. Idempotent: a second pass changes nothing.

    Args:
        stats: {name: CardInfo} (stats.pkl)
        old_stats: {name: [CardInfo]} (old_stats.pkl), oldest first
        index: The card repo's GitIndex
        pickles: bot_pickles() of a clone of the bot's repo (None: no recovery)

    Returns:
        The counts
    """
    report = Refined()
    if pickles is not None:
        report.recovered = recover(stats, old_stats, index, pickles)
    report.normalized = normalize(stats, old_stats)
    report.unknown = unknown_values(stats, old_stats)
    date_cards(stats, index, report, set(stats) | set(old_stats))
    date_archive(stats, old_stats, index, report)
    names = list(old_stats)
    for name in names:
        date_versions(name, stats, old_stats, index, report)
    names, others = set(names), set(stats) | set(old_stats)
    for name, card in stats.items():
        if not getattr(card, "duplicate_of", None) and name not in names:
            card.version = _dated(card, index, None, report, others)
    return report


# ========================
# A build's history
# ========================
def record_versions(before: dict, archived_before: dict, db, reparsed: dict, restored: list, index: GitIndex,
                    edits: Optional[Commit] = None, authors: Optional[Commit] = None) -> list:
    """
    A build's versions, after its parse, edits, authors and archive: a card whose data changed has its card before
    appended to its history (at its own path), closed where the new version starts (never before that version's own
    start: a held PSD change or a re-read after an edit); an art-only change or a move keeps the version going; a
    re-read of the same PSD (same blob and path: a parser change) replaces its reading, its start kept; a card
    restored with the same data drops its archived record. A reading that knew less (unknown: a parse that failed on
    a field) keeps the card before as a version. Duplicates get none.

    Args:
        before: stats at the build's start (a deep copy)
        archived_before: the archived cards at its start ({name: last version})
        db: The StatsDatabase after the build (changed in place)
        reparsed: names parsed from their PSD this build
        restored: names whose archived record came back to stats
        index: The card repo's GitIndex
        edits: The newest commit of Card Edits.json (an edit's start), if known
        authors: The newest commit of Card Authors.txt (an authors change's start), if known

    Returns:
        (name, cause) of each new version
    """
    known = {c.sha: c for c in (edits, authors) if c is not None}
    others = set(db.stats) | set(db.old_stats)          # a rename's new name starts where its archive ends (intro_of)

    def end_of(start: dict) -> Optional[Commit]:
        sha = start["from_commit"] or ""
        return index.by_sha.get(sha) or known.get(sha)
    made = []
    for name, card in db.stats.items():
        if getattr(card, "duplicate_of", None):
            continue
        old = before.get(name)
        if name in restored:
            versions = db.old_stats.get(name) or []
            old = versions[len(versions) - 1] if versions else archived_before.get(name)
        if old is None:
            if getattr(card, "version", None) is None:
                card.version = start_of(card, index, None, others)
            continue
        if knows(card, old):                          # the new reading knew all the old one did: no version
            info = deepcopy(getattr(old, "version", None)) or start_of(card, index, None, others)
            info["to"], info["to_commit"], info["png_commit"] = None, None, None
            card.version = info
            if name in restored:                      # back with the same data: its archived record is no version
                versions = db.old_stats.get(name) or []
                if versions and versions[len(versions) - 1] is old and getattr(old, "archived", None):
                    versions.pop()
                    if not versions:
                        del db.old_stats[name]
            continue
        if name in reparsed and name not in restored and old.path == card.path and unknown(card) <= unknown(old) and \
                getattr(old, "blob_sha", None) and getattr(old, "blob_sha", None) == getattr(card, "blob_sha", None):
            info = deepcopy(getattr(old, "version", None)) or start_of(card, index, None, others)   # read again:
            info["to"], info["to_commit"], info["png_commit"] = None, None, None            # the reading is replaced
            card.version = info
            made.append((name, "reparse"))
            continue
        if name in reparsed:
            start = start_of(card, index, old, others)
        else:
            only_author = {k for k, v in data_of(old).items() if v != data_of(card)[k]} == {"author"}
            source = authors if only_author else edits
            start = meta(source, source.at if source else card.timestamp, "authors" if only_author else "edit")
        prior = getattr(old, "version", None) or {}
        if prior.get("from") is not None and start["from"] is not None and start["from"] < prior["from"]:
            start["from"], start["from_commit"] = prior["from"], prior.get("from_commit")   # never before the one it ends
        card.version = start
        if name in restored:
            close(old, index, start["from"], end_of(start))
        else:
            previous = deepcopy(old)
            vars(previous).pop("created", None)               # the card's date: the card (or its archive) keeps it
            if getattr(previous, "version", None) is None:
                previous.version = start_of(previous, index, None, others)
            close(previous, index, start["from"], end_of(start))
            db.old_stats.setdefault(name, []).append(previous)
        made.append((name, start["cause"]))
    return made


def record_edits(db, before: dict, at: float, commit: Optional[str] = None) -> list:
    """
    Legacy (/update_stats, a Discord edit's in-memory refresh): an applied database edit that changed a card's data
    (its card before knew something it doesn't: knows) keeps the card before as a version; the card's version starts
    at the edit (its commit's time, never before the version it ends; cause edit). Duplicates get none.

    Args:
        db: The StatsDatabase after the edits (changed in place)
        before: {name: its card before the edits (a copy)}
        at: When the edit was made (the edits file's commit time, else now)
        commit: That commit's sha, if known

    Returns:
        The names given a version
    """
    made = []
    for name, old in before.items():
        card = db.stats.get(name)
        if card is None or getattr(card, "duplicate_of", None) or knows(card, old):
            continue
        prior = start(old)[0]
        when = at if prior is None or at >= prior else prior
        db.add_history(name, old)
        card.version = meta(None, when, "edit")
        card.version["from_commit"] = commit
        made.append(name)
    return made


def close_archived(old_stats: dict, names: list, index: GitIndex) -> None:
    """The last version of each card archived this build ends at its deleting commit (the tag's at and commit), its
    PNG pinned to that moment."""
    for name in names:
        versions = old_stats.get(name) or []
        last = versions[len(versions) - 1] if versions else None
        tag = getattr(last, "archived", None) if last is not None else None
        if not tag:
            continue
        if getattr(last, "version", None) is None:
            last.version = start_of(last, index, versions[len(versions) - 2] if len(versions) > 1 else None)
        close(last, index, tag.get("at"), index.by_sha.get(tag.get("commit") or ""))


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.card_history", description="Card history from the card repo's git log.")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("refine", help="the seed pass on copies of the pickles: prints what changes, writes the copies to --out")
    cmd.add_argument("--git", required=True, help="a full clone of the card repo (blobless is enough)")
    cmd.add_argument("--head", default="HEAD", help="the commit whose history counts (default HEAD)")
    cmd.add_argument("--pickles", required=True, help="a folder holding stats.pkl and old_stats.pkl (read only)")
    cmd.add_argument("--out", help="a folder for the refined stats.pkl and old_stats.pkl (never --pickles)")
    cmd.add_argument("--bot-git", default=environ.get("AUTOCROISSANT_BOT_CLONE") or BOT_CHECKOUT,
                     help="a full clone of AutoCroissant, whose PICKLE commits recover lost history (default: this checkout)")
    args = parser.parse_args(argv)
    stats, old = read_pickles(args.pickles)
    before = sum(len(v) for v in old.values())
    report = refine(stats, old, GitIndex.from_git(args.git, args.head), bot_pickles(args.bot_git))
    print(f"{len(stats)} cards, {before} versions before, {sum(len(v) for v in old.values())} after.")
    print("\n".join(report.lines()))
    if args.out:
        makedirs(args.out, exist_ok=True)
        if samefile(abspath(args.out), abspath(args.pickles)):
            print("--out must not be the --pickles folder.")
            return 1
        dump_pickle(old, join(args.out, "old_stats.pkl"))
        dump_pickle(stats, join(args.out, "stats.pkl"))
        print(f"Refined copies written to {args.out}.")
    return 0


# ========================
# Module Exports
# ========================
__all__ = [
    'BOT_CHECKOUT',
    'CAUSES',
    'DATA_FIELDS',
    'HIGH_STATS_LISTED',
    'RETIRED',
    'TYPE_RENAMES',
    'UNKNOWN_IF',
    'Commit',
    'GitIndex',
    'Reading',
    'Pickles',
    'Recovered',
    'Refined',
    'Snapshot',
    'allows_high',
    'bot_pickles',
    'bot_snapshots',
    'carry',
    'carry_to',
    'close',
    'close_archived',
    'collapse',
    'commits_of',
    'data_of',
    'date_archive',
    'date_cards',
    'date_versions',
    'dedupe',
    'high_values',
    'kept_values',
    'knows',
    'leaf',
    'main',
    'meta',
    'misreads',
    'normalize',
    'out_of_range',
    'record_edits',
    'record_versions',
    'recover',
    'recovered_before',
    'refine',
    'same_data',
    'start',
    'start_of',
    'unchecked',
    'unknown',
    'unknown_values',
]


if __name__ == "__main__":
    raise SystemExit(main())
