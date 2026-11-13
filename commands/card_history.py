"""
Card history from git, without Discord, network or psd-tools: one `git log --raw` of the card repo gives the newest
commit, first author and deletions per path, each blob's intro (the oldest commit that added it at any path) and
exact renames. On that index: the version rules (a version only when a card's data changes; start dates and a cause
from git; versions without a blob cause legacy) and the archive tags' dates. Design:
docs/Campaigns/DATA_BUILD_ACTION.md section 7.4.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from os import environ
from os.path import splitext
from subprocess import run
from typing import Optional

from global_config import MAILMAP_PATH
from commands.card_db import record
from commands.card_edits import as_parsed
from commands.card_models import RAID_STARS, RENAMED, CardInfo, card_name, picture_only, rename_trailer

# ========================
# Configuration
# ========================
# the fields a version is about (the path, blob, timestamp, parse and edit bookkeeping are not data)
DATA_FIELDS = ("card_type", "display_name", "ability", "stars", "subtype", "series", "subseries", "types", "stats",
               "creatures", "author", "note", "hidden", "duplicate_of")
# one header line per commit: sha, committer time, author (.mailmap'd), the bake's Renamed trailers (a renamed and edited file)
LOG_FORMAT = f"C%x09%H%x09%ct%x09%aN%x09%(trailers:key={RENAMED},valueonly,separator=%x1f)"
NO_BLOB = "0" * 40
# a parse problem -> the field that reading could not know (compared as unknown)
UNKNOWN_IF = (("NO STARS FOUND", "stars"), ("MISSPELT TYPE", "types"), ("NO ABILITY LAYER", "ability"),
              ("NO STATS FOUND", "stats"), ("OUT OF RANGE: stats", "stats"))
STAT_KINDS = ("creature", "minion", "creatures")       # psd_parser.STAT_KINDS: the kinds with stat bars
SERIES = ("series", "subseries")
RETIRED = ("MDW", "Markers")                          # no card folders (card_data.EXCLUDE_FOLDERS): a PSD there retired
STAT_MAX, STAT_CAP, STARS_MAX = 20, 99, 7             # a stat above 20 needs an allowance (allows_high); 3 digits: none
RANGE_PARSER = 2                                      # parser 2 (5835139): the validator's range check, allow_high_stats
STAT_KEYS = ("hp", "defense", "attack", "speed")


@dataclass
class Commit:
    """One commit of the index: its place in the log (0 = newest), sha, committer time and author name (as .mailmap maps it)."""
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
    renames: dict = field(default_factory=dict)       # old path -> (new path, Commit): the newest exact (or told) rename
    renamed_from: dict = field(default_factory=dict)  # new path -> old path of its newest exact (or told) rename into it
    by_sha: dict = field(default_factory=dict)        # commit sha -> Commit
    at_path: dict = field(default_factory=dict)       # path -> [(Commit, blob)] that put a blob there, newest first

    @classmethod
    def parse(cls, text: str) -> "GitIndex":
        """The index of `git log <head> --format=LOG_FORMAT --raw --no-abbrev --no-renames -z -- '*.psd' '*.png'`."""
        index = cls()
        commit, meta, gone, added, told = None, None, {}, {}, []

        def close():
            for blob, old in gone.items():
                if blob in added and old not in index.renames:
                    index.renames[old] = (added[blob], commit)
                    index.renamed_from.setdefault(added[blob], old)
            for old, new in told:                         # a Renamed trailer: the file renamed and edited in one commit
                if index.deleted.get(old) is commit and new in added.values() and old not in index.renames:
                    index.renames[old] = (new, commit)
                    index.renamed_from.setdefault(new, old)

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
                _, sha, at, author, trailers = (token.split("\t", 4) + ["", ""])[:5]
                commit, gone, added = Commit(len(index.commits), sha, float(at), author), {}, {}
                told = [pair for pair in map(rename_trailer, trailers.split("\x1f")) if pair]
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
            gone[src] = path
        elif dst != NO_BLOB:
            self.intro[dst] = (commit, path)
            self.intro_at[(dst, path)] = commit
            self.paths_of.setdefault(dst, {})[path] = commit
            self.at_path.setdefault(path, []).append((commit, dst))
            if status == "A":
                added[dst] = path

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
        done = run(["git", "-C", repo, *mailmap_args(repo, head), "log", head, f"--format={LOG_FORMAT}", "--raw", "--no-abbrev", "--no-renames", "-z",
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

    def origin(self, path: str, once: bool = False) -> str:
        """The path an exact rename (or a chain of them; once: the newest only) brought path from, else path."""
        seen = {path}
        while path in self.renamed_from and self.renamed_from[path] not in seen:
            path = self.renamed_from[path]
            seen.add(path)
            if once:
                break
        return path

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


def mailmap_args(repo: str, head: str = "HEAD") -> list:
    """
    git's `-c mailmap.blob=<blob>` for head's .mailmap ([] without one), so %aN maps in any clone: a non-bare one reads
    only its worktree's (a sparse checkout has none) and a blobless one may lack the blob (fetched here, the one lazy
    fetch).
    """
    if len(head) == 40 and not head.strip("0123456789abcdef"):     # a commit's .mailmap never changes
        return list(_mailmap_at(repo, head))
    return list(_mailmap_at.__wrapped__(repo, head))


@lru_cache(maxsize=64)
def _mailmap_at(repo: str, head: str) -> tuple:
    env = {**environ, "GIT_NO_LAZY_FETCH": "1"}
    found = run(["git", "-C", repo, "rev-parse", "--verify", "-q", f"{head}:{MAILMAP_PATH}"], capture_output=True, text=True,
                env=env)
    blob = found.stdout.strip()
    if found.returncode != 0 or not blob:
        return ()
    if run(["git", "-C", repo, "cat-file", "-e", blob], capture_output=True, env=env).returncode != 0:
        run(["git", "-C", repo, "cat-file", "blob", blob], capture_output=True, check=True)
    return ("-c", f"mailmap.blob={blob}")


# ========================
# Versions
# ========================
def data_of(card: CardInfo) -> dict:
    """A card's data fields as plain values (card_db's record), the ones a version is about; one form each for the
    same data: a title equal to the name is none, an empty ability none."""
    rec = record(card)
    data = {key: rec[key] for key in DATA_FIELDS}
    if data["display_name"] == rec["name"]:
        data["display_name"] = None
    data["ability"] = data["ability"] or None
    data["types"] = list(data["types"] or [])
    return data


def unknown(card: CardInfo) -> set:
    """Fields a reading could not know: the title before parser 2 (it read none) and its types (it missed and mistook
    icons), a field its parse failed on, stats it misread (misread_stats)."""
    found = set()
    if getattr(card, "parser_version", None) is None:
        found.add("types")
        if not getattr(card, "display_name", None):
            found.add("display_name")
    for problem in getattr(card, "problems", None) or []:
        found.update(key for prefix, key in UNKNOWN_IF if str(problem).startswith(prefix))
    if misread_stats(card):
        found.add("stats")
    return found


def misread_stats(card: CardInfo) -> bool:
    """Whether a reading from parser 2 on has a stat note of CardValidator's (never stored): a stat above STAT_MAX not
    allowed (STATS TOO HIGH), a bar of a kind with bars not read (HP or DEF; ATK or SPD unless active). Chris & Jordi's
    four bars all named Health read HP 26 = 7+6+7+6."""
    if unchecked(card):
        return False                                  # before the validator's range check: not judged
    if "stats" in out_of_range(card):
        return True
    if getattr(card, "card_type", None) not in STAT_KINDS:
        return False
    stats = getattr(card, "stats", None)
    need = STAT_KEYS[:2] if "active" in (getattr(card, "types", None) or []) else STAT_KEYS
    return any(getattr(stats, key, -1) == -1 for key in need)


def same_series(x: dict, y: dict) -> bool:
    """Whether two readings' series and subseries are the same, the 2026-10-20 split included (a series became the
    subseries of a parent: one side has no subseries and its series is the other's subseries). A subseries is one name
    across series since 2026-10-25, so the same subseries in another series is a change."""
    if (x["series"], x["subseries"]) == (y["series"], y["subseries"]):
        return True
    return any(not a["subseries"] and a["series"] and a["series"] == b["subseries"] for a, b in ((x, y), (y, x)))


def knows(a: CardInfo, b: CardInfo, ignore: tuple = ()) -> bool:
    """
    Whether reading a knew all that b did (data_of; not the fields in ignore): every field b could know (unknown), a
    knew too, with b's value; a series moved under a parent (same_series) is the same. So a reading that knew less
    (a parse that failed on a field) never stands for one that knew more. A value record() can't read counts as not
    known (a version is kept), so a build never fails on it.
    """
    try:
        x, y = data_of(a), data_of(b)
    except (TypeError, AttributeError) as error:
        print(f"History: {getattr(a, 'name', '?')} compared as changed ({error})")
        return False
    skip = unknown(b) | set(ignore)
    if not unknown(a) <= skip:
        return False
    if same_series(x, y):
        skip = skip | set(SERIES)
    return all(x[key] == y[key] for key in DATA_FIELDS if key not in skip)


def psd_changed(old: CardInfo, new: CardInfo) -> bool:
    """Whether a re-read PSD changed a card's data: its parse (as_parsed: Card Edits.json's edits undone) did not know
    all the one before did (knows), the author left out (git and Card Authors.txt give it, never the PSD)."""
    return not knows(as_parsed(new), as_parsed(old), ignore=("author",))


def start(card: CardInfo) -> tuple:
    """(when, commit) a version starts: its `version` record's from, else its timestamp."""
    info = getattr(card, "version", None)
    if info and info.get("from") is not None:
        return (info["from"], info.get("from_commit"))
    return (card.timestamp, None)


def carry(dropped: CardInfo, holder: CardInfo) -> None:
    """
    A dropped version's earlier start goes to the version that holds its data now (as_of keeps the first date); the
    cause stays the holder's own. A holder without a record gets one, cause legacy. The website dates a version by
    this start too (TTSCardMaker site/build/analytics.py when).
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
        holder.version = info = meta(None, at, "legacy")
    info["from"], info["from_commit"] = at, commit


def meta(start: Optional[Commit], start_at: float, cause: str) -> dict:
    """A version's `version` record: its start and cause."""
    return {"from": start.at if start else start_at, "from_commit": start.sha if start else None, "cause": cause}


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
        return meta(None, card.timestamp, "legacy")
    if previous is not None and getattr(previous, "blob_sha", None) == blob:
        here = index.intro_at.get((blob, card.path))
        if previous.path != card.path and here is not None:
            return meta(here, here.at, "moved")
        return meta(intro, intro.at, "reparse")
    return meta(intro, intro.at, "psd")


def allows_high(name: str, holder: Optional[CardInfo]) -> bool:
    """Whether a name's history may hold stats above STAT_MAX (up to STAT_CAP): its card has allow_high_stats or is a
    raid boss (RAID_STARS), or it has no current card to tell (archived)."""
    return holder is None or bool(getattr(holder, "allow_high_stats", False)) \
        or (getattr(holder, "stars", None) or 0) >= RAID_STARS


def out_of_range(card: CardInfo, allowed: bool = False) -> list:
    """The values no card prints, as fields: a stat above STAT_CAP (three digits), or above STAT_MAX without an
    allowance (the version's allow_high_stats or raid boss stars, or `allowed`: allows_high), stars outside 0..STARS_MAX."""
    found = []
    stats = getattr(card, "stats", None)
    high = allowed or getattr(card, "allow_high_stats", False) or (getattr(card, "stars", None) or 0) >= RAID_STARS
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


def kept_values(stats: dict, old_stats: dict) -> list:
    """History values out of range read by RANGE_PARSER or later (kept: an allow_high_stats entry allows a stat up to
    STAT_CAP), for the owner: the names, sorted."""
    return sorted({name for name, versions in old_stats.items() for v in versions
                   if not unchecked(v) and out_of_range(v, allows_high(name, stats.get(name)))})


# ========================
# Archive tags
# ========================
def date_archive(stats: dict, old_stats: dict, index: GitIndex) -> None:
    """Names gone from stats, their last version tagged archived (an untagged one changes nothing): every tag's at and
    commit from its path's deleting commit (an archived duplicate's from its PNG's, or its original's when the PNG
    stayed), renamed_to from an exact rename chain that ends at a listed card (else its PNG's: a rename that also
    edited the PSD), moved_to from one into a folder of no cards (MDW/The Morgue: retired, not deleted), either dated
    by the chain's step that reached it (Pandiety's move into MDW, not its rename to Pandeity 44 min before; the new
    name starts there too: intro_of). A deletion or rename older than the record's own reading, or an index that lacks
    the tag's commit (a clone not fetched since: a path deleted, added back and deleted again), changes nothing."""
    def copy_of(card) -> Optional[str]:
        return getattr(card, "duplicate_of", None) if picture_only(card) else None   # a variant, a marked PSD: its own
    for name, versions in sorted(old_stats.items(), key=lambda item: bool(copy_of(
            item[1][len(item[1]) - 1]) if item[1] else False)):     # originals before their copies
        if not versions or name in stats:
            continue
        last = versions[len(versions) - 1]
        original = copy_of(last)
        if not getattr(last, "archived", None):
            continue
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
        if original or stale or tag.get("renamed_to") or tag.get("moved_to"):
            continue
        if new:
            tag["renamed_to"] = new
        elif retired:
            tag["moved_to"] = retired


# ========================
# A build's history
# ========================
def record_versions(before: dict, archived_before: dict, db, reparsed: dict, restored: list, index: GitIndex,
                    edits: Optional[Commit] = None, authors: Optional[Commit] = None) -> list:
    """
    A build's versions, after its parse, edits, authors and archive: a card whose data changed has its card before
    appended to its history (at its own path); the new version never starts before the one it ends (a held PSD change
    or a re-read after an edit); an art-only change or a move keeps the version going; a
    re-read of the same PSD (same blob and path: a parser change) replaces its reading, its start kept; a re-read PSD
    whose own data did not change (psd_changed) is an edit or authors version, the edit landing in the same build; a
    card restored with the same data drops its archived record. A reading that knew less (unknown: a parse that failed
    on a field) keeps the card before as a version. Duplicates get none; a variant picture (its own data) does.

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
    others = set(db.stats) | set(db.old_stats)          # a rename's new name starts where its archive ends (intro_of)
    made = []
    for name, card in db.stats.items():
        if picture_only(card):                                # a variant, a marked PSD: its own versions
            continue
        ignore = ("duplicate_of",)                            # marked a duplicate: no version
        old = before.get(name)
        if name in restored:
            versions = db.old_stats.get(name) or []
            old = versions[len(versions) - 1] if versions else archived_before.get(name)
        if old is None:
            if getattr(card, "version", None) is None:
                card.version = start_of(card, index, None, others)
            continue
        if knows(card, old, ignore):                  # the new reading knew all the old one did: no version
            card.version = deepcopy(getattr(old, "version", None)) or start_of(card, index, None, others)
            if name in restored:                      # back with the same data: its archived record is no version
                versions = db.old_stats.get(name) or []
                if versions and versions[len(versions) - 1] is old and getattr(old, "archived", None):
                    versions.pop()
                    if not versions:
                        del db.old_stats[name]
            continue
        psd = name in reparsed and psd_changed(old, card)     # else an edit in the same build as an art change
        if psd and name not in restored and old.path == card.path and unknown(card) <= unknown(old) and \
                getattr(old, "blob_sha", None) and getattr(old, "blob_sha", None) == getattr(card, "blob_sha", None):
            # read again: the reading is replaced
            card.version = deepcopy(getattr(old, "version", None)) or start_of(card, index, None, others)
            made.append((name, "reparse"))
            continue
        if psd:
            start = start_of(card, index, old, others)
        else:
            only_author = {k for k, v in data_of(old).items() if v != data_of(card)[k]} == {"author"}
            source = authors if only_author else edits
            start = meta(source, source.at if source else card.timestamp, "authors" if only_author else "edit")
        prior = getattr(old, "version", None) or {}
        if prior.get("from") is not None and start["from"] is not None and start["from"] < prior["from"]:
            start["from"], start["from_commit"] = prior["from"], prior.get("from_commit")   # never before the one it ends
        card.version = start
        if name not in restored:
            previous = deepcopy(old)
            vars(previous).pop("created", None)               # the card's date: the card (or its archive) keeps it
            if getattr(previous, "version", None) is None:
                previous.version = start_of(previous, index, None, others)
            db.old_stats.setdefault(name, []).append(previous)
        made.append((name, start["cause"]))
    return made


def start_archived(old_stats: dict, names: list, index: GitIndex) -> None:
    """The last version of each card archived this build gets its start record if it has none."""
    for name in names:
        versions = old_stats.get(name) or []
        last = versions[len(versions) - 1] if versions else None
        tag = getattr(last, "archived", None) if last is not None else None
        if not tag:
            continue
        if getattr(last, "version", None) is None:
            last.version = start_of(last, index, versions[len(versions) - 2] if len(versions) > 1 else None)


# ========================
# Module Exports
# ========================
__all__ = [
    'DATA_FIELDS',
    'RETIRED',
    'UNKNOWN_IF',
    'Commit',
    'GitIndex',
    'allows_high',
    'carry',
    'carry_to',
    'data_of',
    'date_archive',
    'kept_values',
    'knows',
    'meta',
    'misread_stats',
    'out_of_range',
    'psd_changed',
    'record_versions',
    'same_series',
    'start',
    'start_archived',
    'start_of',
    'unchecked',
    'unknown',
]

