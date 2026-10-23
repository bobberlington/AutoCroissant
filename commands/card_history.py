"""
Card history from git, without Discord, network or psd-tools: one `git log --raw` of the card repo gives the newest
commit, first author, deletions and PNG commits per path, each blob's intro (the oldest commit that added it at any
path) and exact renames. On that index: the version rules (a version only when a card's data changes; from/to dates
and a cause from git; versions without a blob marked legacy), the seed pass over old_stats, and the archive tags'
dates. Design: docs/Campaigns/DATA_BUILD_ACTION.md section 7.4.

CLI (copies only): python -m commands.card_history refine --git <full clone> --pickles <dir> --out <dir>
"""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field
from os import environ, makedirs
from os.path import abspath, join, samefile, splitext
from subprocess import run
from typing import Optional

from commands.card_db import dump_pickle, read_pickles, record
from commands.card_models import CardInfo, card_name

# ========================
# Configuration
# ========================
# the fields a version is about (the path, blob, timestamp, parse and edit bookkeeping are not data)
DATA_FIELDS = ("card_type", "display_name", "ability", "stars", "subtype", "series", "subseries", "types", "stats",
               "creatures", "author", "note", "hidden", "duplicate_of")
CAUSES = ("psd", "reparse", "moved", "edit", "authors", "legacy")   # moved: the same blob in another folder (its data)
LOG_FORMAT = "C%x09%H%x09%ct%x09%an"                  # one header line per commit: sha, committer time, author
NO_BLOB = "0" * 40


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
    pngs: dict = field(default_factory=dict)          # PNG path -> Commits that changed it, newest first
    renames: dict = field(default_factory=dict)       # old path -> (new path, Commit): the newest exact rename
    by_sha: dict = field(default_factory=dict)        # commit sha -> Commit

    @classmethod
    def parse(cls, text: str) -> "GitIndex":
        """The index of `git log <head> --format=LOG_FORMAT --raw --no-abbrev --no-renames -z -- '*.psd' '*.png'`."""
        index = cls()
        commit, meta, gone, added = None, None, {}, {}

        def close():
            for blob, old in gone.items():
                if blob in added and old not in index.renames:
                    index.renames[old] = (added[blob], commit)

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
            gone[src] = path
        elif dst != NO_BLOB:
            self.intro[dst] = (commit, path)
            self.intro_at[(dst, path)] = commit
            if status == "A":
                added[dst] = path
        if path.lower().endswith(".png"):
            self.pngs.setdefault(path, []).append(commit)

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

    def png_commit(self, psd_path: str, before: Optional[Commit] = None, before_at: Optional[float] = None) -> Optional[str]:
        """The newest commit of the card's PNG older than `before` (else older than before_at; else the newest)."""
        for commit in self.pngs.get(splitext(psd_path)[0] + ".png", []):
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
    """A card's data fields as plain values (card_db's record), the ones a version is about."""
    rec = record(card)
    return {key: rec[key] for key in DATA_FIELDS}


def same_data(a: CardInfo, b: CardInfo) -> bool:
    """Whether two versions hold the same data; a value record() can't read counts as different (a version is kept,
    as before 2026-10-22), so the cleanup at each /update_stats never fails on it."""
    try:
        return data_of(a) == data_of(b)
    except (TypeError, AttributeError) as error:
        print(f"History: {getattr(a, 'name', '?')} compared as changed ({error})")
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
    now = start(holder)[0]
    if at is None or (now is not None and at >= now):
        return
    info = getattr(holder, "version", None)
    if info is None:
        holder.version = info = meta(None, at, "legacy", legacy=not getattr(holder, "blob_sha", None))
    info["from"], info["from_commit"] = at, commit


def collapse(versions: list, holder: Optional[CardInfo]) -> tuple:
    """
    One name's history without duplicates (design 7.4): of each run of consecutive versions with the same data
    (same_data: the blob, path, dates and parse bookkeeping are no data) the first stays; a run with the data of the
    card that follows it (the current card, or the archived record ending the list) goes into that card. Either way
    the run's earliest start stays (carry). A version tagged `archived` is never dropped and ends its run (the
    versions after it are the card's next life). Idempotent.

    Args:
        versions: The name's old_stats versions, oldest first (not changed)
        holder: The current card, or None

    Returns:
        (kept versions, dropped versions), each oldest first
    """
    runs: list = []
    for version in versions:
        tail = runs[len(runs) - 1] if runs else None
        if tail and not getattr(tail[len(tail) - 1], "archived", None) and same_data(tail[0], version):
            tail.append(version)
        else:
            runs.append([version])
    kept, dropped = [], []
    for i, run in enumerate(runs):
        last = run[len(run) - 1]
        if i + 1 == len(runs) and holder is not None and same_data(run[0], holder):
            into = holder
        elif getattr(last, "archived", None):
            into = last
        else:
            into = run[0]
        for version in run:
            if version is into or getattr(version, "archived", None):
                kept.append(version)
            else:
                carry(version, into)
                dropped.append(version)
    return (kept, dropped)


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


def start_of(card: CardInfo, index: GitIndex, previous: Optional[CardInfo]) -> dict:
    """
    A version's start from git: its blob's intro (psd); the previous version's blob again: reparse from the same intro,
    or moved from the commit that put the blob at this path (a folder's data); no blob in the index: legacy, at the
    stored timestamp.
    """
    blob = getattr(card, "blob_sha", None)
    intro = index.blob_intro(blob)
    if intro is None:
        return meta(None, card.timestamp, "legacy", legacy=True)
    if previous is not None and getattr(previous, "blob_sha", None) == blob:
        here = index.intro_at.get((blob, card.path))
        if previous.path != card.path and here is not None:
            return meta(here, here.at, "moved")
        return meta(intro, intro.at, "reparse")
    return meta(intro, intro.at, "psd")


# ========================
# The seed pass
# ========================
@dataclass
class Refined:
    """What refine changed (each count once)."""
    dropped: list = field(default_factory=list)       # (name, version timestamp) of each duplicate version dropped
    versions_dated: int = 0                           # versions with a blob whose timestamp moved to the blob's intro
    versions_with_blob: int = 0
    legacy: int = 0                                   # versions without a blob, marked legacy (before the drop)
    legacy_kept: int = 0                              # of those, still versions after it
    cards_dated: list = field(default_factory=list)   # current cards (no duplicates) re-dated
    duplicates_dated: int = 0                         # duplicates that followed their original
    cards: int = 0                                    # current cards that are no duplicate
    tagged: list = field(default_factory=list)        # archive tags added (backfill)
    tags_dated: list = field(default_factory=list)    # archive tags given git's date and commit
    renamed: list = field(default_factory=list)       # (old, new) exact renames recorded on tags

    def changed(self) -> bool:
        return bool(self.dropped or self.versions_dated or self.cards_dated or self.duplicates_dated or self.tagged
                    or self.tags_dated or self.renamed)

    def lines(self) -> list:
        def names(items):
            return f" ({', '.join(items)})" if items else ""
        return [f"Duplicate versions dropped: {len(self.dropped)}{names([n for n, _ in self.dropped])}",
                f"Versions re-dated: {self.versions_dated} of {self.versions_with_blob} with a blob",
                f"Versions marked legacy (no blob): {self.legacy}, {self.legacy_kept} after the drop",
                f"Cards re-dated: {len(self.cards_dated)} of {self.cards} (and {self.duplicates_dated} duplicates follow)",
                f"Archive tags added: {len(self.tagged)}{names(self.tagged)}",
                f"Archive tags dated from git: {len(self.tags_dated)}{names(self.tags_dated)}",
                f"Renames recorded: {len(self.renamed)}{names([f'{a} -> {b}' for a, b in self.renamed])}"]


def date_cards(stats: dict, index: GitIndex, report: Refined) -> None:
    """Each current card's timestamp becomes its blob's intro; a duplicate takes its original's."""
    for name, card in stats.items():
        if getattr(card, "duplicate_of", None):
            continue
        report.cards += 1
        intro = index.blob_intro(getattr(card, "blob_sha", None))
        if intro is not None and intro.at != card.timestamp:
            card.timestamp = intro.at
            report.cards_dated.append(name)
    for card in stats.values():
        original = stats.get(getattr(card, "duplicate_of", None) or "")
        if original is not None and card.timestamp != original.timestamp:
            card.timestamp = original.timestamp
            report.duplicates_dated += 1


def date_archive(stats: dict, old_stats: dict, index: GitIndex, report: Refined) -> None:
    """Names gone from stats: the last version tagged archived (backfill) if untagged; every tag's at and commit from its
    path's deleting commit, renamed_to from an exact rename whose new card is listed."""
    for name, versions in old_stats.items():
        if not versions or name in stats:
            continue
        last = versions[len(versions) - 1]
        if getattr(last, "duplicate_of", None):
            continue
        if not getattr(last, "archived", None):
            last.archived = {"at": None, "path": last.path, "blob": getattr(last, "blob_sha", None) or None,
                             "commit": None, "backfill": True}
            report.tagged.append(name)
        tag = last.archived
        gone = index.deleted.get(tag.get("path") or "")
        if gone is not None and (tag.get("at"), tag.get("commit")) != (gone.at, gone.sha):
            tag["at"], tag["commit"] = gone.at, gone.sha
            report.tags_dated.append(name)
        moved = index.renames.get(tag.get("path") or "")
        if moved and not tag.get("renamed_to") and card_name(moved[0]) in stats:
            tag["renamed_to"] = card_name(moved[0])
            report.renamed.append((name, tag["renamed_to"]))


def date_versions(name: str, stats: dict, old_stats: dict, index: GitIndex, report: Refined) -> None:
    """
    One name's versions: each one's `version` (kept when the seed or a build decided it; dated again from git when
    /update_stats made it) and, with a blob, its timestamp from the blob's intro; duplicates dropped (collapse), their
    starts carried; then each end set to the next start, the deleting commit, or open for the current card's.
    """
    versions = old_stats[name]
    current = stats.get(name)
    previous = None
    for version in versions:
        version.version = _dated(version, index, previous)
        info = version.version
        if info["legacy"] and not getattr(version, "blob_sha", None):
            report.legacy += 1
        elif getattr(version, "blob_sha", None):
            report.versions_with_blob += 1
            intro = index.blob_intro(version.blob_sha)
            if intro is not None and version.timestamp != intro.at:
                version.timestamp = intro.at
                report.versions_dated += 1
        previous = version
    if current is not None:
        current.version = _dated(current, index, versions[len(versions) - 1] if versions else None)

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


def _dated(card: CardInfo, index: GitIndex, previous: Optional[CardInfo]) -> dict:
    """card's `version` for the seed: its own when the seed or a build decided it; else start_of(), keeping an earlier
    start that /update_stats carried (its records are `legacy`, made without git)."""
    info = getattr(card, "version", None)
    if info is not None and info.get("cause") != "legacy":
        return info
    fresh = start_of(card, index, previous)
    if info and info.get("from") is not None and (fresh["from"] is None or info["from"] < fresh["from"]):
        fresh["from"], fresh["from_commit"] = info["from"], info.get("from_commit")
    return fresh


def refine(stats: dict, old_stats: dict, index: GitIndex) -> Refined:
    """
    The seed pass, in place: cards and versions dated from git, duplicate versions dropped (collapse; never an
    archived one), blob-less versions marked legacy, archive tags added and dated, every version and current card given
    its `version` record. Idempotent: a second pass changes nothing.

    Args:
        stats: {name: CardInfo} (stats.pkl)
        old_stats: {name: [CardInfo]} (old_stats.pkl), oldest first
        index: The card repo's GitIndex

    Returns:
        The counts
    """
    report = Refined()
    date_cards(stats, index, report)
    date_archive(stats, old_stats, index, report)
    names = list(old_stats)
    for name in names:
        date_versions(name, stats, old_stats, index, report)
    names = set(names)
    for name, card in stats.items():
        if not getattr(card, "duplicate_of", None) and name not in names:
            card.version = _dated(card, index, None)
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
    card restored with the same data drops its archived record. Duplicates get none.

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
                card.version = start_of(card, index, None)
            continue
        if same_data(old, card):
            info = deepcopy(getattr(old, "version", None)) or start_of(card, index, None)
            info["to"], info["to_commit"], info["png_commit"] = None, None, None
            card.version = info
            if name in restored:                      # back with the same data: its archived record is no version
                versions = db.old_stats.get(name) or []
                if versions and versions[len(versions) - 1] is old and getattr(old, "archived", None):
                    versions.pop()
                    if not versions:
                        del db.old_stats[name]
            continue
        if name in reparsed:
            start = start_of(card, index, old)
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
            if getattr(previous, "version", None) is None:
                previous.version = start_of(previous, index, None)
            close(previous, index, start["from"], end_of(start))
            db.old_stats.setdefault(name, []).append(previous)
        made.append((name, start["cause"]))
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
    args = parser.parse_args(argv)
    stats, old = read_pickles(args.pickles)
    before = sum(len(v) for v in old.values())
    report = refine(stats, old, GitIndex.from_git(args.git, args.head))
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
    'CAUSES',
    'DATA_FIELDS',
    'Commit',
    'GitIndex',
    'Refined',
    'carry',
    'close',
    'close_archived',
    'collapse',
    'data_of',
    'date_archive',
    'date_cards',
    'date_versions',
    'dedupe',
    'main',
    'meta',
    'record_versions',
    'refine',
    'same_data',
    'start',
    'start_of',
]


if __name__ == "__main__":
    raise SystemExit(main())
