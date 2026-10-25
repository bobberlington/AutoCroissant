"""
The push guards (docs/Campaigns/PUSH_GUARDS.md): TTSCardMaker's card-guards.yml runs this on every push to main. Guard 1
(png_export) gives a card PSD pushed without its PNG the PNG the bake would make (the fork's render of the changed layers,
merged into the old PNG; Photoshop's stored composite only cross-checks, and a PSD without one gets the PNG under the
bake's own gates, told "no stored image to cross-check"); guard 2 (card_standards) checks the pushed PSDs against the card
standards (missing fonts are font-check.yml's alone) and fixes only the 0-pixel classes. One report per push: the job
summary and commit comment (report.md), one Discord post for the standards channel (discord.json: no mentions, no
previews), and a plan of the writes (plan.json) that `land` commits once, after the 30-minute quiet window. Each card
pushed over an older PSD carries what changed inside it (psd_diff: "added Ability FX: Blazing", "Name text: 'a' -> 'b'").
Every card PNG the push brought is recompressed losslessly in that commit (owner, 2026-10-23; png_export.recompress,
the same pixels) when that saves more than 10% and 64 KiB (our thresholds), and so is every PNG the bot writes.

    python -m commands.card_guards check --clone <dir> --after <sha> [--before <sha>] [--repo <owner/name>] [--author <login>]
                                         [--cards <names>] [--whole-tree] [--mode dry|report|on] --out <dir>
    python -m commands.card_guards land --plan <out>/plan.json [--quiet-from <ISO time>] [--dry-run] [--local <clone>] [--rerun]

check reads trees from a local clone (the Action's blobless checkout with its history; a full clone locally) and blobs
from it or, when it lacks them, by sha through the API (verified). It writes nothing to the card repo, ever: a dry run
and a replay are the same command. land re-reads the branch head, drops a card whose PSD or PNG moved or that a later
push touched (card_bake.recent_paths), and makes one commit (git_commit.commit_files; --local: in a clone, nothing
pushed). The guards' own commits and the bake's, the save's and Card sync's are skipped (the AutoCroissant author or
github-actions; a path a person also changed in the range is still guarded). No Discord or config imports (the token:
config through github_api, else GITHUB_TOKEN); SPARSE_FILES is what card-guards.yml checks out: the import closure and
requirements-cards.txt.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from json import dump, dumps, load, loads
from os import cpu_count, environ, makedirs
from os.path import basename, dirname, join, splitext
from re import fullmatch, sub
from subprocess import CalledProcessError, TimeoutExpired
from sys import stderr
from time import sleep as _sleep
from traceback import format_exc
from typing import Callable, Optional

import commands.card_standards as standards
import commands.png_export as png_export
import commands.psd_diff as psd_diff
import commands.psd_edit as psd_edit
from commands.card_bake import RECENT, Snapshot, fetch_verified, local_snapshot, recent_paths, snapshot_at
from commands.card_check import vocabulary
from commands.card_edits import CardAuthors
from commands.font_check import is_card_psd
from commands.git_commit import (AUTHOR, COMMIT_TRIES, NO_LAZY, Transient, api, commit_files, git, local_commit, plural,
                                 ref_head, token)
from commands.github_api import blob_sha, token_from_env
from commands.psd_edit import Refused
from global_config import CARD_REPO, ICON_MANIFEST_PATH

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_guards.py", "commands/png_export.py", "commands/card_standards.py", "commands/txt2_repair.py",
                "commands/psd_diff.py", "commands/card_bake.py", "commands/card_check.py", "commands/psd_edit.py", "commands/psd_regions.py",
                "commands/psd_parser.py", "commands/card_models.py", "commands/card_edits.py", "commands/font_check.py",
                "commands/github_api.py", "commands/git_commit.py", "commands/repo_history.py", "global_config.py",
                "requirements-cards.txt")
SUBJECT = "Card guards"                         # the bot commit: "Card guards: 2 PNGs re-exported, 1 PSD fixed (push 7cb4c867)"
TRAILER = "Guarded-push"                        # the push the commit guards
RECOMPRESSED = "Recompressed"                   # a trailer per PNG only recompressed: no image version (build_site)
BOT_EMAILS = ("41898282+github-actions[bot]@users.noreply.github.com",)   # besides git_commit.AUTHOR: never guarded
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZEROS = "0" * 40
MODES = ("dry", "report", "on")
LINE = "{login} pushed {card}{changes} with these issues: {issues}. [{short}]({url})"
CHANGES_DISCORD = 120                           # characters of a card's changes in its Discord line (psd_diff)
CHANGES_COMMENT = 600                           # and in the commit comment
MARKDOWN = r"([\\`*_~|<\[\]])"                   # escaped in card text and layer names (GitHub and Discord markdown)
DISCORD_LIMIT = 2000
SUPPRESS_EMBEDS = 4                             # Discord message flag: no link previews
URL_SHA = 12                                    # hex digits of a commit link (GitHub resolves it; 28 characters a line saved)
NOBODY = "Someone"
NOREPLY = r"(?:\d+\+)?([A-Za-z0-9-]+)@users\.noreply\.github\.com"   # a GitHub noreply email names its login
QUIET_MARGIN = timedelta(seconds=60)            # land waits RECENT plus this after the push
ANCHOR_WINDOW = timedelta(hours=2)              # a burst: PSD commits this close to the push may stand in for its PNG's PSD
POOL_MIN = 8                                    # --whole-tree checks this many cards or more in a process pool
CARD_ROOTS = ("Auxiliary/", "Creatures/", "Field/", "Items/", "N.M.E/")   # where card PNGs live (Types/ icons are not cards)
TELL_PNG_MISMATCH = False                       # R10 and "current but outside the bar" on Discord too (beyond the owner's
                                                # two rules: the commit comment only until the owner asks)
_worker: dict = {}


# ========================
# Data
# ========================
@dataclass
class Change:
    """One path the push changed: git diff --raw."""
    status: str                                 # A, M or D
    old: str                                    # blob before ("" when added)
    new: str                                    # blob after ("" when deleted)
    path: str


@dataclass
class Writer:
    """The commit that last changed a path in the push."""
    commit: str
    name: str
    email: str
    subject: str
    login: str = ""

    @property
    def bot(self) -> bool:
        """By author only: a person's commit named like a bot's ("Card edits: ...") is still guarded."""
        return self.email in (AUTHOR["email"], *BOT_EMAILS) or self.name == AUTHOR["name"]


@dataclass
class Job:
    """One card's work in the push."""
    psd: str                                    # the PSD path at after (the deleted one's for "deleted")
    case: str                                   # stale, pair, new, beside, rename, psd-only, png-only, deleted, png-deleted, whole,
                                                # numbered (numbered PNGs pushed alone)
    psd_old: str = ""                           # its blob before the push
    psd_new: str = ""                           # its blob at after
    png: str = ""                               # the PNG path (to check, or to write)
    png_old: str = ""                           # the PNG blob an export rests on
    png_new: str = ""                           # the pushed PNG (R10)
    numbered: dict = field(default_factory=dict)   # numbered PNG path -> blob at after
    copy_from: str = ""                         # rename: the old PNG's blob, copied to png
    psd_anchor: str = ""                        # stale: the PSD as of its PNG's last change, when a burst's earlier push
    anchor_commit: str = ""                     # changed the PSD without the PNG (G5's second try); that commit
    left: str = ""                              # rename: the old PNG, still at its old path
    pushed: dict = field(default_factory=dict)  # numbered PNG path -> blob the push brought (recompressed only)
    login: str = NOBODY
    commit: str = ""                            # the last writer's commit (the Discord link)


@dataclass
class Result:
    """One card's verdicts."""
    job: Job
    export: dict = field(default_factory=dict)  # guard 1: outcome, why, numbers...
    issues: list = field(default_factory=list)  # card_standards.Issue as dicts
    fixed: list = field(default_factory=list)   # the fix lines
    fix_refused: list = field(default_factory=list)   # (code, why)
    r10: dict = field(default_factory=dict)     # png_check of a pushed PNG
    notes: list = field(default_factory=list)   # (text, told on Discord)
    writes: list = field(default_factory=list)  # {"path", "file", "sha", "kind" (png, psd, recompress), "line"}
    changes: list = field(default_factory=list) # what changed inside a modified PSD (psd_diff phrases)
    error: str = ""

    @property
    def name(self) -> str:
        return card_name(self.job.psd)

    def told(self, mode: str = "on") -> list:
        """The lines the author is told on Discord; in report mode the PNGs the bot would make and the fixable issues
        too (nothing is made or fixed there)."""
        out = []
        if self.export.get("outcome") == "refused":
            out.append(self.export["why"])
        if self.export.get("unchecked"):        # G2 rendered anyway (owner, 2026-10-24): told instead of refused
            out.append(self.export["unchecked"])
        out += [why for why, _ in self.export.get("numbered_refused", [])]
        if mode == "report":
            out += [would_make(self.job.case, self.job.png, w) for w in self.writes if w["kind"] == "png"]
        out += [f"{line} (the bot could not fix it: {why})" for line, why in self.fix_refused]
        if self.r10.get("ok") is False and TELL_PNG_MISMATCH:
            out.append(self.r10["line"])
        out += [issue["line"] for issue in self.issues
                if issue["discord"] and (issue["told"] or (mode == "report" and issue["fix"] and not issue["waived"]))]
        out += [text for text, discord in self.notes if discord]
        if self.error:
            out.append(self.error_line())
        return out

    def error_line(self) -> str:
        """An error, told (the owner's "no PNG and a report instead"): the PNG part when the card needed one and got none."""
        needs = (self.job.case in ("stale", "beside", "new", "rename") and self.export.get("outcome") not in ("current", "refused")
                 and not any(w["kind"] == "png" for w in self.writes))
        kind = self.error.split(":", 1)[0]
        return f"the bot could not finish checking it ({kind})" + ("; PNG not made: export it in Photoshop" if needs else "")


# ========================
# Helper Functions
# ========================
def card_name(path: str) -> str:
    """The card's name from its PSD path (card_models.card_name)."""
    return splitext(basename(path))[0].replace("_", " ")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def would_make(case: str, png: str, write: dict) -> str:
    """Report mode's line for a PNG write the bot would make with CARD_GUARDS=on (nothing is committed there)."""
    if write["path"] != png:
        return f"{basename(write['path'])} not updated with the PSD: re-export it in Photoshop"
    if case == "new":
        return "no PNG pushed with the new PSD: export it in Photoshop"
    if case == "rename":
        return "PNG not renamed with the PSD: rename it too"
    return "PNG not exported with the PSD: export it in Photoshop"


def _key(path: str) -> tuple:
    return dirname(path), splitext(basename(path))[0].lower()


def _is_png(path: str) -> bool:
    return path.lower().endswith(".png")


def card_png(path: str) -> bool:
    """A PNG in the card folders (not Types/, site/, TTS/; MDW/ and Markers/ hold no card)."""
    return _is_png(path) and path.startswith(CARD_ROOTS) and is_card_psd(path[:len(path) - 4] + ".psd")


# ========================
# The card repo: a local clone, blobs by sha (local first, else the API)
# ========================
class Source:
    """Trees and history from a local clone (never a lazy fetch), blobs from it or the API."""

    def __init__(self, clone: str, repository: str = ""):
        self.clone, self.repository = clone, repository
        self.logins: dict = {}

    def git(self, *args: str) -> str:
        try:
            return git(self.clone, *args, env={**environ, **NO_LAZY}).decode("utf-8", "replace")
        except CalledProcessError as error:
            raise Transient(f"git {args[0]} in {self.clone} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
        except (TimeoutExpired, OSError) as error:
            raise Transient(f"git in {self.clone} failed: {error}") from error

    def commit(self, ref: str) -> str:
        """The commit a ref names; "" when the clone lacks it."""
        try:
            return self.git("rev-parse", "-q", "--verify", f"{ref}^{{commit}}").strip()
        except Transient:
            return ""

    def parent(self, sha: str) -> str:
        return self.commit(f"{sha}^1")

    def is_ancestor(self, a: str, b: str) -> bool:
        try:
            self.git("merge-base", "--is-ancestor", a, b)
            return True
        except Transient:
            return False

    def diff(self, before: str, after: str) -> list:
        """[Change] between two commits (or the empty tree), renames off: trees only, no blob read."""
        raw = self.git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", before, after)
        parts, out, i = raw.split("\0"), [], 0
        while i < len(parts) - 1:
            meta = parts[i]
            if not meta.startswith(":"):
                i += 1
                continue
            fields = meta[1:].split(" ")
            status = fields[4][0]
            out.append(Change(status, "" if set(fields[2]) == {"0"} else fields[2],
                              "" if set(fields[3]) == {"0"} else fields[3], parts[i + 1]))
            i += 2
        return out

    def tree(self, commit: str) -> dict:
        """{path: (mode, blob sha)} of a commit's tree ({} for the empty tree)."""
        if commit == EMPTY_TREE:
            return {}
        out = {}
        for line in self.git("ls-tree", "-r", "-z", commit).split("\0"):
            meta, _, path = line.partition("\t")
            if path and meta.split(" ")[1] == "blob":
                out[path] = (meta.split(" ")[0], meta.split(" ")[2])
        return out

    def writers(self, before: str, after: str) -> dict:
        """{path: Writer}: the last person's commit in before..after that changed it, else the last bot commit (a path is
        left out as the bot's only when no person changed it in the range)."""
        span = after if before == EMPTY_TREE else f"{before}..{after}"
        raw = self.git("log", "--no-renames", "--reverse", "--format=%x1e%H%x1f%an%x1f%ae%x1f%s", "--name-only", "-z", span)
        out, people = {}, {}
        for chunk in raw.split("\x1e"):
            if not chunk.strip():
                continue
            head, _, rest = chunk.partition("\0")
            sha, name, email, subject = (head.split("\x1f") + ["", "", ""])[:4]
            writer = Writer(sha, name, email, subject)
            for path in rest.lstrip("\n").split("\0"):
                if path.strip():
                    out[path.strip("\n")] = writer
                    if not writer.bot:
                        people[path.strip("\n")] = writer
        return {**out, **people}

    def anchor(self, png: str, psd: str, before: str, after: str, psd_old: str, window: timedelta = ANCHOR_WINDOW) -> tuple:
        """
        (blob, commit): the PSD as of the commit that last changed its PNG (up to before), when the PSD changed since only
        in commits within window of after (a burst: an earlier push's PNG was not made, its run superseded); else ("", "").
        """
        if before == EMPTY_TREE:
            return "", ""
        try:
            last = self.git("log", "-1", "--format=%H", before, "--", png).strip()
            if not last:
                return "", ""
            blob = self.git("rev-parse", f"{last}:{psd}").strip()
            since = [int(t) for t in self.git("log", "--format=%ct", f"{last}..{before}", "--", psd).split()]
            now = int(self.git("log", "-1", "--format=%ct", after).strip())
        except (Transient, ValueError):         # the PSD did not exist then, or no history: no anchor
            return "", ""
        if not blob or blob == psd_old or not since or any(now - t > window.total_seconds() for t in since):
            return "", ""
        return blob, last

    def blob(self, sha: str) -> bytes:
        """A blob by sha, verified; raises Refused("blob")."""
        return fetch_verified(self.repository, sha, "", self.clone)

    def login(self, writer: Writer, fallback: str) -> str:
        """The commit author's GitHub login: the commits API (once per commit, when a token and repo are given), else the
        login in a GitHub noreply email; with the API, an author without a GitHub account is named by the pusher's login
        (fallback, github.actor: who pushed the card); without it (a local run) or without a pusher (a dispatch: whoever
        ran it did not push the card), by the git author name."""
        if writer.commit not in self.logins:
            login, online = "", bool(self.repository and token())
            if online:
                try:
                    login = ((api(self.repository, "GET", f"commits/{writer.commit}").json().get("author") or {}).get("login") or "")
                except Exception:               # no login: the fallbacks stand
                    login = ""
            noreply = fullmatch(NOREPLY, writer.email or "")
            pusher = fallback if online and fallback != NOBODY else ""
            self.logins[writer.commit] = login or (noreply.group(1) if noreply else "") or pusher or writer.name or fallback
        return self.logins[writer.commit]


# ========================
# The push range and the jobs (section 4)
# ========================
def push_range(source: Source, before: str, after: str) -> tuple:
    """
    (before, after, notes): before resolved; an empty or all-zero before, one the clone lacks, or one that is not an
    ancestor of after (a force push) is replaced by after's first parent (the empty tree for a root commit).
    """
    notes = []
    head = source.commit(after)
    if not head:
        raise Transient(f"the clone has no commit {after[:8]}")
    base = source.commit(before) if before and before != ZEROS else ""
    if base and not source.is_ancestor(base, head):
        notes.append(f"`{before[:8]}` is not an ancestor of `{head[:8]}` (a force push?): checked against its first parent")
        base = ""
    elif before and before != ZEROS and not base:
        notes.append(f"the clone lacks `{before[:8]}`: checked against the first parent")
    elif not before or before == ZEROS:
        if before == ZEROS:
            notes.append("a new branch: checked against the first parent")
    if not base:
        base = source.parent(head) or EMPTY_TREE
    return base, head, notes


def plan_jobs(source: Source, before: str, after: str, author: str, cards: Optional[set] = None) -> tuple:
    """
    The push's jobs per section 4.2.

    Returns:
        ([Job], {path: Writer} left out as bot commits, [notes], after tree)
    """
    changes = source.diff(before, after)
    writers = source.writers(before, after)
    head_writer = Writer(after, "", "", "")
    tree = source.tree(after)
    old_tree = source.tree(before)
    snapshot = Snapshot(source.repository, after, {}, tree)
    old_snapshot = Snapshot(source.repository, before, {}, old_tree)
    skipped = {c.path: writers[c.path] for c in changes if c.path in writers and writers[c.path].bot}
    changes = [c for c in changes if c.path not in skipped]
    psds = [c for c in changes if is_card_psd(c.path)]
    pngs = {c.path: c for c in changes if card_png(c.path)}
    deleted: dict = {}                          # blob -> deleted paths: an added path with one of them is a rename
    for c in psds:
        if c.status == "D":
            deleted.setdefault(c.old, []).append(c.path)
    renamed_new = {c.path: deleted[c.new].pop(0) for c in psds if c.status == "A" and deleted.get(c.new)}
    renamed_old = set(renamed_new.values())
    png_deleted = {_key(p) for p, c in pngs.items() if c.status == "D"}
    notes, jobs = [], []

    def who(path: str) -> tuple:
        writer = writers.get(path, head_writer)
        return source.login(writer, author) if writer.commit != after or writer.name else author, writer.commit or after

    def numbered(path: str) -> dict:
        return {p: tree[p][1] for p in snapshot.png_for(path)[1] if p not in pngs and p in tree}

    def pushed(path: str) -> dict:
        return {p: pngs[p].new for p in snapshot.png_for(path)[1] if p in pngs and pngs[p].status in "AM"}

    for change in psds:
        if change.path in renamed_old:
            continue
        png = snapshot.png_for(change.path)[0] if change.status != "D" else ""
        login, commit = who(change.path)
        job = Job(change.path, "", change.old, change.new, login=login, commit=commit, pushed=pushed(change.path))
        if change.status == "D":
            left = snapshot.png_for(change.path)[0]
            if left:
                jobs.append(Job(change.path, "deleted", change.old, "", left, tree[left][1], login=login, commit=commit))
            continue
        if change.path in renamed_new:
            old_path = renamed_new[change.path]
            if not png:
                old_png = old_snapshot.png_for(old_path)[0]
                if old_png:
                    job.case, job.png = "rename", splitext(change.path)[0] + splitext(old_png)[1]
                    job.copy_from, job.psd_old = old_tree[old_png][1], change.new
                    job.left = old_png if old_png in tree else ""
                    jobs.append(job)
            continue
        if png and png in pngs and pngs[png].status in "AM":
            job.case, job.png, job.png_new = "pair", png, pngs[png].new
        elif png and change.status == "M":
            job.case, job.png, job.png_old, job.numbered = "stale", png, tree[png][1], numbered(change.path)
            job.psd_anchor, job.anchor_commit = source.anchor(png, change.path, before, after, change.old)
        elif png:
            job.case, job.png, job.png_old, job.numbered = "beside", png, tree[png][1], numbered(change.path)
        elif _key(change.path) in png_deleted:      # the author deleted the PNG: the standards only
            job.case = "psd-only"
            notes.append(f"`{change.path}` was pushed with its PNG deleted: no PNG is made")
        else:
            job.case, job.png = "new", splitext(change.path)[0] + ".png"
        jobs.append(job)
    pushed_psds = {c.path for c in psds}
    psd_by_key: dict = {}                       # stems matched in any case (L12), so one key can hold several PSDs
    for p in tree:
        if is_card_psd(p):
            psd_by_key.setdefault(_key(p), []).append(p)
    for path, change in sorted(pngs.items()):
        found = psd_by_key.get(_key(path), [])
        exact = [p for p in found if splitext(p)[0] == splitext(path)[0]]
        if len(exact or found) > 1:
            notes.append(f"`{path}` matches {len(found)} PSDs that differ only in letter case: not checked")
            continue
        psd = (exact or found or [None])[0]
        if psd is None or psd in pushed_psds:
            if psd is None and change.status in "AM" and not snapshot.duplicate_of(CardAuthors.key(card_name(path))):
                if not any(p for p in pushed_psds if _key(p) == _key(path)):
                    notes.append(f"`{path}` has no PSD beside it")
            continue
        login, commit = who(path)
        if change.status in "AM":
            jobs.append(Job(psd, "png-only", "", tree[psd][1], path, change.old, change.new, login=login, commit=commit,
                            pushed=pushed(psd)))
        elif change.status == "D":
            jobs.append(Job(psd, "png-deleted", "", tree[psd][1], path, change.old, login=login, commit=commit,
                            pushed=pushed(psd)))  # its numbered PNGs too: one entry per PSD
    covered = {path for job in jobs for path in job.pushed}
    alone: dict = {}                            # numbered PNGs pushed without their PSD or its PNG: recompressed only
    for path, change in sorted(pngs.items()):
        psd = snapshot.duplicate_of(CardAuthors.key(card_name(path))) if change.status in "AM" and path not in covered else ""
        if psd in tree and path in snapshot.png_for(psd)[1]:
            alone.setdefault(psd, {})[path] = change.new
    for psd, found in alone.items():
        login, commit = who(sorted(found)[0])
        jobs.append(Job(psd, "numbered", "", tree[psd][1], login=login, commit=commit, pushed=found))
    if cards:
        jobs = [job for job in jobs if CardAuthors.key(card_name(job.psd)) in cards]
    return sorted(jobs, key=lambda job: job.psd), skipped, notes, tree


def whole_tree_jobs(tree: dict, author: str, cards: Optional[set] = None) -> list:
    """Every card PSD of the tree: guard 2 and R10 against its PNG."""
    snapshot = Snapshot("", "", {}, tree)
    jobs = []
    for path in sorted(tree):
        if is_card_psd(path) and (not cards or CardAuthors.key(card_name(path)) in cards):
            png = snapshot.png_for(path)[0]
            jobs.append(Job(path, "whole", "", tree[path][1], png, "", tree[png][1] if png else "", login=author))
    return jobs


# ========================
# Context: Types/, the status chart
# ========================
def build_context(source: Source, tree: dict, errors: Optional[list] = None) -> tuple:
    """
    (card_standards.Context, Vocabulary) from the tree at after.

    The manifest and the status chart come by blob; a failed one is an error line and the rules run without it.
    """
    errors = errors if errors is not None else []
    manifest = None
    if ICON_MANIFEST_PATH in tree:
        try:
            manifest = source.blob(tree[ICON_MANIFEST_PATH][1])
        except Exception as error:              # the parser runs without the manifest
            errors.append(f"{ICON_MANIFEST_PATH}: {error}")
    vocab = vocabulary(list(tree), manifest)
    chart = None
    if psd_edit.STATUS_CHART in tree:
        try:
            chart = psd_edit.status_fills(png_export.open_psd(source.blob(tree[psd_edit.STATUS_CHART][1])))
        except Exception as error:              # R5 reports every overlay without the chart
            errors.append(f"{psd_edit.STATUS_CHART}: {error}")
    context = standards.Context(frozenset(vocab.all_types), chart, psd_edit.story_sync())
    return context, vocab


# ========================
# One card
# ========================
def _save(out: str, rel: str, data: bytes) -> str:
    """An artifact under out/files/; returns its path relative to out."""
    path = join("files", rel)
    makedirs(dirname(join(out, path)), exist_ok=True)
    with open(join(out, path), "wb") as fh:
        fh.write(data)
    return path


def _crops(out: str, rel: str, before, after, box) -> None:
    if before is None or after is None or not box:
        return
    stem = join(out, "crops", splitext(rel)[0])
    makedirs(dirname(stem), exist_ok=True)
    for name, image in psd_edit.artifacts(before, after, tuple(box)).items():
        image.save(f"{stem}.{name}.png")


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def _recompress(out: str, result: Result, path: str, read: Callable[[], bytes]) -> None:
    """A PNG the push brought (read() gives it), recompressed losslessly (owner, 2026-10-23) when more than RECOMPRESS_MIN
    and RECOMPRESS_BYTES smaller (our thresholds): a write of its own kind, resting on that PNG alone and never told on
    Discord. A failure is a comment note, never the card's error (which Discord would tell)."""
    try:
        data = read()
        smaller = png_export.recompress(data, png_export.RECOMPRESS_MIN, png_export.RECOMPRESS_BYTES)
    except Exception as error:                  # it only saves bytes: the card goes on
        print(format_exc(), file=stderr)
        result.notes.append((f"{basename(path)} not recompressed ({type(error).__name__}: {error})", False))
        return
    if smaller is not None:
        result.writes.append({"path": path, "file": _save(out, path, smaller), "sha": blob_sha(smaller), "kind": "recompress",
                              "line": f"recompressed losslessly ({_size(len(data))} -> {_size(len(smaller))}, the same pixels)"})


def _export_dict(found) -> dict:
    """An Export as plan.json keeps it; "unchecked" is the line told for a PNG made with no stored image (G2), or for one
    kept (KEPT beside such a PSD, RESTYLED over one)."""
    return {"outcome": found.outcome, "why": found.why, "gate": found.gate, "changed_px": found.changed_px,
            "box": list(found.box) if found.box else None, "icc": found.icc, "writer": found.writer,
            "vote": list(found.vote) if found.vote else None, "numbers": found.numbers,
            "misses": png_export.miss_notes(found),
            "unchecked": png_export.unchecked_line(found)}


def _r10_line(job: Job, measured: dict) -> str:
    bad = measured.get("bad_pct", 0)
    if job.case == "png-only":
        return f"{basename(job.png)} does not match its PSD ({bad:.1f}% of pixels differ): push the PSD too?"
    return f"{basename(job.png)} does not match the PSD ({bad:.1f}% of pixels differ): export it again"


def added(job: Job) -> bool:
    """A new card PSD (the new-card rules, R4 in full): no PNG yet, beside a PNG, or added with its PNG or with it deleted."""
    return job.case in ("new", "beside") or (job.case in ("pair", "psd-only") and not job.psd_old)


def run_job(source: Source, job: Job, context, parser, out: str, mode: str) -> Result:
    """
    Both guards on one card; never raises (an error is the result's error line).

    Args:
        source: The card repo
        job: The card's job
        context: card_standards.Context
        parser: The offline parser (the fixes' reparse gate)
        out: The artifacts dir
        mode: dry, report or on (wording only: the writes are planned either way, land commits them)
    """
    result = Result(job)
    try:
        _run(source, job, context, parser, out, result)
    except Exception as error:                  # one bad card never stops the others
        result.error = f"{type(error).__name__}: {error}"
        result.notes.append((f"error: {result.error}", False))
        print(format_exc(), file=stderr)
    return result


def _run(source: Source, job: Job, context, parser, out: str, result: Result) -> None:
    for path, sha in sorted(job.pushed.items()):
        _recompress(out, result, path, partial(source.blob, sha))
    if job.case == "numbered":
        return
    if job.case == "deleted":
        result.notes.append((f"{basename(job.png)} is left without its PSD", False))
        return
    if job.case == "png-deleted":
        result.notes.append((f"{basename(job.png)} was deleted; the PSD stays", False))
        return
    if job.case == "rename":
        data = source.blob(job.copy_from)
        smaller = png_export.recompress(data, png_export.RECOMPRESS_MIN, png_export.RECOMPRESS_BYTES)
        line = "the PNG copied to the PSD's new name" + (
            f"; recompressed losslessly ({_size(len(data))} -> {_size(len(smaller))})" if smaller else "")
        data = smaller or data
        file = _save(out, job.png, data)
        result.export = {"outcome": "re-exported", "why": "", "copied": True}
        result.writes.append({"path": job.png, "file": file, "sha": blob_sha(data), "kind": "png", "line": line})
        if job.left:
            result.notes.append((f"{basename(job.left)} is left at the PSD's old name", False))
        return
    new_bytes = source.blob(job.psd_new)
    new = png_export.open_psd(new_bytes)
    old = png_export.open_psd(source.blob(job.psd_old)) if job.psd_old else None
    if old is not None:
        result.changes = changes(old, new, parser)
    rule_page = standards.kind_of(job.psd) == "rule"
    # guard 1
    if job.case in ("stale", "beside"):
        old_png = png_export.open_png(source.blob(job.png_old))
        numbered = {path: png_export.open_png(source.blob(sha)) for path, sha in job.numbered.items()}
        found = png_export.export(old if job.case == "stale" else None, new, old_png, numbered)
        # a burst: when the old PNG shows the anchor rather than the old PSD (not a PNG pushed before its PSD), the
        # anchored verdict, a refusal too (a small push-1 change hides under G5's bar, review 2026-10-24); else, checked,
        # a G5/G6 refusal retried, the anchored export kept unless it refuses
        if job.psd_anchor and old is not None:
            anchor = png_export.open_psd(source.blob(job.psd_anchor))
            shown = png_export.shows(old_png, anchor, old)
            if shown or (not found.unchecked and found.outcome == "refused" and found.gate in ("G5", "G6")):
                anchored = png_export.export(anchor, new, old_png, numbered)
                if shown or anchored.outcome != "refused":   # the PNG matched the PSD before the burst
                    found = anchored
                    result.notes.append((f"checked against the PSD as of {basename(job.png)}'s last change "
                                         f"(`{job.anchor_commit[:8]}`): an earlier push changed the PSD without its PNG",
                                         False))
        result.export = _export_dict(found)
        stale = found.numbers.get("png_vs_psd", {})
        if found.outcome == "current" and stale.get("px") and not psd_edit._within_bar(stale):
            off = (f"{stale['bad_pct']:.1f}% of pixels differ" if stale["bad_pct"] > psd_edit.PARITY_BAD_MAX
                   else f"mean difference {stale['mean']:.1f}")
            why = "the PSD is new beside it" if job.case == "beside" else "though this push moved no pixel"
            result.notes.append((f"{basename(job.png)} does not match the PSD ({off}), {why}: export it in Photoshop",
                                 TELL_PNG_MISMATCH))
        if found.outcome == "re-exported":
            file = _save(out, job.png, found.data)
            result.writes.append({"path": job.png, "file": file, "sha": blob_sha(found.data), "kind": "png",
                                  "line": f"re-exported by the bot's render of the PSD ({found.changed_px:,} px changed)"
                                          + "".join(f"; {note}" for note in png_export.miss_notes(found))})
            _crops(out, job.png, old_png, found.image, found.box)
            refused = []
            for path, (data, why) in sorted(found.numbered.items()):
                if data:
                    result.writes.append({"path": path, "file": _save(out, path, data), "sha": blob_sha(data), "kind": "png",
                                          "line": "updated with the same change (delta transfer)"})
                else:
                    refused.append((why, path))
            result.export["numbered_refused"] = refused
    elif job.case == "new":
        found = png_export.export_new(new)
        result.export = _export_dict(found)
        if found.outcome == "re-exported":
            file = _save(out, job.png, found.data)
            result.writes.append({"path": job.png, "file": file, "sha": blob_sha(found.data), "kind": "png",
                                  "line": "made by the bot's render of the PSD (new card)"
                                          + "".join(f"; {note}" for note in png_export.miss_notes(found))})
    # R10: a pushed PNG against the PSD
    if job.case in ("pair", "png-only", "whole") and job.png_new:
        data = source.blob(job.png_new)
        if job.case != "whole":
            _recompress(out, result, job.png, lambda: data)
        png = png_export.open_png(data)
        measured = png_export.png_check(new, png)
        result.r10 = {**measured, "line": _r10_line(job, measured) if measured.get("ok") is False else ""}
        if measured.get("ok") is None:
            result.notes.append((f"{basename(job.png)} could not be checked against the PSD ({measured.get('why')})", False))
    if job.case == "png-only" or rule_page:
        return
    # guard 2
    issues = standards.check(new, job.psd, context, old=old, added=added(job))
    told = {id(issue) for issue in standards.reported(issues)}
    result.issues = [{**issue.as_dict(), "line": issue.line(), "told": id(issue) in told, "discord": issue.discord}
                     for issue in issues]
    if any(issue.fix and not issue.waived for issue in issues):
        fixed = standards.fix(new_bytes, job.psd, issues, join(out, "work"), parser)
        result.fixed = fixed.applied
        lines = {issue.code: issue.line() for issue in issues if issue.fix}
        result.fix_refused = [(lines.get(code, code), why) for code, why in fixed.refused]
        if fixed.data:
            file = _save(out, job.psd, fixed.data)
            result.writes.append({"path": job.psd, "file": file, "sha": blob_sha(fixed.data), "kind": "psd",
                                  "line": "; ".join(fixed.applied)})


def changes(old, new, parser) -> list:
    """What changed inside the PSD (psd_diff), icons named by the parser's vocabulary; [] when psd_diff fails (it only
    tells, so it never stops a card)."""
    try:
        return psd_diff.changes(old, new, getattr(getattr(parser, "icons", None), "codepoints", None))
    except Exception:                           # report only: the guards go on without it
        print(format_exc(), file=stderr)
        return []


def _init_worker(clone: str, repository: str, context, vocab) -> None:
    token_from_env()        # a spawned worker (macOS; forkserver on 3.14) starts without main()'s header
    _worker.update(source=Source(clone, repository), context=context, parser=vocab.parser())


def _pooled(item: tuple) -> Result:
    job, out, mode = item
    return run_job(_worker["source"], job, _worker["context"], _worker["parser"], out, mode)


# ========================
# Reports
# ========================
def _expect(job: Job) -> dict:
    """The blobs a card's writes rest on at the head (G9): the PNG as checked ("" for one the bot makes)."""
    expect = {job.psd: job.psd_new}
    if job.png:
        expect[job.png] = job.png_old if job.case in ("stale", "beside") else job.png_new
    for path, sha in (*job.numbered.items(), *job.pushed.items()):
        expect[path] = sha
    return expect


def plan(results: list, repository: str, before: str, after: str, mode: str) -> dict:
    """plan.json: the writes per card with the blobs they rest on, and the verdicts."""
    cards = []
    for result in results:
        entry = {"card": result.name, "psd": result.job.psd, "case": result.job.case, "login": result.job.login,
                 "commit": result.job.commit, "export": result.export, "r10": result.r10, "issues": result.issues,
                 "fixed": result.fixed, "fix_refused": result.fix_refused, "notes": [t for t, _ in result.notes],
                 "told": result.told(mode), "error": result.error, "writes": result.writes, "png": result.job.png,
                 "changes": result.changes, "expect": _expect(result.job), "pushed": sorted(result.job.pushed)}
        cards.append(entry)
    return {"repo": repository, "before": before, "after": after, "mode": mode, "checked_at": _now().isoformat(),
            "cards": cards}


def _box(numbers: dict, box) -> str:
    if not box:
        return ""
    x0, y0, x1, y1 = box
    return f" in {x1 - x0}x{y1 - y0} at ({x0}, {y0})"


def _guard1_line(entry: dict, mode: str = "on") -> str:
    found = entry["export"]
    outcome = found.get("outcome")
    if not outcome:
        return ""
    if outcome == "current" and "png_ahead" in found.get("numbers", {}):
        return "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"
    if outcome == "current" and found.get("unchecked"):
        return found["unchecked"]
    if outcome == "current":
        return "PNG still current (the push moved no pixel of the card)"
    if outcome == "refused":
        return found["why"]
    would = " would be" if mode != "on" else ""
    switch = " with CARD_GUARDS=on" if mode != "on" else ""
    if found.get("copied"):
        return f"PNG{would} copied to the PSD's new name{switch}"
    numbers = found.get("numbers", {})
    parts = [f"{found['changed_px']:,} px changed{_box(numbers, found.get('box'))}", f"ICC {found.get('icc') or 'as stored'}"]
    if "baseline" in numbers:
        parts.append(f"the old PSD drawn against the old PNG: bad {numbers['baseline'].get('bad_pct', 0):.2f}%")
    # png_vs_render (G2 beside) is plan.json's alone: it measures the author's change (default 2026-10-24, owner to confirm)
    if "cross" in numbers:
        parts.append(f"against the PSD's own image: bad {numbers['cross'].get('bad_pct', 0):.2f}%, "
                     f"mean {numbers['cross'].get('mean', 0):.2f}")
    if found.get("writer") in ("photoshop", "tool"):
        parts.append("saved by Photoshop" if found["writer"] == "photoshop" else "not saved by Photoshop")
    if found.get("unchecked"):
        parts.append("no stored image to cross-check (saved without Maximize Compatibility)")
    verb = "made" if entry["case"] == "new" else "re-exported"
    misses = "".join(f"; {note}" for note in found.get("misses", []))
    return f"PNG{would} {verb} by the bot's render of the PSD ({'; '.join(parts)}){switch}{misses}"


ALREADY = "already on the branch (an earlier run of this push landed it)"
UNREAD = "an earlier run may have landed it (the branch could not be read)"


def _already(landed: dict, card: str) -> set:
    """The paths of a card's writes the branch already held (a re-run after a landing)."""
    return {path for name, path in landed.get("already", []) if name == card}


def _landed_head(landed: dict) -> str:
    sha, error = landed.get("sha", ""), landed.get("error", "")
    left = len({card for card, *_ in landed.get("dropped", [])} - set(landed.get("landed", [])))
    already = {card for card, _ in landed.get("already", [])} - set(landed.get("landed", []))
    out = (f"Landed in `{sha[:8]}`: {plural(len(landed.get('landed', [])), 'card')}." if sha else
           f"Not landed: {error.rstrip('.')}. Nothing was committed by this run; {UNREAD}." if error and landed.get("unread") else
           f"Not landed: {error.rstrip('.')}. Nothing was committed." if error else
           f"Nothing new to land: {plural(len(already), 'card')} {ALREADY}." if already else "Nothing landed.")
    return out + (f" {plural(left, 'card')} left out: changed again on the branch." if left and not error else "")


def _landing(entry: dict, landed: dict) -> list:
    """The comment's lines on what became of a card's writes after the quiet window."""
    if not entry["writes"]:
        return []
    already = _already(landed, entry["card"])
    if landed.get("error"):
        if already and len(already) == len(entry["writes"]):
            return [f"- Not committed again: {ALREADY}"]
        if landed.get("unread"):
            return [f"- Not committed by this run: {UNREAD}"]
        return ["- Not landed (the bot's commit failed): nothing above was committed for this card"]
    dropped = {path: why for card, why, path in landed.get("dropped", []) if card == entry["card"]}
    if len(dropped) == len(entry["writes"]):
        return [f"- Not landed ({next(iter(dropped.values()))}): nothing above was committed for this card"]
    out = [f"- Not landed ({why}): `{path}`" for path, why in dropped.items()]
    if landed.get("sha") and any(w["path"] not in dropped and w["path"] not in already for w in entry["writes"]):
        out.append(f"- {'The rest landed' if out else 'Landed'} in `{landed['sha'][:8]}`")
    if already:
        out.append(f"- {'The rest is' if out else 'Every write is'} {ALREADY}")
    return out


def markdown(data: dict, notes: list, skipped, errors: list, landed: Optional[dict] = None) -> str:
    """The job summary and commit comment; with land's report (landed.json), the comment as edited after the quiet window."""
    mode = data["mode"]
    head = {"dry": "Dry run: nothing is posted or committed.", "report": "Report only (CARD_GUARDS=report): nothing is committed.",
            "on": "Writes land after the 30-minute quiet window, in one bot commit."}[mode]
    final = landed is not None and not landed.get("dry")
    if final:
        head = _landed_head(landed)
    out = ["## Card guards", "", f"Push `{data['after'][:8]}` (against `{data['before'][:8]}`): "
                                 f"{plural(len(data['cards']), 'card')}. {head}"]
    for note in notes:
        out.append(f"- {note}")
    if skipped:
        out.append(f"- Left out (bot commits): {', '.join(f'`{p}`' for p in sorted(skipped))}")
    out.append("")
    if not data["cards"]:
        out.append("No card PSD or PNG to guard.")
    for entry in data["cards"]:
        out.append(f"**{entry['card']}** (`{entry['psd']}`, {entry['login']}, `{entry['commit'][:8]}`)")
        if entry.get("changes"):
            out.append(f"- Changed: {_changes_line(entry, CHANGES_COMMENT)}")
        line = _guard1_line(entry, mode)
        if line:
            out.append(f"- {line}")
        for why, path in entry["export"].get("numbered_refused", []):
            out.append(f"- {why}")
        if entry["r10"].get("ok") is False:
            out.append(f"- {entry['r10']['line']}")
        elif entry["r10"].get("ok"):
            out.append(f"- {basename(entry['psd'])[:-4]}.png matches the PSD (mean {entry['r10'].get('mean', 0):.2f})")
        for write in entry.get("writes", []):
            if write.get("kind") == "recompress":
                out.append(f"- {basename(write['path'])} {write['line']}" if mode == "on" else
                           f"- {basename(write['path'])} would be {write['line']} with CARD_GUARDS=on")
        told = [i["line"] for i in entry["issues"] if i["told"]]
        if told:
            out.append("- Issues: " + "; ".join(told))
        if entry["fixed"]:
            psd_landed = final and (entry["psd"] in _already(landed, entry["card"]) or (
                landed.get("sha") and not landed.get("error") and not any(
                    card == entry["card"] and path == entry["psd"] for card, _, path in landed.get("dropped", []))))
            verb = ("The bot would fix (CARD_GUARDS=on)" if mode != "on" else "The bot will fix" if not final else
                    "Fixed by the bot" if psd_landed else "Not fixed by this run" if landed.get("unread") else "Not fixed")
            out.append(f"- {verb}: " + "; ".join(entry["fixed"]))
        for line, why in entry["fix_refused"]:
            out.append(f"- Not fixed: {line} ({why})")
        waived = [f"{i['code']} {i['line']}" for i in entry["issues"] if i["waived"]]
        if waived:
            out.append("- Waived (owner decisions): " + "; ".join(waived))
        for note in entry["notes"]:
            out.append(f"- {note}")
        if final:
            out += _landing(entry, landed)
        for note in (landed or {}).get("after", {}).get(entry["psd"], []):
            out.append(f"- After the quiet window: {note}")
        if not (line or told or entry["fixed"] or entry["fix_refused"] or entry["notes"] or entry["r10"] or entry.get("writes")):
            out.append("- Nothing to recompress." if entry.get("case") == "numbered" else "- Meets the standards.")
        out.append("")
    if errors:
        out += ["Errors:"] + [f"- {e}" for e in errors] + [""]
    return "\n".join(out)


def _changes_line(entry: dict, limit: int) -> str:
    """The card's changes (psd_diff) as markdown text: card text and layer names never format or link."""
    return psd_diff.line([sub(MARKDOWN, r"\\\1", phrase) for phrase in entry.get("changes", [])], limit)


def _changed(entry: dict) -> str:
    """" (what changed)" after the card's name in its Discord line; "" for a new PSD or a re-save."""
    text = _changes_line(entry, CHANGES_DISCORD)
    return f" ({text})" if text else ""


def discord_payload(data: dict, repository: str) -> dict:
    """
    The webhook body: one LINE per card with something to tell (a refusal or a remaining issue), with what changed in
    its PSD; to fit DISCORD_LIMIT, cards lose their changes (the last card first) before whole lines go from the end; {}
    when nothing. Nobody is pinged and links get no preview.
    """
    url = f"https://github.com/{repository}/commit/"
    told = [entry for entry in data["cards"] if entry["told"]]

    def card_line(entry: dict, changes: bool = True) -> str:
        return LINE.format(login=entry["login"], card=entry["card"], changes=_changed(entry) if changes else "",
                           issues="; ".join(entry["told"]).rstrip("."), short=(entry["commit"] or data["after"])[:8],
                           url=url + (entry["commit"] or data["after"])[:URL_SHA])
    lines = [card_line(entry) for entry in told]
    if not lines:
        return {}

    def assemble(kept: int) -> str:
        rest = len(lines) - kept
        more = [f"... and {plural(rest, 'more card')}: [details]({url}{data['after']})"] if rest else []
        return "\n".join(lines[:kept] + more)

    kept = len(lines)
    content = assemble(kept)
    for i in range(len(lines) - 1, -1, -1):     # the issues are what the post is for
        if len(content) <= DISCORD_LIMIT:
            break
        lines[i] = card_line(told[i], False)
        content = assemble(kept)
    while len(content) > DISCORD_LIMIT and kept > 0:
        kept -= 1
        content = assemble(kept)
    return {"content": content, "allowed_mentions": {"parse": []}, "flags": SUPPRESS_EMBEDS}


# ========================
# check
# ========================
def check(clone: str, after: str, before: str = "", repository: str = "", author: str = NOBODY, cards: Optional[set] = None,
          whole_tree: bool = False, mode: str = "dry", out: str = "guards", workers: int = 0) -> dict:
    """
    Both guards on a push (or a whole tree); writes report.md, discord.json and plan.json (with the artifacts) to out.

    Returns:
        plan.json's data (with "markdown" and "discord")
    """
    makedirs(out, exist_ok=True)
    source = Source(clone, repository)
    errors: list = []
    if whole_tree:
        head = source.commit(after)
        if not head:
            raise Transient(f"the clone has no commit {after[:8]}")
        tree, base, notes, skipped = source.tree(head), "", ["the whole tree: standards and PNG-vs-PSD only"], {}
        jobs = whole_tree_jobs(tree, author, cards)
    else:
        base, head, notes = push_range(source, before, after)
        jobs, skipped, more, tree = plan_jobs(source, base, head, author, cards)
        notes += more
    context, vocab = build_context(source, tree, errors) if jobs else (standards.Context(), None)
    if len(jobs) >= POOL_MIN and workers != 1:
        count = workers or max(1, min((cpu_count() or 2) // 2, len(jobs)))
        with ProcessPoolExecutor(count, initializer=_init_worker, initargs=(clone, repository, context, vocab)) as pool:
            results = list(pool.map(_pooled, [(job, out, mode) for job in jobs], chunksize=2))
    else:
        parser = vocab.parser() if vocab is not None else None
        results = [run_job(source, job, context, parser, out, mode) for job in jobs]
    data = plan(results, repository, base, head, mode)
    errors += [f"{r.job.psd}: {r.error}" for r in results if r.error]
    data["report"] = {"notes": notes, "skipped": sorted(skipped), "errors": errors}   # landed_markdown re-renders from these
    data["markdown"] = markdown(data, notes, skipped, errors)
    data["discord"] = discord_payload(data, repository or CARD_REPO)
    with open(join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(data["markdown"])
    with open(join(out, "discord.json"), "w", encoding="utf-8") as fh:
        dump(data["discord"], fh)
    with open(join(out, "plan.json"), "w", encoding="utf-8") as fh:
        dump({k: v for k, v in data.items() if k not in ("markdown", "discord")}, fh, indent=1)
    data["summary"] = summary(data, errors)
    with open(join(out, "summary.json"), "w", encoding="utf-8") as fh:
        dump(data["summary"], fh)
    return data


def _for_comment(entry: dict) -> bool:
    """A card the commit comment has something on: told, written, an error, or comment-only lines (R10, a non-Discord
    note, a told issue, a fix or a refused one)."""
    return bool(entry["told"] or entry["writes"] or entry["error"] or entry["notes"] or entry["r10"].get("ok") is False
                or entry["fixed"] or entry["fix_refused"] or any(issue["told"] for issue in entry["issues"]))


def summary(data: dict, errors: list) -> dict:
    """summary.json, the workflow's switches: cards checked; cards with something told (Land, the post), for the comment
    (the Comment step: a context error alone, a standing manifest or chart one say, comments on no push) and with an error
    (a red run); writes planned; errors (the cards' and the context's)."""
    cards = data["cards"]
    return {"after": data["after"], "cards": len(cards), "told": sum(1 for c in cards if c["told"]),
            "comment": sum(1 for c in cards if _for_comment(c)), "writes": sum(len(c["writes"]) for c in cards),
            "errors": len(errors), "card_errors": sum(1 for c in cards if c["error"])}


# ========================
# land
# ========================
def landed_markdown(data: dict, landed: dict) -> str:
    """The commit comment after land: what landed, and the cards left out (their PNG and fixes were not committed)."""
    report = data.get("report", {})
    return markdown(data, report.get("notes", []), report.get("skipped", []), report.get("errors", []), landed)


def message(landed: list, after: str) -> str:
    """The bot commit's message: subject with the counts, one line per file, then the trailers: one Recompressed per PNG
    only recompressed, the Guarded-push."""
    pngs = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "png")
    psds = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "psd")
    smaller = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "recompress")
    counts = ", ".join(part for part in (plural(pngs, "PNG") + " re-exported" if pngs else "",
                                         plural(psds, "PSD") + " fixed" if psds else "",
                                         plural(smaller, "PNG") + " recompressed" if smaller else "") if part)
    lines = [f"{w['path']}: {w['line']} ({entry['login']}, {(entry['commit'] or after)[:8]})"
             for entry in landed for w in entry["writes"]]
    trailers = sorted(f"{RECOMPRESSED}: {w['path']}" for entry in landed for w in entry["writes"] if w["kind"] == "recompress")
    trailers.append(f"{TRAILER}: {after}")
    return f"{SUBJECT}: {counts} (push {after[:8]})\n\n" + "\n".join(lines) + "\n\n" + "\n".join(trailers) + "\n"


def _rests_on(entry: dict, write: dict) -> list:
    """The paths a write rests on: a PSD fix (0 pixels) on its PSD alone, a recompressed PNG on itself, a PNG on the whole
    card (PSD, PNG, numbered) but the numbered PNGs the push brought (only their recompressions rest on them)."""
    if write["kind"] == "psd":
        return [entry["psd"]]
    pushed = set(entry.get("pushed", ()))
    return [write["path"]] if write["kind"] == "recompress" else sorted(p for p in entry["expect"] if p not in pushed)


class Head:
    """A branch head's blobs, looked up in any letter case (a teammate's foo.png stands for the bot's Foo.png, L12)."""

    def __init__(self, tip: Snapshot):
        self.tree = tip.tree
        self.lower = {path.lower(): sha for path, (_, sha) in tip.tree.items()}

    def __call__(self, path: str) -> str:
        return self.tree[path][1] if path in self.tree else self.lower.get(path.lower(), "")


def _ours(entry: dict) -> dict:
    """{path: blob} of a card's planned writes."""
    return {write["path"]: write["sha"] for write in entry.get("writes", [])}


def _holds(entry: dict, path: str, blob: str) -> bool:
    """The head still holds what a card's check rests on at path: the blob as checked, or the bot's own write of it."""
    return blob == entry.get("expect", {}).get(path, "") or (bool(blob) and blob == _ours(entry).get(path))


def _superseded(entry: dict, write: dict, here: Head, recent: Optional[set]) -> str:
    rests = _rests_on(entry, write)
    moved = [p for p in rests if not _holds(entry, p, here(p))]
    if moved:
        return f"superseded: {moved[0]} changed on the branch"
    if here(write["path"]) == write["sha"]:
        return ""                               # already there (checked by the caller): no later push to look for
    if recent is None:
        return "superseded: too many pushes to tell"
    touched = sorted(p for p in rests if p.lower() in {r.lower() for r in recent})
    return f"superseded: a later push touched {touched[0]}" if touched else ""


def _bot_changed(repository: str, local: str, head: str, path: str) -> bool:
    """The last commit on the branch that changed path is a bot's (no run of ours follows it); False when unknown."""
    try:
        if local:
            name, _, email = git(local, "log", "-1", "--format=%an%x1f%ae", head, "--", path).decode().strip().partition("\x1f")
        else:
            commits = api(repository, "GET", "commits", params={"sha": head, "path": path, "per_page": 1}).json()
            author = commits[0]["commit"]["author"] if commits else {}
            name, email = author.get("name", ""), author.get("email", "")
    except Exception:                           # unknown: a person's, as before (that push's run tells)
        return False
    return bool(name or email) and Writer("", name, email, "").bot


def after_window(data: dict, landed: dict, tip: Optional[Snapshot], bot_moved: frozenset = frozenset()) -> tuple:
    """
    What the one Discord post says after the quiet window (section 7.3), per card PSD: the check's lines, less a card whose
    PSD a person changed again (the later push's own check tells; a bot's change starts no run, so the lines stay) and a
    PNG refusal, or in report mode a PNG the bot would make, whose PNG was pushed since; plus the writes that did not land
    and were not replaced by the author (the bot's commit failed, a later push touched the card). A write the branch
    already holds (a re-run) is not told, nor any write when a re-run could not read the branch (an earlier attempt may
    have landed it). A numbered PNG's refusal (G8) is withdrawn as the main PNG's is, and so is the line of a PNG made with
    no stored image (G2) whose PNG was pushed since or did not land.

    Args:
        bot_moved: The PSDs whose last change on the branch is a bot commit's

    Returns:
        ({psd: [Discord lines]}, {psd: [comment notes]})
    """
    told, after = {}, {}
    dropped = {path: why for _, why, path in landed.get("dropped", [])}
    here = Head(tip) if tip is not None else None
    for entry in data["cards"]:
        psd, expect, lines, notes = entry["psd"], entry.get("expect", {}), list(entry.get("told", [])), []
        by_bot = psd in bot_moved
        if here and entry["case"] != "deleted" and psd in expect and not _holds(entry, psd, here(psd)) and not by_bot:
            told[psd], after[psd] = [], ["the PSD changed again on the branch: that push's own check tells"]
            continue
        if by_bot:
            notes.append("a bot commit changed the PSD since: the push's lines stand")
        export, png = entry.get("export", {}), entry.get("png", "")
        refused = [(export.get("why"), png)] if export.get("outcome") == "refused" and png else []
        for why, path in refused + [tuple(item) for item in export.get("numbered_refused", [])]:
            if here and here(path) != expect.get(path, ""):
                lines = [line for line in lines if line != why]
                notes.append(f"{basename(path)} was pushed since: the refusal is withdrawn")
        if data.get("mode") == "report":
            for write in entry.get("writes", []):
                if write["kind"] == "png" and here and not _holds(entry, write["path"], here(write["path"])):
                    lines = [line for line in lines if line != would_make(entry["case"], png, write)]
                    notes.append(f"{basename(write['path'])} was pushed since: not told")
        already = _already(landed, entry["card"])
        for write in [] if landed.get("dry") or landed.get("unread") else entry.get("writes", []):
            why = "the bot's commit failed" if landed.get("error") else dropped.get(write["path"], "")
            if not why or write["path"] in already or write["kind"] == "recompress":
                continue                        # landed (now or by an earlier run); a recompression is never told
            if here and not _holds(entry, write["path"], here(write["path"])) and not (by_bot and write["path"] == psd):
                continue                        # the author replaced that file: nothing to tell
            why = why.replace("superseded: ", "")
            if write["kind"] == "psd":
                lines.append(f"{write['line']} (the bot could not fix it: {why})")
            elif write["path"] == png:
                lines.append(f"PNG not made: {why}; export it in Photoshop")
            else:
                lines.append(f"{basename(write['path'])} not updated: {why}; re-export it in Photoshop")
        unchecked = export.get("unchecked")
        if unchecked in lines and png and (any(line.startswith("PNG not made: ") for line in lines)
                                           or (here and not _holds(entry, png, here(png)))):
            lines.remove(unchecked)             # no bot PNG stands: the author's, or none
        told[psd], after[psd] = list(dict.fromkeys(lines)), notes
    return told, after


def _read_files(out_dir: str, entries: list) -> dict:
    files = {}
    for entry in entries:
        for write in entry["writes"]:
            with open(join(out_dir, write["file"]), "rb") as fh:
                files[write["path"]] = fh.read()
            if blob_sha(files[write["path"]]) != write["sha"]:
                raise Transient(f"{write['file']} does not hash to the planned {write['sha'][:8]}")
    return files


def land(plan_path: str, repository: str = "", quiet_from: str = "", dry_run: bool = False, local: str = "",
         rerun: bool = False, sleep: Callable[[float], None] = _sleep, now: Callable[[], datetime] = _now) -> dict:
    """
    After the quiet window (quiet_from + RECENT), commit a plan's writes once (G9) and settle what the one Discord post
    says (after_window). A write is left out when its card no longer holds the blobs it rests on (a PSD fix rests on its
    PSD alone, a PNG on the whole card; any letter case) or, after a push's window, a later push touched them; a write the
    branch already holds (a re-run after a landing) is not written again. The rest is one commit on the branch head
    (retried on a moved head, COMMIT_TRIES times); --local commits in that clone instead (nothing pushed). dry_run (report
    mode) commits nothing. Without quiet_from (a dispatch) nothing waits and the blob check alone decides: the pushes in the
    last 30 minutes would include the checked push itself. Never raises once the plan is read: any failure is the error.
    rerun (a re-run or a dispatch: an earlier run may have landed the writes): when the branch could not be read, no
    write is told as not made ("unread").

    Returns:
        {"sha": the commit or "", "landed": [card], "dropped": [[card, why, path]], "already": [[card, path]],
         "error": why nothing was committed, "told": {psd: [Discord lines]}, "after": {psd: [comment notes]}, "dry": dry_run,
         "unread": a rerun that could not read the branch}
    """
    out_dir = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    repository = repository or data.get("repo") or ("" if local else CARD_REPO)
    entries = [entry for entry in data["cards"] if entry.get("writes")]
    report = {"sha": "", "landed": [], "dropped": [], "already": [], "error": "", "told": {}, "after": {}, "dry": dry_run,
              "unread": False}
    if not entries and not any(entry.get("told") for entry in data["cards"]):
        return report
    if quiet_from:
        mark = datetime.fromisoformat(quiet_from.replace("Z", "+00:00")) + RECENT + QUIET_MARGIN
        wait = (mark - now()).total_seconds()
        if wait > 0:
            print(f"Waiting {wait / 60:.1f} min for the quiet window.", file=stderr)
            sleep(wait)
    tip, bot_moved = None, frozenset()
    try:
        files = _read_files(out_dir, entries)
        uploaded: set = set()
        for _ in range(COMMIT_TRIES if not local else 1):
            tip = local_snapshot(repository, local) if local else snapshot_at(repository, ref_head(repository))
            here = Head(tip)
            # a clone alone has no push times; a dispatch's window would hold its own push. A first attempt's from the
            # push, not now: the post job starts 30 minutes after the check job ends, and pushes made during the check
            # count too. A re-run's from now, as before (from the push, a late one counts the whole day's pushes)
            recent = (recent_paths(repository, tip.head, now() if rerun else min(now(), mark), local)
                      if repository and quiet_from else set())
            included, report["dropped"], report["already"] = [], [], []
            for entry in entries:
                for write in entry["writes"]:
                    why = _superseded(entry, write, here, recent)
                    if why:
                        report["dropped"].append([entry["card"], why, write["path"]])
                    elif here(write["path"]) == write["sha"]:
                        report["already"].append([entry["card"], write["path"]])
                    else:
                        included.append((entry, write))
            cards = list(dict.fromkeys(entry["card"] for entry, _ in included))
            report["landed"] = cards
            if not included or dry_run:
                break
            payload = {write["path"]: files[write["path"]] for _, write in included}
            text = message([{**entry, "writes": [w for e, w in included if e is entry]} for entry in entries
                            if entry["card"] in cards], data["after"])
            if local:
                report["sha"] = local_commit(local, tip.head, payload, text)
                break
            modes = {path: tip.tree[path][0] for path in payload if path in tip.tree}
            sha = commit_files(repository, tip.head, tip.info["tree"]["sha"], modes, payload, text, uploaded)
            if sha:
                report["sha"] = sha
                break
        else:
            raise Transient(f"the branch kept moving: not committed after {COMMIT_TRIES} tries")
    except Exception as error:                  # the API, git, a bad file: told as not made, never a lost post
        if not isinstance(error, (Transient, Refused)):
            print(format_exc(), file=stderr)
        why = str(error) if isinstance(error, (Transient, Refused)) else f"{type(error).__name__}: {error}"
        report.update(sha="", landed=[], error=why or type(error).__name__, unread=rerun and tip is None)
    if tip is not None:
        here = Head(tip)
        bot_moved = frozenset(entry["psd"] for entry in data["cards"]
                              if entry["case"] != "deleted" and entry["psd"] in entry.get("expect", {})
                              and not _holds(entry, entry["psd"], here(entry["psd"]))
                              and _bot_changed(repository, local, tip.head, entry["psd"]))
    report["told"], report["after"] = after_window(data, report, tip, bot_moved)
    return report


def finish(plan_path: str, landed: dict) -> dict:
    """land's files beside the plan: landed.json, report-landed.md (the comment's new body) and discord-final.json (the
    one post, as after_window settled it). Returns the post."""
    out_dir = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    with open(join(out_dir, "landed.json"), "w", encoding="utf-8") as fh:
        dump(landed, fh)
    with open(join(out_dir, "report-landed.md"), "w", encoding="utf-8") as fh:
        fh.write(landed_markdown(data, landed))
    told = landed.get("told", {})
    final = {**data, "cards": [dict(entry, told=told.get(entry["psd"], entry.get("told", []))) for entry in data["cards"]]}
    post = discord_payload(final, data.get("repo") or CARD_REPO)
    with open(join(out_dir, "discord-final.json"), "w", encoding="utf-8") as fh:
        dump(post, fh)
    return post


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """check: the report on stdout (exit 0 whatever it finds, 1 when it could not check); land: one line per card, exit 1
    when the commit failed (its files are written either way), exit 2 for a plan not checked with --mode on that would
    commit to the repo (neither --dry-run nor --local)."""
    cli = ArgumentParser(prog="python -m commands.card_guards",
                         description="The push guards: a forgotten PNG drawn from the PSD by the fork (the bake's render), the card "
                                     "standards checked (0-pixel fixes planned). check never writes to the card repo; land commits a plan once.")
    sub = cli.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help="both guards on a push (or the whole tree)")
    one.add_argument("--clone", required=True, metavar="DIR", help="a clone of the card repo with the push's history (blobless is enough)")
    one.add_argument("--after", required=True, metavar="SHA", help="the push's head")
    one.add_argument("--before", default="", metavar="SHA", help="the push's previous head; empty: the first parent")
    one.add_argument("--repo", default="", metavar="OWNER/NAME", help="blobs the clone lacks and the commit logins by API")
    one.add_argument("--author", default=NOBODY, metavar="LOGIN",
                     help="the pusher, named when a login is unknown (a push's github.actor; a dispatch passes none)")
    one.add_argument("--cards", default="", metavar="NAMES", help="only these cards (comma separated, matched like Card Authors.txt)")
    one.add_argument("--whole-tree", action="store_true", help="every card PSD at --after: standards and PNG-vs-PSD")
    one.add_argument("--mode", default="dry", choices=MODES, type=str.lower,
                     help="dry, report or on (CARD_GUARDS, any case): the report's wording")
    one.add_argument("--workers", type=int, default=0, metavar="N", help="processes for 8 cards or more (0: half the cores)")
    one.add_argument("--out", required=True, metavar="DIR", help="report.md, discord.json, plan.json and the artifacts")
    two = sub.add_parser("land", help="commit a plan's writes once, after the quiet window")
    two.add_argument("--plan", required=True, metavar="FILE", help="check's plan.json (its files beside it)")
    two.add_argument("--repo", default="", metavar="OWNER/NAME", help="the card repo (default: the plan's)")
    two.add_argument("--quiet-from", default="", metavar="ISO", help="the push time: land waits until 30 min after it")
    two.add_argument("--dry-run", action="store_true", help="decide what would land, commit nothing (needed for a plan not checked with --mode on)")
    two.add_argument("--local", default="", metavar="DIR", help="commit in this clone (nothing pushed)")
    two.add_argument("--rerun", action="store_true",
                     help="a re-run or a dispatch: an earlier run may have landed the writes, so an unread branch tells none as not made")
    options = cli.parse_args(args)
    if options.command == "land" and not (options.dry_run or options.local):
        with open(options.plan, encoding="utf-8") as fh:
            planned = load(fh).get("mode", "dry")
        if planned != "on":                     # a dry or report check must never reach the branch
            cli.error(f"the plan was checked with --mode {planned}: land commits only an `on` plan (pass --dry-run, or --local DIR)")
    token_from_env()
    if options.command == "check":
        cards = {CardAuthors.key(n) for n in options.cards.split(",") if n.strip()} or None
        try:
            data = check(options.clone, options.after, options.before, options.repo, options.author, cards, options.whole_tree,
                         options.mode, options.out, options.workers)
        except (Transient, Refused) as error:   # the clone or the API failed: one line; the run fails (card-guards.yml alerts)
            print(f"## Card guards\n\nNot checked: {error}\n")
            return 1
        print(data["markdown"], end="")
        return 0
    report = land(options.plan, options.repo, options.quiet_from, options.dry_run, options.local, options.rerun)
    finish(options.plan, report)
    if report["error"]:
        print(f"Not landed: {report['error']}")
    for card in report["landed"]:
        print(f"{card}: {'landed in `' + report['sha'][:8] + '`' if report['sha'] else 'would land'}")
    for card, why, path in report["dropped"]:
        print(f"{card}: {why} ({path})")
    for card, path in report["already"]:
        print(f"{card}: {ALREADY} ({path})")
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BOT_EMAILS',
    'Change',
    'Head',
    'Job',
    'LINE',
    'RECOMPRESSED',
    'Result',
    'SPARSE_FILES',
    'Source',
    'TELL_PNG_MISMATCH',
    'Writer',
    'added',
    'build_context',
    'card_png',
    'changes',
    'check',
    'discord_payload',
    'after_window',
    'finish',
    'land',
    'landed_markdown',
    'main',
    'markdown',
    'message',
    'plan',
    'plan_jobs',
    'push_range',
    'run_job',
    'summary',
    'whole_tree_jobs',
]
