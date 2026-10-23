"""
The push guards (docs/Campaigns/PUSH_GUARDS.md): TTSCardMaker's card-guards.yml runs this on every push to main. Guard 1
(png_export) gives a card PSD pushed without its PNG the PNG from Photoshop's stored composite; guard 2 (card_standards)
checks the pushed PSDs against the card standards and fixes only the 0-pixel classes. One report per push: the job
summary and commit comment (report.md), one Discord post for the standards channel (discord.json: no mentions, no
previews), and a plan of the writes (plan.json) that `land` commits once, after the 30-minute quiet window.

    python -m commands.card_guards check --clone <dir> --after <sha> [--before <sha>] [--repo <owner/name>] [--author <login>]
                                         [--cards <names>] [--whole-tree] [--mode dry|report|on] [--fonts <dir>] --out <dir>
    python -m commands.card_guards land --plan <out>/plan.json [--quiet-from <ISO time>] [--dry-run] [--local <clone>]

check reads trees from a local clone (the Action's blobless checkout with its history; a full clone locally) and blobs
from it or, when it lacks them, by sha through the API (verified). It writes nothing to the card repo, ever: a dry run
and a replay are the same command. land re-reads the branch head, drops a card whose PSD or PNG moved or that a later
push touched (card_bake.recent_paths), and makes one commit (git_commit.commit_files; --local: in a clone, nothing
pushed). The guards' own commits and the bake's, the save's and Card sync's are skipped (BOT_SUBJECTS, the AutoCroissant
author). No Discord or config imports (the token: config through github_api, else GITHUB_TOKEN); SPARSE_FILES is the
import closure card-guards.yml checks out.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from json import dump, dumps, load, loads
from os import cpu_count, environ, makedirs
from os.path import basename, dirname, join, splitext
from subprocess import CalledProcessError, TimeoutExpired
from sys import stderr
from time import sleep as _sleep
from traceback import format_exc
from typing import Callable, Optional

import commands.card_standards as standards
import commands.png_export as png_export
import commands.psd_edit as psd_edit
from commands.card_bake import RECENT, Snapshot, fetch_verified, local_snapshot, recent_paths, snapshot_at
from commands.card_check import vocabulary
from commands.card_edits import CardAuthors, CardEdits
from commands.font_check import FontIndex, is_card_psd
from commands.git_commit import (AUTHOR, COMMIT_TRIES, NO_LAZY, Transient, api, commit_files, git, local_commit, plural,
                                 ref_head, token)
from commands.github_api import blob_sha, token_from_env
from commands.psd_edit import Refused
from global_config import CARD_REPO, EDITS_PATHS, ICON_MANIFEST_PATH

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_guards.py", "commands/png_export.py", "commands/card_standards.py", "commands/txt2_repair.py",
                "commands/card_bake.py", "commands/card_check.py", "commands/psd_edit.py", "commands/psd_regions.py",
                "commands/psd_parser.py", "commands/card_models.py", "commands/card_edits.py", "commands/font_check.py",
                "commands/github_api.py", "commands/git_commit.py", "commands/repo_history.py", "global_config.py")
SUBJECT = "Card guards"                         # the bot commit: "Card guards: 2 PNGs re-exported, 1 PSD fixed (push 7cb4c867)"
TRAILER = "Guarded-push"                        # the push the commit guards
BOT_SUBJECTS = ("Card guards", "Card edits", "Card edit:", "TTS save", "Card sync")   # bot commits: never guarded
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZEROS = "0" * 40
MODES = ("dry", "report", "on")
LINE = "{login} pushed {card} with these issues: {issues}. [{short}]({url})"
DISCORD_LIMIT = 2000
SUPPRESS_EMBEDS = 4                             # Discord message flag: no link previews
NOBODY = "Someone"
QUIET_MARGIN = timedelta(seconds=60)            # land waits RECENT plus this after the push
POOL_MIN = 8                                    # --whole-tree checks this many cards or more in a process pool
CARD_ROOTS = ("Auxiliary/", "Creatures/", "Field/", "Items/", "N.M.E/")   # where card PNGs live (Types/ icons are not cards)
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
        return (self.email == AUTHOR["email"] or self.name == AUTHOR["name"]
                or any(self.subject.startswith(prefix) for prefix in BOT_SUBJECTS))


@dataclass
class Job:
    """One card's work in the push."""
    psd: str                                    # the PSD path at after (the deleted one's for "deleted")
    case: str                                   # stale, pair, new, beside, rename, psd-only, png-only, deleted, png-deleted, whole
    psd_old: str = ""                           # its blob before the push
    psd_new: str = ""                           # its blob at after
    png: str = ""                               # the PNG path (to check, or to write)
    png_old: str = ""                           # the PNG blob an export rests on
    png_new: str = ""                           # the pushed PNG (R10)
    numbered: dict = field(default_factory=dict)   # numbered PNG path -> blob at after
    copy_from: str = ""                         # rename: the old PNG's blob, copied to png
    left: str = ""                              # rename: the old PNG, still at its old path
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
    writes: list = field(default_factory=list)  # {"path", "file", "sha", "kind", "line"}
    error: str = ""

    @property
    def name(self) -> str:
        return card_name(self.job.psd)

    def told(self) -> list:
        """The lines the author is told on Discord."""
        out = []
        if self.export.get("outcome") == "refused":
            out.append(self.export["why"])
        out += [why for why, _ in self.export.get("numbered_refused", [])]
        out += [f"{line} (the bot could not fix it: {why})" for line, why in self.fix_refused]
        if self.r10.get("ok") is False:
            out.append(self.r10["line"])
        out += [issue["line"] for issue in self.issues if issue["told"] and issue["discord"]]
        out += [text for text, discord in self.notes if discord]
        return out


# ========================
# Helper Functions
# ========================
def card_name(path: str) -> str:
    """The card's name from its PSD path (card_models.card_name)."""
    return splitext(basename(path))[0].replace("_", " ")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _key(path: str) -> tuple:
    return dirname(path), splitext(basename(path))[0].lower()


def _is_png(path: str) -> bool:
    return path.lower().endswith(".png")


def _card_png(path: str) -> bool:
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
        """{path: Writer}: the last commit in before..after that changed it (oldest first, the last one wins)."""
        span = after if before == EMPTY_TREE else f"{before}..{after}"
        raw = self.git("log", "--no-renames", "--reverse", "--format=%x1e%H%x1f%an%x1f%ae%x1f%s", "--name-only", "-z", span)
        out = {}
        for chunk in raw.split("\x1e"):
            if not chunk.strip():
                continue
            head, _, rest = chunk.partition("\0")
            sha, name, email, subject = (head.split("\x1f") + ["", "", ""])[:4]
            writer = Writer(sha, name, email, subject)
            for path in rest.lstrip("\n").split("\0"):
                if path.strip():
                    out[path.strip("\n")] = writer
        return out

    def blob(self, sha: str) -> bytes:
        """A blob by sha, verified; raises Refused("blob")."""
        return fetch_verified(self.repository, sha, "", self.clone)

    def login(self, writer: Writer, fallback: str) -> str:
        """The commit author's GitHub login (the commits API, once per commit, when a token and repo are given), else the
        git author name, else fallback."""
        if writer.commit not in self.logins:
            login = ""
            if self.repository and token():
                try:
                    login = ((api(self.repository, "GET", f"commits/{writer.commit}").json().get("author") or {}).get("login") or "")
                except Exception:               # no login: the git name stands
                    login = ""
            self.logins[writer.commit] = login or writer.name or fallback
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
    pngs = {c.path: c for c in changes if _card_png(c.path)}
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

    for change in psds:
        if change.path in renamed_old:
            continue
        png = snapshot.png_for(change.path)[0] if change.status != "D" else ""
        login, commit = who(change.path)
        job = Job(change.path, "", change.old, change.new, login=login, commit=commit)
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
        elif png:
            job.case, job.png, job.png_old, job.numbered = "beside", png, tree[png][1], numbered(change.path)
        elif _key(change.path) in png_deleted:      # the author deleted the PNG: the standards only
            job.case = "psd-only"
            notes.append(f"`{change.path}` was pushed with its PNG deleted: no PNG is made")
        else:
            job.case, job.png = "new", splitext(change.path)[0] + ".png"
        jobs.append(job)
    pushed_psds = {c.path for c in psds}
    psd_by_key = {_key(p): p for p in tree if is_card_psd(p)}
    for path, change in sorted(pngs.items()):
        psd = psd_by_key.get(_key(path))
        if psd is None or psd in pushed_psds:
            if psd is None and change.status in "AM" and not snapshot.duplicate_of(CardAuthors.key(card_name(path))):
                if not any(p for p in pushed_psds if _key(p) == _key(path)):
                    notes.append(f"`{path}` has no PSD beside it")
            continue
        login, commit = who(path)
        if change.status in "AM":
            jobs.append(Job(psd, "png-only", "", tree[psd][1], path, change.old, change.new, login=login, commit=commit))
        elif change.status == "D":
            jobs.append(Job(psd, "png-deleted", "", tree[psd][1], path, change.old, login=login, commit=commit))
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
# Context: Types/, fonts, the status chart, the edits file
# ========================
def build_context(source: Source, tree: dict, fonts_dir: str = "", errors: Optional[list] = None) -> tuple:
    """
    (card_standards.Context, Vocabulary) from the tree at after.

    The status chart and the edits file come by blob; a failed one is an error line and the rules run without it.
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
    ok = set()
    path = next((p for p in EDITS_PATHS if p in tree), None)
    if path:
        try:
            edits = CardEdits.parse(source.blob(tree[path][1]).decode("utf-8"))
            ok = {key for key, fields in edits.cards.items() if fields.get("problems_ok")}
        except Exception as error:              # no per-card waivers
            errors.append(f"{path}: {error}")
    fonts = None
    if fonts_dir:
        fonts = FontIndex(fonts_dir)
        errors += [f"MDW/Fonts {e}" for e in fonts.errors]
    context = standards.Context(frozenset(vocab.all_types), fonts, chart, frozenset(ok), psd_edit.story_sync())
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


def _export_dict(found) -> dict:
    return {"outcome": found.outcome, "why": found.why, "gate": found.gate, "changed_px": found.changed_px,
            "box": list(found.box) if found.box else None, "icc": found.icc, "writer": found.writer,
            "vote": list(found.vote) if found.vote else None, "numbers": found.numbers}


def _r10_line(job: Job, measured: dict) -> str:
    bad = measured.get("bad_pct", 0)
    if job.case == "png-only":
        return f"{basename(job.png)} does not match its PSD ({bad:.1f}% of pixels differ): push the PSD too?"
    return f"{basename(job.png)} does not match the PSD ({bad:.1f}% of pixels differ): export it again"


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
    if job.case == "deleted":
        result.notes.append((f"{basename(job.png)} is left without its PSD", True))
        return
    if job.case == "png-deleted":
        result.notes.append((f"{basename(job.png)} was deleted; the PSD stays", False))
        return
    if job.case == "rename":
        data = source.blob(job.copy_from)
        file = _save(out, job.png, data)
        result.export = {"outcome": "re-exported", "why": "", "copied": True}
        result.writes.append({"path": job.png, "file": file, "sha": blob_sha(data), "kind": "png",
                              "line": "the PNG copied to the PSD's new name"})
        if job.left:
            result.notes.append((f"{basename(job.left)} is left at the PSD's old name", True))
        return
    new_bytes = source.blob(job.psd_new)
    new = png_export.open_psd(new_bytes)
    old = png_export.open_psd(source.blob(job.psd_old)) if job.psd_old else None
    rule_page = standards.kind_of(job.psd) == "rule"
    # guard 1
    if job.case in ("stale", "beside"):
        old_png = png_export.open_png(source.blob(job.png_old))
        numbered = {path: png_export.open_png(source.blob(sha)) for path, sha in job.numbered.items()}
        found = png_export.export(old if job.case == "stale" else None, new, old_png, numbered)
        result.export = _export_dict(found)
        stale = found.numbers.get("png_vs_psd", {})
        if found.outcome == "current" and stale.get("px") and not psd_edit._within_bar(stale):
            result.notes.append((f"{basename(job.png)} does not match the PSD ({stale['bad_pct']:.1f}% of pixels differ), "
                                 "though this push moved no pixel: export it in Photoshop", True))
        if found.outcome == "re-exported":
            file = _save(out, job.png, found.data)
            result.writes.append({"path": job.png, "file": file, "sha": blob_sha(found.data), "kind": "png",
                                  "line": f"re-exported from the PSD's composite ({found.changed_px:,} px changed)"})
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
                                  "line": "made from the PSD's composite (new card)"})
    # R10: a pushed PNG against the PSD
    if job.case in ("pair", "png-only", "whole") and job.png_new:
        png = png_export.open_png(source.blob(job.png_new))
        measured = png_export.png_check(new, png)
        result.r10 = {**measured, "line": _r10_line(job, measured) if measured.get("ok") is False else ""}
        if measured.get("ok") is None:
            result.notes.append((f"{basename(job.png)} could not be checked against the PSD ({measured.get('why')})", False))
    if job.case == "png-only" or rule_page:
        return
    # guard 2
    issues = standards.check(new, job.psd, context, old=old, added=job.case in ("new", "beside"))
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


def _init_worker(clone: str, repository: str, context, vocab) -> None:
    _worker.update(source=Source(clone, repository), context=context, parser=vocab.parser())


def _pooled(item: tuple) -> Result:
    job, out, mode = item
    return run_job(_worker["source"], job, _worker["context"], _worker["parser"], out, mode)


# ========================
# Reports
# ========================
def _expect(job: Job) -> dict:
    """The blobs a card's writes rest on at the head (G9)."""
    expect = {job.psd: job.psd_new}
    if job.png:
        expect[job.png] = job.png_old if job.case in ("stale", "beside") else ""
    for path, sha in job.numbered.items():
        expect[path] = sha
    return expect


def plan(results: list, repository: str, before: str, after: str, mode: str) -> dict:
    """plan.json: the writes per card with the blobs they rest on, and the verdicts."""
    cards = []
    for result in results:
        entry = {"card": result.name, "psd": result.job.psd, "case": result.job.case, "login": result.job.login,
                 "commit": result.job.commit, "export": result.export, "r10": result.r10, "issues": result.issues,
                 "fixed": result.fixed, "fix_refused": result.fix_refused, "notes": [t for t, _ in result.notes],
                 "told": result.told(), "error": result.error, "writes": result.writes}
        if result.writes:
            entry["expect"] = _expect(result.job)
        cards.append(entry)
    return {"repo": repository, "before": before, "after": after, "mode": mode, "checked_at": _now().isoformat(),
            "cards": cards}


def _box(numbers: dict, box) -> str:
    if not box:
        return ""
    x0, y0, x1, y1 = box
    return f" in {x1 - x0}x{y1 - y0} at ({x0}, {y0})"


def _guard1_line(entry: dict) -> str:
    found = entry["export"]
    outcome = found.get("outcome")
    if not outcome:
        return ""
    if outcome == "current":
        return "PNG still current (the push moved no pixel of the composite)"
    if outcome == "refused":
        return found["why"]
    if found.get("copied"):
        return "PNG copied to the PSD's new name"
    numbers = found.get("numbers", {})
    parts = [f"{found['changed_px']:,} px changed{_box(numbers, found.get('box'))}", f"ICC {found.get('icc') or 'as stored'}",
             "saved by Photoshop" if found.get("writer") == "photoshop" else
             f"refereed by the fork (bad {numbers.get('referee', {}).get('bad_pct', 0):.2f}%)"]
    if "outside" in numbers:
        parts.append(f"outside the change: mean {numbers['outside'].get('mean', 0):.2f}")
    verb = "made" if entry["case"] == "new" else "re-exported"
    return f"PNG {verb} from the PSD's composite ({'; '.join(parts)})"


def markdown(data: dict, notes: list, skipped: dict, errors: list) -> str:
    """The job summary and commit comment."""
    mode = data["mode"]
    head = {"dry": "Dry run: nothing is posted or committed.", "report": "Report only (CARD_GUARDS=report): nothing is committed.",
            "on": "Writes land after the 30-minute quiet window, in one bot commit."}[mode]
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
        line = _guard1_line(entry)
        if line:
            out.append(f"- {line}")
        for why, path in entry["export"].get("numbered_refused", []):
            out.append(f"- {why}")
        if entry["r10"].get("ok") is False:
            out.append(f"- {entry['r10']['line']}")
        elif entry["r10"].get("ok"):
            out.append(f"- {basename(entry['psd'])[:-4]}.png matches the PSD (mean {entry['r10'].get('mean', 0):.2f})")
        told = [i["line"] for i in entry["issues"] if i["told"]]
        if told:
            out.append("- Issues: " + "; ".join(told))
        if entry["fixed"]:
            verb = "Fixed by the bot" if mode == "on" else "The bot would fix (CARD_GUARDS=on)"
            out.append(f"- {verb}: " + "; ".join(entry["fixed"]))
        for line, why in entry["fix_refused"]:
            out.append(f"- Not fixed: {line} ({why})")
        waived = [f"{i['code']} {i['line']}" for i in entry["issues"] if i["waived"]]
        if waived:
            out.append("- Waived (owner decisions): " + "; ".join(waived))
        for note in entry["notes"]:
            out.append(f"- {note}")
        if not (line or told or entry["fixed"] or entry["fix_refused"] or entry["notes"] or entry["r10"]):
            out.append("- Meets the standards.")
        out.append("")
    if errors:
        out += ["Errors:"] + [f"- {e}" for e in errors] + [""]
    return "\n".join(out)


def discord_payload(data: dict, repository: str) -> dict:
    """
    The webhook body: one LINE per card with something to tell (a refusal or a remaining issue), whole lines dropped from
    the end to fit DISCORD_LIMIT; {} when nothing. Nobody is pinged and links get no preview.
    """
    url = f"https://github.com/{repository}/commit/"
    lines = [LINE.format(login=entry["login"], card=entry["card"], issues="; ".join(entry["told"]).rstrip("."),
                         short=(entry["commit"] or data["after"])[:8], url=url + (entry["commit"] or data["after"]))
             for entry in data["cards"] if entry["told"]]
    if not lines:
        return {}

    def assemble(kept: int) -> str:
        rest = len(lines) - kept
        more = [f"... and {plural(rest, 'more card')}: [details]({url}{data['after']})"] if rest else []
        return "\n".join(lines[:kept] + more)

    kept = len(lines)
    content = assemble(kept)
    while len(content) > DISCORD_LIMIT and kept > 0:
        kept -= 1
        content = assemble(kept)
    return {"content": content, "allowed_mentions": {"parse": []}, "flags": SUPPRESS_EMBEDS}


# ========================
# check
# ========================
def check(clone: str, after: str, before: str = "", repository: str = "", author: str = NOBODY, cards: Optional[set] = None,
          whole_tree: bool = False, mode: str = "dry", fonts_dir: str = "", out: str = "guards", workers: int = 0) -> dict:
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
    context, vocab = build_context(source, tree, fonts_dir, errors) if jobs else (standards.Context(), None)
    if len(jobs) >= POOL_MIN and workers != 1:
        count = workers or max(1, min((cpu_count() or 2) // 2, len(jobs)))
        with ProcessPoolExecutor(count, initializer=_init_worker, initargs=(clone, repository, context, vocab)) as pool:
            results = list(pool.map(_pooled, [(job, out, mode) for job in jobs], chunksize=2))
    else:
        parser = vocab.parser() if vocab is not None else None
        results = [run_job(source, job, context, parser, out, mode) for job in jobs]
    data = plan(results, repository, base, head, mode)
    errors += [f"{r.job.psd}: {r.error}" for r in results if r.error]
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


def summary(data: dict, errors: list) -> dict:
    """summary.json, the workflow's switches: cards checked, cards with something told, writes planned, errors."""
    return {"after": data["after"], "cards": len(data["cards"]), "told": sum(1 for c in data["cards"] if c["told"]),
            "writes": sum(len(c["writes"]) for c in data["cards"]), "errors": len(errors)}


# ========================
# land
# ========================
def message(landed: list, after: str) -> str:
    """The bot commit's message: subject with the counts, one line per file, the Guarded-push trailer."""
    pngs = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "png")
    psds = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "psd")
    counts = ", ".join(part for part in (plural(pngs, "PNG") + " re-exported" if pngs else "",
                                         plural(psds, "PSD") + " fixed" if psds else "") if part)
    lines = [f"{w['path']}: {w['line']} ({entry['login']}, {(entry['commit'] or after)[:8]})"
             for entry in landed for w in entry["writes"]]
    return f"{SUBJECT}: {counts} (push {after[:8]})\n\n" + "\n".join(lines) + f"\n{TRAILER}: {after}\n"


def land(plan_path: str, repository: str = "", quiet_from: str = "", dry_run: bool = False, local: str = "",
         sleep: Callable[[float], None] = _sleep, now: Callable[[], datetime] = _now) -> dict:
    """
    Commit a plan's writes once (G9): wait until quiet_from + RECENT, leave out a card a later push touched or whose PSD,
    PNG or numbered PNGs no longer hold the blobs it rests on, and commit the rest on the branch head (retried on a
    moved head, COMMIT_TRIES times); --local commits in that clone instead (nothing pushed).

    Returns:
        {"sha": the commit or "", "landed": [card], "dropped": [[card, why]]}; raises Transient when the commit fails
    """
    out_dir = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    repository = repository or data.get("repo") or ("" if local else CARD_REPO)
    entries = [entry for entry in data["cards"] if entry.get("writes")]
    report = {"sha": "", "landed": [], "dropped": []}
    if not entries:
        return report
    if quiet_from:
        mark = datetime.fromisoformat(quiet_from.replace("Z", "+00:00")) + RECENT + QUIET_MARGIN
        wait = (mark - now()).total_seconds()
        if wait > 0:
            print(f"Waiting {wait / 60:.1f} min for the quiet window.", file=stderr)
            sleep(wait)
    files = {}
    for entry in entries:
        for write in entry["writes"]:
            with open(join(out_dir, write["file"]), "rb") as fh:
                files[write["path"]] = fh.read()
            if blob_sha(files[write["path"]]) != write["sha"]:
                raise Transient(f"{write['file']} does not hash to the planned {write['sha'][:8]}")
    uploaded: set = set()
    for attempt in range(COMMIT_TRIES if not local else 1):
        tip = local_snapshot(repository, local) if local else snapshot_at(repository, ref_head(repository))
        recent = recent_paths(repository, tip.head, now(), local) if repository else set()   # a clone alone: no push times
        included, report["dropped"] = [], []
        for entry in entries:
            moved = [p for p, sha in entry["expect"].items() if (tip.tree.get(p, ("", ""))[1]) != sha]
            touched = sorted(set(entry["expect"]) & recent) if recent is not None else ["(too many pushes to tell)"]
            if moved:
                report["dropped"].append([entry["card"], f"superseded: {moved[0]} changed on the branch"])
            elif touched:
                report["dropped"].append([entry["card"], f"superseded: a later push touched {touched[0]}"])
            else:
                included.append(entry)
        report["landed"] = [entry["card"] for entry in included]
        if not included or dry_run:
            return report
        payload = {w["path"]: files[w["path"]] for entry in included for w in entry["writes"]}
        text = message(included, data["after"])
        if local:
            report["sha"] = local_commit(local, tip.head, payload, text)
            return report
        modes = {path: tip.tree[path][0] for path in payload if path in tip.tree}
        sha = commit_files(repository, tip.head, tip.info["tree"]["sha"], modes, payload, text, uploaded)
        if sha:
            report["sha"] = sha
            return report
    raise Transient(f"the branch kept moving: not committed after {COMMIT_TRIES} tries")


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """check: the report on stdout (exit 0 whatever it finds); land: one line per card, exit 1 when the commit failed."""
    cli = ArgumentParser(prog="python -m commands.card_guards",
                         description="The push guards: a forgotten PNG made from the PSD's stored composite, the card standards "
                                     "checked (0-pixel fixes planned). check never writes to the card repo; land commits a plan once.")
    sub = cli.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help="both guards on a push (or the whole tree)")
    one.add_argument("--clone", required=True, metavar="DIR", help="a clone of the card repo with the push's history (blobless is enough)")
    one.add_argument("--after", required=True, metavar="SHA", help="the push's head")
    one.add_argument("--before", default="", metavar="SHA", help="the push's previous head; empty: the first parent")
    one.add_argument("--repo", default="", metavar="OWNER/NAME", help="blobs the clone lacks and the commit logins by API")
    one.add_argument("--author", default=NOBODY, metavar="LOGIN", help="who to name when a login is unknown (github.actor)")
    one.add_argument("--cards", default="", metavar="NAMES", help="only these cards (comma separated, matched like Card Authors.txt)")
    one.add_argument("--whole-tree", action="store_true", help="every card PSD at --after: standards and PNG-vs-PSD")
    one.add_argument("--mode", default="dry", choices=MODES, help="dry, report or on (CARD_GUARDS): the report's wording")
    one.add_argument("--fonts", default="", metavar="DIR", help="MDW/Fonts (R7); skipped without")
    one.add_argument("--workers", type=int, default=0, metavar="N", help="processes for 8 cards or more (0: half the cores)")
    one.add_argument("--out", required=True, metavar="DIR", help="report.md, discord.json, plan.json and the artifacts")
    two = sub.add_parser("land", help="commit a plan's writes once, after the quiet window")
    two.add_argument("--plan", required=True, metavar="FILE", help="check's plan.json (its files beside it)")
    two.add_argument("--repo", default="", metavar="OWNER/NAME", help="the card repo (default: the plan's)")
    two.add_argument("--quiet-from", default="", metavar="ISO", help="the push time: land waits until 30 min after it")
    two.add_argument("--dry-run", action="store_true", help="decide what would land, commit nothing")
    two.add_argument("--local", default="", metavar="DIR", help="commit in this clone (nothing pushed)")
    options = cli.parse_args(args)
    token_from_env()
    if options.command == "check":
        cards = {CardAuthors.key(n) for n in options.cards.split(",") if n.strip()} or None
        try:
            data = check(options.clone, options.after, options.before, options.repo, options.author, cards, options.whole_tree,
                         options.mode, options.fonts, options.out, options.workers)
        except (Transient, Refused) as error:   # the clone or the API failed: one line, the job summary says so
            print(f"## Card guards\n\nNot checked: {error}\n")
            return 0
        print(data["markdown"], end="")
        return 0
    try:
        report = land(options.plan, options.repo, options.quiet_from, options.dry_run, options.local)
    except (Transient, Refused) as error:
        print(f"Not landed: {error}")
        return 1
    with open(join(dirname(options.plan), "landed.json"), "w", encoding="utf-8") as fh:
        dump(report, fh)
    for card in report["landed"]:
        print(f"{card}: {'landed in `' + report['sha'][:8] + '`' if report['sha'] else 'would land'}")
    for card, why in report["dropped"]:
        print(f"{card}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BOT_SUBJECTS',
    'Change',
    'Job',
    'LINE',
    'Result',
    'SPARSE_FILES',
    'Source',
    'Writer',
    'build_context',
    'check',
    'discord_payload',
    'land',
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
