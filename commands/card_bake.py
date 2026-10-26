"""
The edit pipeline's driver (docs/EDIT_PIPELINE_DESIGN.md sections 4, 5, 8-11): bake the PSD-kind edits of
TTSCardMaker's site/Card Edits.json into the card PSDs and PNGs, one commit per run.

    python -m commands.card_bake --repo MichaelJSr/TTSCardMaker --head <sha> [--dry-run] [--card <Name>] [--pending]
                                 [--local <clone>] [--verdicts <db/verdicts.json>]
    run(repository, head, dry_run, card, pending_only, local, verdicts) -> (markdown report, commit shas: one at most)
    pending(repository, head, local, verdicts) -> the cards a live run has work for; due() those past the 30-minute grace

Input: one tree snapshot at head and blobs by sha (the card folders are never checked out). A website-only field (the
entry's `_site`, owner 2026-10-25) is never baked, dropped or given a verdict: it stays in the file. Per card named in the
file with a PSD-kind field: refuse `recent` (PSD or PNG pushed in the last 30 min), fetch its PSD and official PNG (gate
`blob`), classify the edits against a fresh offline parse (psd_edit.classify), apply them with every gate
(psd_edit.apply_jobs; an ability edit also makes its FX layers again, Effect Layers.jsx's Update all: section 7 idea 14,
BUG 2).
Then one commit on the branch's current head (land): every baked card's PSD and PNG and the file rewritten with the
baked and redundant fields removed (a Python port of edit.js editsFile), through git_commit (the GitHub API, or git
objects in the history clone pushed over SSH when the token can't write). A card whose PSD, PNG or edit changed meanwhile
is left out of the commit as a try-again refusal; the others still land. The verdicts live on the data branch
(db/verdicts.json, docs/Future/CARD_SYNC_VERDICTS.md): the run writes them, with its bake events, to the work dir's
verdicts.json and bake-events.json (records()), split into what needs no main commit (a refusal: never one) and what
holds only once the commit lands; Card sync merges them into the data (card_data --verdicts, --events). A verdict is
read from --verdicts, else (one Card sync cycle, the migration) from the entry's old `_psd`.
--local (card-sync.yml's stage 1, docs/Campaigns/DATA_BUILD_ACTION.md 6.3): the tree and blobs come from a local clone
(blobless is enough; a blob it lacks by REST) and the run's commit B is made there on head, pushed by no one here; a
checkout at head moves to B, and the report says where HEAD is. A --head the clone lacks is a failed run (a report,
exit 1, no B). The recent gate diffs the clone's trees when it has the pushes' base (head may be a previous B).
A name edit with `_rename` (the editor's "Also rename the card's files", owner 2026-10-25) also renames the card's PSD and
PNGs in that commit (plan_rename), with Renamed trailers and the files keyed by the card (rename_edits, side_files).
Refusals are data: one bad card never blocks another, and a refused stars or types edit never blocks its card's other
fields (they bake; the refused field stays in the file, its refused verdict on the PSD the commit leaves). A
content problem (text that won't fit, a typed badge or token, a styled word, a font stand-in) never refuses (owner
2026-10-24): it bakes, told in the commit line, the report and the baked verdict's reason (psd_edit's notes). A run
bakes at most MAX_BAKES cards and starts none after BUDGET (the rest are refused `capped` and wait for the next run).
The report goes to stdout, the artifacts (edited PSD, merged PNG, before/after/diff crops, the rewritten file) to the
work dir. The runner supplies the token
(GITHUB_TOKEN, or config.GIT_TOKEN on the bot), the fonts (PSD_TOOLS_FONT_DIRS; else the checkout's MDW/Fonts) and the
work dir (RUNNER_TEMP).
No Discord or config imports; never the pickles; --dry-run commits nothing. GitHub, git and runner failures (a 5xx,
a rate limit, no text engine) are reported but never written as a verdict; such a card error, or a commit that failed,
exits 1 (failure()). A face no file holds is drawn in the fork's stand-in, told (psd_edit.font_map); a `font` verdict
(the stand-in itself missing) holds until MDW/Fonts changes.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from json import dumps, loads
from os import environ, makedirs, pathsep, remove
from os.path import abspath, basename, dirname, exists, expanduser, isdir, join, splitext
from re import IGNORECASE, escape, findall, fullmatch, search
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import stderr
from tempfile import mkdtemp
from time import perf_counter
from traceback import format_exc
from typing import Callable, Optional

from PIL import Image
from psd_tools import PSDImage, __version__ as PSD_TOOLS_VERSION
from requests import RequestException

import commands.github_api as github_api
import commands.png_export as png_export
import commands.psd_edit as psd_edit
from commands.card_edits import ORDER, RENAME, SITE, CardAuthors, CardDates, CardEdits, CardRatings, dump_edits
from commands.card_models import RENAMED, card_name, file_stem
from commands.git_commit import (API_TIMEOUT, AUTHOR, COMMIT_TRIES, GIT_TIMEOUT, NO_LAZY, Transient, api, checkout_head, commit_files,
                                 commit_info, git, local_commit, plural, ref_head)
from commands.github_api import blob_sha, token_from_env
from commands.psd_edit import Refusal, Refused
from commands.psd_parser import shown_in
from global_config import (AUTHORS_PATH, CARD_BRANCH, CARD_REPO, CREATED_PATH, EDITS_PATH, FONT_DIRS_ENV,
                           ICON_MANIFEST_PATH, LOCAL_DIR_LOC, RATINGS_PATH, RULES_PATH, STARS_DIR, TYPES_DIR)

# ========================
# Configuration
# ========================
RECENT = timedelta(minutes=30)                  # a PSD or PNG pushed more recently is its author's: don't race them
PUSHES = 100                                    # activity entries per request (GitHub's max)
COMPARE_FILES = 300                             # a compare lists at most this many files
SUBJECT = "Card edits"                          # the run's commit: "Card edits: 5 baked, 20 redundant, 4 refused"
EDIT_BY = "Edit-by"                             # trailer key per editor credited (TTSCardMaker build_site.EDIT_BY)
NO_PNG = "no PNG beside the PSD"
RULES = "2026-10-26"        # a refused verdict's rules: one written under older ones is tried again (10-24: content
#   refused; 10-26: Lane 2's bars, titles and website-only fields)
MAX_BAKES = 40                                  # PSD edits per run; the rest wait for the next run (a title took ~30 s)
BUDGET = timedelta(minutes=40)                  # no card starts later: the commit lands within the job's 60 min
TRANSIENT = ("recent", "head-moved", "edit-changed", "capped")   # try-again refusals: not a verdict for the editor
EXCLUDE_FOLDERS = ("Markers", "MDW")            # not cards (card_data.EXCLUDE_FOLDERS)
WORK_ENV = "RUNNER_TEMP"                        # the Action's temp dir; the work dir is card-bake/ in it
WORK_NAME = "card-bake"
VERDICTS_OUT = "verdicts.json"                  # in the work dir: {"schema", "cards": no main commit needed, "landed": with it}
EVENTS_OUT = "bake-events.json"                 # in the work dir: {"events": [...], "landed": [...]} (changelog kinds)
FIELDS = frozenset(CardEdits.TYPES)            # ORDER (edit.js FIELDS) and PUNCTUATION live in card_edits
KIND = CardEdits.KIND
BOT_BY = AUTHOR["name"]                         # the bot's _by (changelog.BOT)


# ========================
# Data
# ========================
@dataclass
class Card:
    """One card named in the edits file with a PSD-kind field: what the run did with it."""
    name: str                                   # the PSD stem with spaces
    key: str                                    # CardAuthors.key
    fields: dict                                # its PSD-kind fields and values (the job key, section 5)
    rel: str = ""                               # PSD repo path ("" when the card has none)
    psd_sha: str = ""
    png: str = ""                               # official PNG repo path
    png_sha: str = ""
    jobs: dict = field(default_factory=dict)
    redundant: list = field(default_factory=list)
    refusal: Optional[Refusal] = None
    transient: bool = False                     # the refusal is a runner or GitHub failure: no verdict for the editor
    baked: Optional[psd_edit.Baked] = None
    new_psd_sha: str = ""
    files: dict = field(default_factory=dict)   # {repo path: bytes}: the baked PSD and PNG for the commit
    expect: dict = field(default_factory=dict)  # {repo path: blob sha}: what its PSD and PNG must still be at the commit
    commit: Optional[str] = None                # the run's commit, when the card is in it
    included: bool = False                      # in the commit land made (or would make: a dry run)
    dropped: list = field(default_factory=list)     # fields removed from the file
    creatures: bool = False                     # the entry has a creatures list: a baked stat is creature 1's (its rows)
    shared: dict = field(default_factory=dict)  # creature n (2...) -> {stat: value} a creature 1 bake changed too (a
    #   shared bar: Pix's HP, Twin Emperors' shared health), for its creatures list
    listed: list = field(default_factory=list)  # the entry's creatures list: a value set apart from the print stays
    title_anchor: int = 0                       # px an earlier bake of this PSD moved its title on (R4, its verdict)
    site: dict = field(default_factory=dict)    # field -> why: website only now, the card image can't print it (classify)
    by: object = None                           # the entry's _by: who asked for its fields (Edit-by trailers)
    verdict: Optional[dict] = None              # its verdict (baked or refused), for the data's verdicts.json
    partial: list = field(default_factory=list)  # Refusals of its stars or types edit while its other fields bake
    fonts: str = ""                             # a `font` refusal's Snapshot.fonts_key: final only while MDW/Fonts holds
    pngs: str = ""                              # a PNG refusal's Snapshot.pngs_key: final only while the PNGs hold
    note: str = ""
    traceback: str = ""
    seconds: float = 0.0
    parts: Optional[psd_edit.PartsPlan] = None  # its `parts` sorted (plan_parts): outcomes per layer, skipped ones
    part_drop: list = field(default_factory=list)   # part keys the commit takes out of the entry
    pictures: dict = field(default_factory=dict)    # a PSD with variant tags: picture n -> its PNG repo path (§9)
    view: Optional[int] = None                  # bake this picture only (a preview of a variant's page)
    views: dict = field(default_factory=dict)   # picture n -> its Baked (the saved picture's is `baked`)
    moves: dict = field(default_factory=dict)   # a stars edit's move into its "N Stars" folder: old path -> new path
    moved_to: str = ""                          # that folder
    move_notes: list = field(default_factory=list)  # the folder created, the printed stars not redrawn
    rename: bool = False                        # the entry's `_rename` (CardEdits.renames): its files take its name
    renames: dict = field(default_factory=dict) # that rename: old path -> new file name (plan_rename; after any move)
    renamed: str = ""                           # the card's new name
    relocated: dict = field(default_factory=dict)   # the commit's moves and renames: old path -> new path


@dataclass
class Report:
    """The run's head lines and cards, rendered in tts_save._render style."""
    repository: str
    head: str
    dry_run: bool
    work: str
    lines: list = field(default_factory=list)   # head lines after the repo line
    note: str = ""                              # why the run stopped early or committed nothing, if so
    cards: list = field(default_factory=list)
    commit: Optional[str] = None                # the run's one commit
    subject: str = ""                           # its subject (a dry run's: what it would be)
    local: str = ""                             # --local's clone: the commit is made there, not pushed
    traceback: str = ""                         # when the commit failed outside a card's steps
    commit_failed: bool = False                 # land raised: nothing committed although cards had shares
    error: str = ""                             # the run stopped before any card (--local: a --head the clone lacks)
    checkout: Optional[str] = None              # --local, after B: the clone's HEAD ("" bare or unborn, "?" unreadable)
    text: str = ""                              # the rendered report
    seconds: float = 0.0


@dataclass
class Share:
    """One card's part of the run's commit (land): its files, the blobs they rely on, its change to the edits file."""
    card: Card
    files: dict                                 # {repo path: bytes} ({} for a file-only share)
    expect: dict                                # {repo path: blob sha} these paths must still hold at H'
    rewrite: Callable[[str], Optional[str]]     # edits file text -> new text, None when unchanged; raises Refused to stay out
    out: Optional[Refusal] = None               # why the last attempt left it out
    included: bool = False                      # in the last attempt's commit
    side: dict = field(default_factory=dict)    # {repo path: text -> new text or None}: other files keyed by the card (a rename)


class Snapshot:
    """The card repo at one commit: the tree, the PSDs by card key and where the edits file is."""

    def __init__(self, repository: str, head: str, info: dict, tree: dict, local: str = ""):
        self.repository, self.head, self.info, self.tree, self.local = repository, head, info, tree, local
        self.lower = {path.lower() for path in tree}
        self.psds: dict = {}
        for path in tree:
            if path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/")):
                self.psds[CardAuthors.key(card_name(path))] = path
        self.edits_path = EDITS_PATH if EDITS_PATH in tree else None

    def sha(self, path: str) -> str:
        """The blob sha of a path in the tree."""
        return self.tree[path][1]

    def fetch(self, sha: str, path: str = "") -> bytes:
        """fetch_verified, from the local clone first."""
        return fetch_verified(self.repository, sha, path, self.local)

    def fonts_key(self) -> str:
        """MDW/Fonts' top-level files (what the Action installs) as 8 hex digits: a `font` verdict holds while they do."""
        prefix = psd_edit.FONTS_DIR + "/"
        entries = sorted(f"{path} {sha}" for path, (_, sha) in self.tree.items()
                         if path.startswith(prefix) and "/" not in path[len(prefix):])
        return sha1("\n".join(entries).encode("utf-8")).hexdigest()[:8]

    def pngs_key(self, rel: str, key: str) -> str:
        """The PNGs a `duplicate` or no-PNG refusal rests on, by path, as 8 hex digits: the verdict holds while they do."""
        if rel:
            png, duplicates = self.png_for(rel)
            paths = [png, *duplicates]
        else:
            paths = [self.duplicate_of(key)]
        return sha1("\n".join(paths).encode("utf-8")).hexdigest()[:8]

    def duplicate_of(self, key: str) -> str:
        """The PSD a numbered PNG keyed like this card was exported from (A_Snack2.png -> A_Snack.psd), or ""."""
        for path in self.tree:
            if path.lower().endswith(".png") and CardAuthors.key(card_name(path)) == key:
                number = fullmatch(r"(.+?)(\d+)", path[:len(path) - 4])
                if number and (number.group(1) + ".psd").lower() in self.lower:
                    return number.group(1) + ".psd"
        return ""

    def png_for(self, rel: str) -> tuple:
        """
        The card's official PNG and the numbered duplicates exported from its PSD.

        Args:
            rel: PSD repo path

        Returns:
            (PNG path with the tree's case, or "", sorted duplicate PNG paths such as A_Snack2.png)
        """
        stem = rel[:len(rel) - 4]
        folder, name = dirname(rel), basename(stem)
        png, duplicates = "", []
        for path in self.tree:
            if dirname(path) != folder or not path.lower().endswith(".png"):
                continue
            base = basename(path)[:len(basename(path)) - 4]
            if base.lower() == name.lower():
                png = path
            else:
                number = fullmatch(escape(name) + r"(\d+)", base, IGNORECASE)
                if number and (stem + number.group(1) + ".psd").lower() not in self.lower:
                    duplicates.append(path)
        return png, sorted(duplicates)


# ========================
# GitHub
# ========================
def _tree(repository: str, tree_sha: str) -> dict:
    """{path: (mode, blob sha)} of a whole tree."""
    data = api(repository, "GET", f"git/trees/{tree_sha}?recursive=1").json()
    if data.get("truncated"):
        raise Transient(f"tree {tree_sha[:8]} is truncated")
    return {entry["path"]: (entry.get("mode", "100644"), entry["sha"]) for entry in data.get("tree", [])
            if entry.get("type") == "blob"}


def snapshot_at(repository: str, head: str) -> Snapshot:
    """The card repo at a commit (two API reads: the commit and its tree)."""
    info = commit_info(repository, head)
    return Snapshot(repository, head, info, _tree(repository, info["tree"]["sha"]))


def _git_local(clone: str, *args: str) -> str:
    """git in a local clone, never a lazy fetch; raises Transient."""
    try:
        return git(clone, *args, env={**environ, **NO_LAZY}).decode("utf-8")
    except CalledProcessError as error:
        raise Transient(f"git {error.cmd[3]} in {clone} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Transient(f"git in {clone} failed: {error}") from error


def _local_blob(clone: str, sha: str) -> Optional[bytes]:
    """A blob from a clone's object store, None when it lacks it (a blobless clone)."""
    try:
        done = run_process(["git", "-C", clone, "cat-file", "blob", sha], capture_output=True, timeout=GIT_TIMEOUT,
                           env={**environ, **NO_LAZY})
    except (TimeoutExpired, OSError):
        return None
    return done.stdout if done.returncode == 0 else None


def _local_commit_exists(clone: str, sha: str) -> bool:
    try:
        return run_process(["git", "-C", clone, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True,
                           timeout=GIT_TIMEOUT, env={**environ, **NO_LAZY}).returncode == 0
    except (TimeoutExpired, OSError):
        return False


def local_snapshot(repository: str, clone: str, head: str = "") -> Snapshot:
    """The card repo at a commit of a local clone (head "" = its HEAD): git ls-tree, no API, never truncated."""
    head = _git_local(clone, "rev-parse", "--verify", f"{head or 'HEAD'}^{{commit}}").strip()
    tree = {}
    for line in _git_local(clone, "ls-tree", "-r", "-z", head).split("\0"):
        meta, _, path = line.partition("\t")
        if path and meta.split(" ")[1] == "blob":
            tree[path] = (meta.split(" ")[0], meta.split(" ")[2])
    info = {"sha": head, "tree": {"sha": _git_local(clone, "rev-parse", f"{head}^{{tree}}").strip()}}
    return Snapshot(repository, head, info, tree, clone)


def fetch_verified(repository: str, sha: str, path: str = "", local: str = "") -> bytes:
    """
    A blob's bytes by sha, checked against the tree (gate blob) and written to path when given.

    Args:
        repository: owner/name of the card repo
        sha: The blob sha the tree lists
        path: Where to write the bytes ("" keeps them in memory only)
        local: A clone whose object store is read first (never a lazy fetch)

    Returns:
        The bytes; raises Refused("blob") when they don't hash to sha
    """
    data = _local_blob(local, sha) if local else None
    if data is not None and blob_sha(data) != sha:
        raise Refused("blob", f"blob {sha[:8]} hashes to {blob_sha(data)[:8]} in {local}")
    if data is None:
        try:
            data = github_api.fetch_verified(repository, sha, timeout=API_TIMEOUT)
        except ValueError as error:
            raise Refused("blob", str(error)) from error
    if path:
        makedirs(dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
    return data


def recent_paths(repository: str, head: str, now: datetime, local: str = "") -> Optional[set]:
    """
    The paths that pushes within RECENT of now changed, up to head: by push time, not commit date (two requests at most).

    Args:
        local: --local's clone: the paths from its trees when it has the window's base (head may be a commit GitHub
            lacks, a previous B), else by the compare API

    Returns:
        The paths, or None when there were too many pushes or files to tell, or no compare (a force push)
    """
    since = (now - RECENT).strftime("%Y-%m-%dT%H:%M:%SZ")
    pushes = api(repository, "GET", "activity", params={"ref": CARD_BRANCH, "time_period": "day", "per_page": PUSHES}).json()
    window = [push for push in pushes if (push.get("timestamp") or "") >= since]
    if not window:
        return set()
    base = min(window, key=lambda push: push["timestamp"]).get("before") or ""
    if len(window) >= PUSHES or not base.strip("0"):  # a new branch has no before
        return None
    if local and _local_commit_exists(local, base):
        return {p for p in _git_local(local, "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}").split("\0") if p}
    reply = api(repository, "GET", f"compare/{base}...{head}", check=False, params={"per_page": 1})
    if not reply.ok:                            # 404/422 after a force push: can't tell
        return None
    files = reply.json().get("files") or []
    if len(files) >= COMPARE_FILES:
        return None
    return {f["filename"] for f in files} | {f["previous_filename"] for f in files if f.get("previous_filename")}


# ========================
# Card Edits.json (the edit.js editsFile port: dump_edits is card_edits')
# ========================
def rewrite_entry(entry: dict, drop=(), psd: Optional[dict] = None, keep: bool = False,
                  part_drop=(), creature=None, site=()) -> Optional[dict]:
    """
    One card's entry as the pipeline leaves it: the dropped fields gone, "_psd" replaced (None removes it), fields
    in ORDER, then the other keys in their order ("_psd" where it was, else last), "_by", "_at" (edit.js editsFile's
    layout; _by/_at are kept as they are).

    Args:
        entry: The card's entry as parsed from the file
        drop: Fields to remove
        psd: The "_psd" verdict to write, None to remove it
        keep: When no field is left, keep the verdict alone (a bake's flags, shown on the card page)
        part_drop: Keys of `parts` to remove (baked, redundant, skipped, deleted: PARTS_EDITOR.md §4.11); a derived
            name or ability goes with the last part of its field, `parts` with its last key
        creature: [{stat: value} per creature] the card now prints (a dict: creature 1's): its entries in the
            `creatures` list take them (the card's stats are creature 1's, a bar it shares may change another's; a
            dropped stat would leave the list's old value showing; owner 2026-10-26)
        site: Fields that are website only from now on (classify's site: the card image can't print them), added to
            `_site` (CardEdits.SITE_FIELDS' order)

    Returns:
        The entry, or None when no field is left and no kept verdict: the entry goes whole, its _-keys with it
        (section 5 step 10)
    """
    entry, drop = dict(entry), list(drop)
    if isinstance(creature, dict):
        creature = [creature]
    if creature and any(creature) and isinstance(entry.get("creatures"), list) \
            and all(isinstance(one, dict) for one in entry["creatures"]):
        entry["creatures"] = [{**one, **(creature[n] if n < len(creature) else {})} for n, one in enumerate(entry["creatures"])]
    if site:
        had = entry.get(SITE) if isinstance(entry.get(SITE), list) else []
        entry[SITE] = [f for f in CardEdits.SITE_FIELDS if f in set(had) | set(site)]
    if part_drop and isinstance(entry.get("parts"), dict):
        fields_of = lambda parts: {p.get("field") for p in parts.values() if isinstance(p, dict)}
        left = {k: v for k, v in entry["parts"].items() if k not in part_drop}
        drop += sorted(fields_of(entry["parts"]) - fields_of(left)) + ([] if left else ["parts"])
        entry["parts"] = left
    current = {k: v for k, v in entry.items() if k not in drop and k not in ("_by", "_at")}
    fields = [f for f in ORDER if f in current]
    if not fields:
        return {"_psd": dict(psd)} if keep and psd is not None else None
    out = {f: current[f] for f in fields}
    others = [k for k in current if k not in FIELDS]    # their order, "_psd" where it was
    if psd is not None and "_psd" not in others:
        others.append("_psd")
    for key in others:
        if key != "_psd":
            out[key] = current[key]
        elif psd is not None:
            out["_psd"] = dict(psd)
    for key in ("_by", "_at"):
        if key in entry:
            out[key] = entry[key]
    return out


def rewrite_edits(text: str, name: str, drop=(), psd: Optional[dict] = None, keep: bool = False,
                  part_drop=(), creature=None, site=()) -> Optional[str]:
    """
    Card Edits.json with one card's entry rewritten (matched like edit.js: keys compared as CardAuthors.key).

    Args:
        text: The file
        name: Card name
        drop: Fields to remove
        psd: The "_psd" verdict to write, None to remove it
        keep: rewrite_entry's
        part_drop: rewrite_entry's

    Returns:
        The new text (dump_edits), or None when the entry is absent or unchanged: a file is never reformatted alone
    """
    data = loads(text)
    cards = data.get("cards")
    if not isinstance(cards, dict):
        return None
    key = next((k for k in cards if CardAuthors.key(k) == CardAuthors.key(name)), None)
    if key is None or not isinstance(cards[key], dict):
        return None
    entry = rewrite_entry(cards[key], drop, psd, keep, part_drop, creature, site)
    if entry == cards[key]:
        return None
    if entry is None:
        del cards[key]
    else:
        cards[key] = entry
    return dump_edits(data)


def psd_fields(fields: dict, site=()) -> dict:
    """The PSD-kind fields of an entry with a value (the job key, section 5), but its website-only ones (`_site`, and
    their part edits): never baked."""
    out = {k: v for k, v in fields.items() if KIND.get(k) == "psd" and v is not None and k not in site}
    if site and isinstance(out.get("parts"), dict):
        out["parts"] = {k: p for k, p in out["parts"].items() if not isinstance(p, dict) or p.get("field") not in site}
        if not out["parts"]:
            del out["parts"]
    return out


def _wanted_at(text: str, name: str) -> dict:
    """A card's fields to bake in a file text (psd_fields of its entry), {} when it has none."""
    edits, key = CardEdits.parse(text), CardAuthors.key(name)
    return psd_fields(edits.cards.get(key) or {}, edits.site(key))


def _same_verdict(a: Optional[dict], b: Optional[dict]) -> bool:
    strip = lambda v: {k: x for k, x in (v or {}).items() if k != "at"}
    return strip(a) == strip(b)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ========================
# Commits (steps 10-11)
# ========================
def land(repository: str, shares: list, message: Callable[[list], str], dry_run: bool, out_path: str = "",
         local: Optional[Snapshot] = None) -> Optional[str]:
    """
    Step 10 for the whole run: one commit on the branch head H' with every share that still holds there. Re-read H'
    and its tree; a share whose paths no longer hold the blobs it read stays out (head-moved), as does one whose rewrite
    raises (edit-changed); the others' rewrites apply in turn to the edits file at H'. On 409/422 or a rejected push,
    all again from the new H', up to COMMIT_TRIES times.

    Args:
        shares: [Share]; each attempt sets its `out` and `included`
        message: The included shares -> commit message (subject, then any trailers)
        dry_run: Write out_path, commit nothing
        out_path: Where the rewritten file goes for the artifacts
        local: --local's snapshot: H' is its head and the commit is made in its clone (local_commit), one attempt

    Returns:
        The commit's sha, None when nothing was committed (dry run, or no share had anything to write);
        raises Refused("head-moved") after COMMIT_TRIES or when H' has no edits file
    """
    uploaded: set = set()                       # blob shas uploaded this run: a retry re-uploads only the rewritten file
    for attempt in range(COMMIT_TRIES if local is None else 1):
        tip = local if local is not None else snapshot_at(repository, ref_head(repository))
        head, tree = tip.head, tip.tree
        edits_path = EDITS_PATH if EDITS_PATH in tree else None
        if edits_path is None:
            raise Refused("head-moved", f"no {EDITS_PATH} on {CARD_BRANCH} `{head[:8]}`")
        text = tip.fetch(tree[edits_path][1]).decode("utf-8")
        new_text, files = text, {}
        for share in shares:
            share.out, share.included = None, False
            moved = [path for path, sha in share.expect.items() if tree.get(path, ("", ""))[1] != sha]
            if moved:
                share.out = Refusal("head-moved", f"{moved[0]} changed on {CARD_BRANCH} `{head[:8]}` meanwhile")
                continue
            try:
                rewritten = share.rewrite(new_text)
            except Refused as refused:
                share.out = refused.refusal
                continue
            if rewritten is None and not share.files:
                continue                        # nothing of its own to write (an unchanged verdict)
            new_text = new_text if rewritten is None else rewritten
            files.update(share.files)
            share.included = True
        included = [share for share in shares if share.included]
        if not included:
            return None
        sides = side_files(tip, included)       # a rename's other files keyed by the card
        if out_path:
            makedirs(dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
            for path, data in sides.items():
                with open(join(dirname(out_path), basename(path)), "wb") as fh:
                    fh.write(data)
        if dry_run:
            return None
        payload = {**files, **({edits_path: new_text.encode("utf-8")} if new_text != text else {}), **sides}
        if not payload:
            return None                         # the rewrites gave back the files as they are
        if local is not None:
            return local_commit(local.local, head, payload, message(included))
        modes = {path: tree[path][0] for path in payload if path in tree}
        commit = commit_files(repository, head, tip.info["tree"]["sha"], modes, payload, message(included), uploaded)
        if commit:
            return commit
    raise Refused("head-moved", f"{CARD_BRANCH} kept moving: not committed after {COMMIT_TRIES} tries")


# ========================
# Per card (section 5)
# ========================
def _work_dir() -> str:
    base = environ.get(WORK_ENV)
    if base:
        path = join(base, WORK_NAME)
        makedirs(path, exist_ok=True)
        return path
    return mkdtemp(prefix=f"{WORK_NAME}_")


def _fonts_dir(snapshot: Optional[Snapshot] = None) -> str:
    """FONT_DIRS_ENV's first dir, else the checkout's MDW/Fonts."""
    dirs = environ.get(FONT_DIRS_ENV, "")
    if dirs:
        return dirs.split(pathsep)[0]
    return join(snapshot.local if snapshot is not None and snapshot.local else expanduser(LOCAL_DIR_LOC), psd_edit.FONTS_DIR)


def vocabulary(snapshot: Snapshot):
    """The icon vocabulary from the tree (Types/, Types/Stars/, the stoogedings manifest), as card_data builds it."""
    all_types = [splitext(basename(p))[0].lower() for p in snapshot.tree
                 if p.startswith(TYPES_DIR) and "Stars" not in p and "." in p]
    stars = [splitext(basename(p))[0].lower().replace(" ", "_") for p in snapshot.tree
             if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
    manifest = None
    if ICON_MANIFEST_PATH in snapshot.tree:
        manifest = loads(snapshot.fetch(snapshot.sha(ICON_MANIFEST_PATH)).decode("utf-8"))
    return psd_edit.offline_parser(all_types, stars, manifest)


def types_icons(snapshot: Snapshot, work: str, parser, names: list, printed: Optional[list] = None) -> str:
    """
    The Types/ PNGs a types edit may make icons from, fetched into the work dir (once a run): the types it adds to
    the ones the card prints.

    Returns:
        The folder (psd_edit.apply_jobs' types_dir)
    """
    folder = join(work, TYPES_DIR)
    wanted = set(Counter(psd_edit.canonical_type(parser, name) for name in names)
                 - Counter(psd_edit.canonical_type(parser, name) for name in printed or []))
    for path, (_, sha) in snapshot.tree.items():
        if (dirname(path) == TYPES_DIR and path.lower().endswith(".png")
                and psd_edit.canonical_type(parser, splitext(basename(path))[0]) in wanted
                and not exists(join(folder, basename(path)))):
            snapshot.fetch(sha, join(folder, basename(path)))
    return folder


def select(edits: CardEdits, snapshot: Snapshot, only: Optional[str] = None) -> list:
    """
    The cards the run looks at: every entry with a PSD-kind field not website only (one card with --card).

    Args:
        edits: The parsed edits file
        snapshot: The tree the cards' PSDs are looked up in
        only: A card name (matched like the authors file), None for every card

    Returns:
        [Card] in the file's order; a card without a PSD keeps rel ""
    """
    cards = []
    for key, fields in edits.cards.items():
        if only is not None and key != CardAuthors.key(only):
            continue
        wanted = psd_fields(fields, edits.site(key))
        if not wanted:
            continue
        rel = snapshot.psds.get(key, "")
        if not rel and derived_picture(edits, snapshot, key, wanted):
            continue                            # its PSD's entry holds the parts: this picture's name and ability follow
        name = card_name(rel) if rel else " ".join(word[:1].upper() + word[1:] for word in key.split())
        cards.append(Card(name, key, wanted, rel, snapshot.sha(rel) if rel else "", creatures=bool(fields.get("creatures")),
                          listed=list(fields["creatures"]) if isinstance(fields.get("creatures"), list) else [],
                          by=edits.meta.get(key, {}).get("_by")))
        cards[len(cards) - 1].rename = edits.renames(key)
    return cards


def derived_picture(edits: CardEdits, snapshot: Snapshot, key: str, wanted: dict) -> bool:
    """A numbered PNG's entry (A Snack2) holding only the derived name or ability of its PSD's part edits (PARTS_EDITOR.md
    §3 Variants): the PSD's bake drops them with the parts; never baked on their own."""
    original = snapshot.duplicate_of(key)
    return bool(original) and set(wanted) <= {"name", "ability"} \
        and bool((edits.cards.get(CardAuthors.key(card_name(original))) or {}).get("parts"))


def plan_move(snapshot: Snapshot, card: Card, parsed: dict, psd, parser) -> dict:
    """
    A stars edit on a card starred by its folder (PARTS_EDITOR.md §4.8, owner 10-25): the card's PSD and every PNG of it
    move into the matching "N Stars" folder (an existing one in any case, else created and told); its printed stars
    are redrawn where the PSD draws them in a layout the bake knows (else told, the move still made). A file of the
    same name there is impossible: refused. Fills card.moves, moved_to, move_notes.

    Returns:
        The jobs apply_jobs bakes (the stars only when they are redrawn)
    """
    jobs = dict(card.jobs)
    target = psd_edit.star_target(card.rel, jobs["stars"]) if "stars" in jobs else None
    if not target:
        return jobs
    low = target.lower() + "/"
    folder = next((path[:len(low) - 1] for path in snapshot.tree if path.lower().startswith(low)), target)
    pngs = [card.png] + [png for n, png in sorted(card.pictures.items()) if n != 1]
    moves = {path: f"{folder}/{basename(path)}" for path in [card.rel] + pngs}
    taken = [new for new in moves.values() if new.lower() in snapshot.lower]
    if taken:
        raise Refused("unsupported-card", f"stars: {taken[0]} is already there: the card can't move into {folder!r}", "stars")
    card.moves, card.moved_to = moves, folder
    if folder == target and not any(path.lower().startswith(low) for path in snapshot.tree):
        card.move_notes.append(f"created the folder {folder!r}")
    try:
        psd_edit.plan_stars(psd, parsed, jobs["stars"], parser, moving=True)
    except Refused as refused:                  # the move stands; the printed stars stay as they are, told
        card.move_notes.append(f"the printed stars are not redrawn: {refused.why.removeprefix('stars: ')}")
        del jobs["stars"]
    return jobs


# ========================
# Renaming a card's files (the editor's "Also rename the card's files", owner 2026-10-25)
# ========================
def plan_rename(snapshot: Snapshot, card: Card, claimed: frozenset = frozenset()) -> None:
    """
    A name edit with `_rename`: the card's PSD and every PNG of it (its pictures NameN.png too) take the name's file stem
    (card_models.file_stem), in the folder a stars move takes them to. Impossible, refused (the name's alone, as a
    stars move's): a name with nothing a file name can hold, another card's file of that name (any case), a name another
    card of the run renames to (claimed: their card keys). Fills card.renames and card.renamed; nothing when the stem is
    the file's already.
    """
    name = card.fields.get("name")
    if not card.rename or not isinstance(name, str) or not ({"name", "parts"} & set(card.jobs) or "name" in card.redundant):
        return
    stem = file_stem(name)
    if not stem:
        raise Refused("unsupported-card", f"name: {name!r} has nothing a file name can hold: the files can't take it", "name")
    old_stem = basename(card.rel)[:len(basename(card.rel)) - 4]
    if stem == old_stem:
        return
    files = {card.rel: stem + card.rel[len(card.rel) - 4:]}
    for n, png in [(1, card.png)] * bool(card.png) + [(n, p) for n, p in sorted(card.pictures.items()) if n != 1]:
        files[png] = stem + ("" if n == 1 else str(n)) + png[len(png) - 4:]
    keys = {CardAuthors.key(card_name(new)) for new in files.values()}
    if keys & claimed:
        raise Refused("unsupported-card", f"name: another card of this run takes the name {name!r}: the files can't take it", "name")
    for path in snapshot.tree:
        if path in files or not path.lower().endswith((".psd", ".png")) or path.startswith(TYPES_DIR + "/") \
                or any(folder in EXCLUDE_FOLDERS for folder in path.split("/")):
            continue
        if CardAuthors.key(card_name(path)) in keys:
            raise Refused("unsupported-card", f"name: {path} already has that name: the files can't take it", "name")
    card.renames, card.renamed = files, card_name(stem)


def _moved_to(card: Card, old: str) -> str:
    """Where one of the card's files goes: its stars move's folder, its rename's file name."""
    path = card.moves.get(old, old)
    folder = path[:path.rfind("/") + 1]
    return folder + card.renames.get(old, path[len(folder):])


def _renamed_names(card: Card) -> list:
    """[(old card name, new card name)] of a rename: the card's, then its pictures' (A Snack2 -> Snack Attack2)."""
    pairs = {card_name(old): card_name(new) for old, new in card.relocated.items() if card.renames}
    return [(old, new) for old, new in pairs.items() if old != new]


def _keyed(names: dict, key: str) -> Optional[str]:
    """The key of a dict naming that card (CardAuthors.key), or None."""
    return next((k for k in names if CardAuthors.key(k) == key), None)


def rename_edits(text: str, card: Card) -> Optional[str]:
    """
    Card Edits.json after a card's files were renamed: its entry (and its pictures') under the new name, `_rename` gone,
    the old name an alias (so /query and the site's search still find it; the bot's field under another's _by: _bot).
    No rename (the files had the name already): only `_rename` goes, once the name left the entry.

    Returns:
        The new text, or None when unchanged
    """
    data = loads(text)
    cards = data.get("cards")
    if not isinstance(cards, dict):
        return None
    before = dump_edits(data)
    pairs = _renamed_names(card)
    if not pairs:
        k = _keyed(cards, card.key)
        if k is not None and isinstance(cards[k], dict) and "name" not in cards[k]:
            cards[k] = {f: v for f, v in cards[k].items() if f != RENAME}
        out = dump_edits(data)
        return out if out != before else None
    for old, new in pairs:
        k = _keyed(cards, CardAuthors.key(old))
        entry = dict(cards.pop(k)) if k is not None and isinstance(cards[k], dict) else None
        if CardAuthors.key(old) == card.key:
            entry = entry if entry is not None else {}
            entry.pop(RENAME, None)
            aliases = list(entry.get("aliases") or [])
            if old.lower() not in (a.lower() for a in aliases) and old.lower() != new.lower():
                if "aliases" not in entry and entry.get("_by") not in (None, BOT_BY):
                    entry["_bot"] = list(entry.get("_bot") or []) + ["aliases"]
                entry["aliases"] = aliases + [old]
            entry.setdefault("_by", BOT_BY)
            entry.setdefault("_at", _now())
        if entry is None:
            continue
        there = _keyed(cards, CardAuthors.key(new))
        if there is not None:                   # an entry under the new name already: the card's fields win
            entry = {**cards.pop(there), **entry}
        entry = rewrite_entry(entry, psd=entry.get("_psd"))
        if entry is not None:
            cards[new] = entry
    out = dump_edits(data)
    return out if out != before else None


def rename_authors(text: str, pairs: list) -> Optional[str]:
    """Card Authors.txt with the renamed cards' [cards] lines under their new names (the line stays where it is)."""
    lines, section, changed = text.split("\n"), None, False
    new = dict((CardAuthors.key(old), name) for old, name in pairs)
    for i, line in enumerate(lines):
        bare = line.strip()
        if bare.startswith("[") and bare.endswith("]"):
            section = bare.strip("[]").strip().lower()
            continue
        left, sep, right = line.partition("=")
        if section == "cards" and sep and not bare.startswith("#") and CardAuthors.key(left) in new:
            lines[i] = new[CardAuthors.key(left)] + " =" + right
            changed = True
    return "\n".join(lines) if changed else None


def _rename_keys(text: str, pairs: list, merge: Callable) -> Optional[str]:
    """A {"cards": {name: value}} file (dump_edits' layout) with the renamed cards' keys moved; merge(theirs, ours) when
    the new name had one."""
    data = loads(text)
    cards = data.get("cards")
    if not isinstance(cards, dict):
        return None
    changed = False
    for old, new in pairs:
        k = _keyed(cards, CardAuthors.key(old))
        if k is None:
            continue
        value = cards.pop(k)
        there = _keyed(cards, CardAuthors.key(new))
        cards[new] = merge(cards.pop(there), value) if there is not None else value
        changed = True
    return dump_edits(data) if changed else None


def _earlier(theirs, ours):
    """Creation dates: a hand entry, else the earlier day."""
    pick = [e for e in (theirs, ours) if isinstance(e, dict) and e.get("from") == "hand"] or \
        sorted((e for e in (theirs, ours) if isinstance(e, dict)), key=lambda e: str(e.get("date")))
    return pick[0] if pick else ours


def rename_paths(text: str, paths: dict) -> Optional[str]:
    """TTS/decks.json with the renamed files' paths (JSON strings, the file's own spelling) replaced."""
    out = text
    for old, new in paths.items():
        out = out.replace(dumps(old, ensure_ascii=False), dumps(new, ensure_ascii=False))
    return out if out != text else None


def _rename_share(card: Card, share: Share) -> Share:
    """A renamed card's share: the edits file's rename after its own rewrite (the rename unticked meanwhile: left out,
    edit-changed), and the other files keyed by the card (side): Card Authors.txt, Card Creation Dates.json, Card
    Ratings.json, TTS/decks.json. The TTS save follows at the Card sync's save step (tts_save, the Renamed trailers)."""
    if not card.rename or not (card.relocated and card.renames or "name" in card.dropped):
        return share
    own = share.rewrite

    def rewrite(text: str) -> Optional[str]:
        if card.renames and not CardEdits.parse(text).renames(card.key):
            raise Refused("edit-changed", f"the rename changed on {CARD_BRANCH} meanwhile: not touched")
        out = own(text)
        return rename_edits(out if out is not None else text, card) or out
    pairs = _renamed_names(card)
    share.rewrite = rewrite
    if pairs:
        share.side = {AUTHORS_PATH: lambda text: rename_authors(text, pairs),
                      CREATED_PATH: lambda text: _rename_keys(text, pairs, _earlier),
                      RATINGS_PATH: lambda text: _rename_keys(text, pairs, lambda a, b: CardRatings.order(list(a) + list(b))),
                      RULES_PATH: lambda text: rename_paths(text, card.relocated)}
    return share


def side_files(tip: Snapshot, shares: list) -> dict:
    """The other files the included shares rewrite (side), at H': {path: bytes} of the ones that changed."""
    out = {}
    for path in sorted({path for share in shares for path in share.side}):
        if path not in tip.tree:
            continue
        text = new = tip.fetch(tip.tree[path][1]).decode("utf-8")
        for share in shares:
            if path in share.side:
                new = share.side[path](new) or new
        if new != text:
            out[path] = new.encode("utf-8")
    return out


def first_type_spot(snapshot: Snapshot, work: str, card: Card, parsed: dict, psd, parser) -> Optional[tuple]:
    """psd_edit.template_icon() of the card's template, when it prints no type icon (its first goes at the template's
    spot, owner 10-25); the templates fetched and opened once a run."""
    if psd_edit.type_row(psd, parser, card.name)[0]:
        return None
    found = card_template(snapshot, work, card, parsed)
    return psd_edit.template_icon(found[0], psd, parser, found[1]) if found else None


def card_template(snapshot: Snapshot, work: str, card: Card, parsed: dict) -> Optional[tuple]:
    """(the card kind's template opened, its name) (psd_edit.template_for), or None; fetched and opened once a run."""
    path = psd_edit.template_for(parsed["card"].get("type"), card.rel)
    if not path or path not in snapshot.tree:
        return None
    cache = snapshot.__dict__.setdefault("templates", {})
    if path not in cache:
        local = join(work, "templates", basename(path))
        snapshot.fetch(snapshot.sha(path), local)
        cache[path] = PSDImage.open(local)
    return cache[path], basename(path)[:len(basename(path)) - 4]


def pictures_of(psd, card: Card, duplicates: list) -> dict:
    """
    The pictures of a PSD with variant tags: {n: PNG repo path}, NameN.png for each tag n (§9); {} for an untagged PSD.
    Raises Refused("unsupported-card") for a tag without its PNG.
    """
    numbers = psd_edit.variants_of(psd)
    if len(numbers) < 2:
        return {}
    stem = basename(card.rel[:len(card.rel) - 4])
    found = {int(m.group(1)): path for path in duplicates
             for m in [fullmatch(escape(stem) + r"(\d+)", basename(path)[:len(basename(path)) - 4], IGNORECASE)] if m}
    missing = [n for n in numbers if n != 1 and n not in found]
    if missing:
        raise Refused("unsupported-card", f"picture {missing[0]} has no PNG ({stem}{missing[0]}.png): export it")
    return {1: card.png, **{n: found[n] for n in numbers if n != 1}}


def _views(card: Card, record: dict, numbers: list, jobs: Optional[dict] = None) -> list:
    """The pictures an edit shows in (§4.9): a part's (None: every one) and its followers'; any other field, every one."""
    every, out = set(numbers), set()
    found = psd_edit.record_parts(record)
    for name, value in (card.jobs if jobs is None else jobs).items():
        if name != "parts":
            out |= every
            continue
        for key in value:
            if key == psd_edit.ADD_ABILITY:     # untagged: every picture
                out |= every
                continue
            shown = found[key][2].get("variants")
            out |= set(shown) if shown else every
            for f in record.get("follow") or []:
                if f["id"] == int(key) or int(key) in f.get("of", ()):
                    out |= set(f["variants"]) if f.get("variants") else every
    if card.view is not None:
        out &= {card.view}
    return sorted(out)


def bake_pictures(card: Card, psd, parsed: dict, parser, folder: str, fonts_dir: str, colors, types_dir,
                  jobs: Optional[dict] = None, target: str = "", template: Optional[tuple] = None,
                  ability_from: Optional[tuple] = None) -> psd_edit.Baked:
    """
    A tagged PSD's bake (§4.9): one apply_jobs per picture the edit shows in, each as that picture (psd_edit.variant)
    onto its own PNG, the saved picture's first (its run writes the PSD and its stored preview); every run's PSD must
    hold the same layer records. A stars, types or stat edit whose value differs by picture is impossible.

    Returns:
        The first run's Baked (card.views: every picture's)
    """
    old_psd = join(folder, "old", basename(card.rel))
    jobs, numbers = card.jobs if jobs is None else jobs, sorted(card.pictures)
    parses = {}
    for n in numbers:
        with psd_edit.variant(psd, n):
            parses[n] = psd_edit.parse_card(parser, old_psd, card.rel, psd)
    for name in jobs:
        if name in psd_edit.ICON_FIELDS + psd_edit.STAT_FIELDS and len({repr(parses[n]["card"].get(name)) for n in numbers}) > 1:
            raise Refused("unsupported-card", f"{name}: its layers differ by picture: edit it in Photoshop", name)
    record = psd_edit.card_parts(psd, card.rel, card.name, parser) if "parts" in jobs else {}
    saved = psd_edit.saved_view(psd)
    views = sorted(_views(card, record, numbers, jobs), key=lambda n: n != saved)
    if not views:
        raise Refused("unsupported-card", "the edit shows in no picture")
    card.views, first = {}, None
    for n in views:
        out = join(folder, "new" if first is None else f"new{n}")
        makedirs(out, exist_ok=True)
        baked = psd_edit.apply_jobs(old_psd, join(folder, "old", basename(card.pictures[n])), target or card.rel, jobs,
                                    parses[n], parser, out, fonts_dir, colors, update_preview=n == saved,
                                    types_dir=types_dir, psd=psd, view=n, template=template, ability_from=ability_from,
                                    title_anchor=card.title_anchor)
        if first is None:
            first = baked
        elif psd_edit.layer_records(baked.psd_path) != psd_edit.layer_records(first.psd_path):
            raise Refused("bytes", f"picture {n}'s bake made other layers than picture {views[0]}'s")
        card.views[n] = baked
    first.fx_notes = _view_notes(card.views)
    return first


def _view_notes(views: dict) -> list:
    """Every picture run's notes, an edit's only from a picture that shows its layer (§4.9: Name #2's gates as picture
    2, never picture 1's layers under it)."""
    out = []
    for n, baked in views.items():
        own = {note for edit in baked.edits for note in edit.notes}
        out += [note for edit in baked.edits if not edit.layers or shown_in(edit.layers[0], n) for note in edit.notes]
        out += [note for note in baked.fx_notes if note not in own]
    return list(dict.fromkeys(out))


def load_verdicts(path: str) -> dict:
    """The data's db/verdicts.json as {card key: verdict}; {} when the path is "" or missing (no card data yet)."""
    if not path or not exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = loads(fh.read())
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, dict):
        raise ValueError(f"{path}: not a verdicts file")
    return cards


def verdict_of(edits: CardEdits, verdicts: Optional[dict], key: str) -> Optional[dict]:
    """A card's verdict: the data's, else (the migration's one cycle, CARD_SYNC_VERDICTS.md step 6) its entry's `_psd`."""
    if verdicts and key in verdicts:
        return verdicts[key]
    return edits.meta.get(key, {}).get("_psd")


MOVED = r"name: moved ([+-]\d+) px to stay on the card"


def title_anchor(verdict: Optional[dict], psd_sha: str) -> int:
    """The px a baked verdict says its bake moved the title on (R4), while the PSD is still the one it baked; else 0."""
    if not (isinstance(verdict, dict) and verdict.get("state") == "baked" and psd_sha
            and verdict.get("psd") == psd_sha[:8]):
        return 0
    found = findall(MOVED, str(verdict.get("reason", "")))
    return int(found[len(found) - 1]) if found else 0


def final(edits: CardEdits, snapshot: Snapshot, card: Card, verdicts: Optional[dict] = None) -> bool:
    """
    Whether the card's verdict is final (pending's rule): refused for exactly its PSD-kind values on the PSD blob the
    tree holds; a `font` verdict only while MDW/Fonts holds, a `duplicate` or no-PNG one only while its PNGs do; a pixel
    refusal under psd_edit.PARITY_FLOOR (the old bar's, before 2026-10-23) or one written before RULES (content
    problems were refused) never: it bakes now. verdicts: the data's (verdict_of).
    """
    verdict = verdict_of(edits, verdicts, card.key)
    if not (isinstance(verdict, dict) and verdict.get("state") == "refused" and verdict.get("for") == card.fields
            and verdict.get("psd") == card.psd_sha[:8] and str(verdict.get("rules", "")) >= RULES):
        return False
    reason = str(verdict.get("reason", ""))
    if reason.startswith("font:") and verdict.get("fonts") != snapshot.fonts_key():
        return False
    if (reason.startswith("duplicate:") or reason == f"unsupported-card: {NO_PNG}") \
            and (verdict.get("pngs") != snapshot.pngs_key(card.rel, card.key)
                 or (card.rel and verdict.get("printed") is not False)):     # written before the redundancy check ran
        return False
    bad = search(r"^(?:parity: baseline|idempotence: \w+ re-render) bad ([\d.]+)%", reason)
    return not (bad and float(bad.group(1)) <= psd_edit.PARITY_FLOOR)


def _misses(card: Card) -> str:
    """
    '; name baked with a pixel difference: bad 1.97%, mean 1.60, max 217' (psd_edit.miss_notes), and
    '; stars not baked: unsupported-card: stars: ...' per stars or types edit refused beside the bake; "" for none.
    """
    notes = psd_edit.miss_notes(card.baked) if card.baked else []
    return "".join(f"; {note}" for note in notes + [f"{r.field} not baked: {r.code}: {r.why}" for r in card.partial])


def _fx_notes(card: Card) -> str:
    """'; Ability FX: Blazing deleted: its words left the ability' per FX layer an ability bake deleted, renamed or cut
    again (BUG 2), for the commit line and the website's verdict as for the report; "" for none."""
    return "".join(f"; {note}" for note in (getattr(card.baked, "fx_notes", None) or []))


def _describe(card: Card) -> str:
    """'hp 5 -> 8, name 'White CLot' -> 'White Clot'': the jobs for the commit subject and the report."""
    parts = []
    for name, value in card.jobs.items():
        old = next((edit.old for edit in (card.baked.edits if card.baked else []) if edit.field == name), None)
        if name == "parts":
            parts += _describe_parts(card)
        elif name == "ability":
            parts.append("ability")
        elif name == "types":
            parts.append(f"types {', '.join(old) if old else 'none'} -> {', '.join(value)}" if old is not None
                         else f"types -> {', '.join(value)}")
        elif name == "stars" and card.moved_to:
            was = psd_edit.star_folder(card.rel)
            parts.append(f"stars {was[1] if was else old} -> {value} (moved to {basename(card.moved_to)}"
                         + "".join(f"; {note}" for note in card.move_notes) + ")")
        elif name == "name":
            parts.append(f"name {old!r} -> {value!r}" if old is not None else f"name -> {value!r}")
        else:
            parts.append(f"{name} {old} -> {value}" if old is not None else f"{name} -> {value}")
    if card.renamed:
        parts.append(f"renamed: {card.name} -> {card.renamed}")
    return ", ".join(parts)


def _gone_fields(card: Card) -> list:
    """The fields whose last part edit the commit takes out (their derived name or ability goes, §4.11)."""
    parts = card.fields.get("parts") or {}
    fields_of = lambda keys: {parts[k]["field"] for k in keys if k in parts}
    return sorted(fields_of(parts) - fields_of(set(parts) - set(card.part_drop)))


def _parts_settled(card: Card) -> list:
    """The part keys a run takes out of the entry without baking them: redundant (done in Photoshop) and skipped."""
    plan = card.parts
    return [key for key, how in (plan.outcome.items() if plan is not None else ()) if how in ("redundant", "skipped")]


def _part_verdict(card: Card) -> None:
    """
    The `parts` side of a card's verdict (PARTS_EDITOR.md §3, §4.11): each part key's outcome (baked, deleted,
    redundant, skipped) in `_psd.parts`, the skipped ones' texts in `_psd.skipped` (the site's Redo), and the keys the
    commit takes out of the entry (Card.part_drop; an added ability that waits stays). A run that baked nothing but
    settled parts writes a baked verdict on the PSD as it is, its reason the count skipped.
    """
    plan = card.parts
    if plan is None:
        return
    baked = (card.jobs.get("parts") or {}) if card.baked is not None else {}
    outcome = {key: "deleted" if job.get("delete") else "baked" for key, job in baked.items()}
    outcome.update({key: how for key, how in plan.outcome.items() if how in ("redundant", "skipped")})
    card.part_drop = sorted(outcome)
    if not outcome:
        return
    skipped = {key: plan.skipped[key] for key, how in outcome.items() if how == "skipped" and key in plan.skipped}
    told = f"{plural(len(skipped), 'layer')} skipped (changed in Photoshop)" if skipped else ""
    if card.verdict is None:
        card.verdict = {"state": "baked", "psd": card.psd_sha[:8],
                        "reason": told or "already on the card", "at": _now()}
    elif told:
        card.verdict["reason"] = "; ".join(r for r in (card.verdict.get("reason"), told) if r)
    card.verdict["parts"] = dict(sorted(outcome.items()))
    if skipped:
        card.verdict["skipped"] = skipped


ORDINAL = ("1st", "2nd", "3rd")


def _nth(k: int) -> str:
    return ORDINAL[k - 1] if k <= len(ORDINAL) else f"{k}th"


def _short_change(old: str, new: str) -> str:
    """'heal 2 HP' -> 'heal 3 HP': the words an edit changed, a few around them (the commit line's)."""
    a, b = old.split(), new.split()
    lo = 0
    while lo < min(len(a), len(b)) and a[lo] == b[lo]:
        lo += 1
    hi = 0
    while hi < min(len(a), len(b)) - lo and a[len(a) - 1 - hi] == b[len(b) - 1 - hi]:
        hi += 1
    lo, hi = max(0, lo - 1), max(0, hi - 1)
    cut = lambda words: " ".join(words[lo:len(words) - hi])[:40]
    return f"{cut(a)!r} -> {cut(b)!r}"


def _describe_parts(card: Card) -> list:
    """"ability (3rd layer) 'heal 2 HP' -> 'heal 3 HP'", "4th ability layer deleted": a parts bake's lines."""
    out, plan = [], card.parts
    was = {key: text for edit in (card.baked.edits if card.baked else []) if isinstance(edit.old, dict)
           for key, text in edit.old.items()}     # the PSD's text, not the editor's `was`
    for key, job in (card.jobs.get("parts") or {}).items():
        field_name, kind, k, n = plan.where.get(key, (job["field"], "", 0, 1)) if plan is not None else (job["field"], "", 0, 1)
        what = "title" if field_name == "name" else "text block" if kind == "blocks" else "ability"
        place = f" ({_nth(k)} layer)" if n > 1 and k else ""
        if key == psd_edit.ADD_ABILITY:
            out.append(f"ability layer added {job['text'][:40]!r}")
        elif job.get("delete"):
            out.append(f"{_nth(k) + ' ' if n > 1 and k else ''}{what} layer deleted")
        else:
            out.append(f"{what}{place} {_short_change(was.get(key, job.get('was') or ''), job['text'])}")
    return out


def editors(card: Card) -> list:
    """
    The logins whose baked fields the commit prints: _by (one login, or {field: login}), sorted, deduplicated.
    """
    if not card.jobs and not card.renamed:      # nothing printed (redundant only): no credit
        return []
    by = card.by
    logins = [by.get(f) for f in card.jobs] if isinstance(by, dict) else [by]
    return sorted({x.strip() for x in logins if isinstance(x, str) and len(x.split()) == 1})   # a login: one word


def _message(cards: list) -> str:
    """
    The run's commit message: "Card edits: 5 baked, 20 redundant, 4 refused", one line per baked card, then one
    "Edit-by: <login>" trailer per baked card and editor, sorted (build_site.EDIT_BY counts them as that editor's cards).

    Args:
        cards: The cards in the commit
    """
    baked = [c for c in cards if (c.jobs or c.renamed) and not c.refusal]
    redundant = [c for c in cards if not c.jobs and not c.renamed and not c.refusal and (c.dropped or c.part_drop or not c.site)]
    site = [c for c in cards if c.site and not c.refusal]
    refused = [c for c in cards if c.refusal]
    counts = [f"{len(group)} {word}" for group, word in ((baked, "baked"), (redundant, "redundant"),
                                                         (site, "website only"), (refused, "refused")) if group]
    lines = [f"- {c.name}: {_describe(c)}{_misses(c)}{_fx_notes(c)}" for c in baked]
    lines += [f"- {c.name}: website only: {'; '.join(c.site.values())}" for c in site]
    trailers = [f"{EDIT_BY}: {login}" for login in sorted(login for c in baked for login in editors(c))]
    trailers += [f"{RENAMED}: {old} -> {new}" for c in baked if c.renames for old, new in c.relocated.items()]
    return "\n\n".join(part for part in (f"{SUBJECT}: " + ", ".join(counts), "\n".join(lines), "\n".join(trailers)) if part)


def _printed_already(snapshot: Snapshot, card: Card, parser, folder: str) -> bool:
    """
    The redundancy check before the duplicate and no-PNG gates: when the PSD already prints every PSD-kind value of the
    entry (NME Stock's name), the values are dropped like any redundant field (no PNG is needed for that) and the
    card goes no further. False when one would need a bake (the gate refuses it then).
    """
    old_psd = join(folder, "old", basename(card.rel))
    snapshot.fetch(card.psd_sha, old_psd)
    psd = PSDImage.open(old_psd)                # one open for the parse and classify
    parsed = psd_edit.parse_card(parser, old_psd, card.rel, psd)
    classified = psd_edit.classify(parsed, card.fields, psd, parser)
    if classified.jobs or classified.refusals or not classified.redundant:
        return False
    card.redundant = list(classified.redundant)
    card.expect = {card.rel: card.psd_sha, **({card.png: snapshot.sha(card.png)} if card.png else {})}
    card.dropped = list(card.redundant)
    return True


def _rename_printed(snapshot: Snapshot, card: Card, folder: str, duplicates: list, pushed: set, claimed: frozenset) -> None:
    """A printed name's rename on a card without PNG or with numbered PNGs (tagged or not): every file takes it."""
    stem = basename(card.rel)[:len(basename(card.rel)) - 4]
    card.pictures = {**({1: card.png} if card.png else {}),
                     **{int(basename(p)[len(stem):len(basename(p)) - 4]): p for p in duplicates}}
    plan_rename(snapshot, card, claimed)
    for png in card.renames:
        if png != card.rel:
            if png in pushed:
                raise Refused("recent", f"PNG pushed in the last {RECENT.seconds // 60} min")
            card.expect[png] = snapshot.sha(png)
            snapshot.fetch(card.expect[png], join(folder, "old", basename(png)))
    _relocate(card, folder)


def _smaller(path: str) -> bytes:
    """A baked PNG file's bytes, recompressed losslessly (png_export.recompress: XMP stripped, an opaque card as RGB;
    docs/Finished/session-2026-10-25/SMALLER_PNG_PUSHES.md) and written back, so the artifacts show what is committed."""
    with open(path, "rb") as fh:
        data = fh.read()
    smaller = png_export.recompress(data)
    if smaller is None:
        return data
    with open(path, "wb") as fh:
        fh.write(smaller)
    return smaller


def bake_card(snapshot: Snapshot, card: Card, parser, fonts_dir: str, chart: Callable[[], Optional[dict]], work: str,
              recent: Callable[[], Optional[set]], claimed: frozenset = frozenset()) -> None:
    """
    Section 5 for one card up to the commit: the recent, duplicate and blob gates, classify, apply with every gate.
    Fills the Card in place (its files, expect, verdict and dropped fields for land); a Refused becomes its refusal,
    any other exception its "error".

    Args:
        snapshot: The tree at the run's head
        card: select()'s Card, filled in place
        parser: vocabulary()'s offline parser
        fonts_dir: MDW/Fonts for the text engine
        chart: Returns status_fills() of the status chart, or None when the tree has none (fetched once)
        work: The run's work dir (the card's artifacts go in a folder named after it)
        recent: Returns recent_paths() (fetched once)
        claimed: Card keys the run's earlier cards rename to (plan_rename)
    """
    started = perf_counter()
    folder = join(work, card.name.replace("/", "_"))
    try:
        if not card.rel:
            original = snapshot.duplicate_of(card.key)
            if original:
                card.pngs = snapshot.pngs_key("", card.key)
                raise Refused("duplicate", f"a numbered PNG exported from `{original}`")
            raise Refused("unsupported-card", "no PSD in the card repo")
        pushed = recent()
        if pushed is None:
            raise Refused("recent", f"too many pushes in the last {RECENT.seconds // 60} min to tell")
        if card.rel in pushed:
            raise Refused("recent", f"PSD pushed in the last {RECENT.seconds // 60} min")
        card.png, duplicates = snapshot.png_for(card.rel)
        if not card.png or duplicates:
            card.pngs = snapshot.pngs_key(card.rel, card.key)
            if _printed_already(snapshot, card, parser, folder):    # nothing to bake: dropped though no PNG can be
                if card.rename:                 # its files still take the name
                    _rename_printed(snapshot, card, folder, duplicates, pushed, claimed)
                return
        if not card.png:
            raise Refused("unsupported-card", NO_PNG)
        if duplicates:                          # a PSD with variant tags bakes each picture (§4.9); untagged, refused
            card.pictures = pictures_of(PSDImage.open(join(folder, "old", basename(card.rel))), card, duplicates)
            if not card.pictures:
                raise Refused("duplicate", "numbered PNGs share the PSD: " + ", ".join(basename(p) for p in duplicates)
                              + " (tag the variant layers: PARTS_EDITOR.md §9)")
        if card.view is not None and card.view not in (card.pictures or {1: card.png}):
            raise Refused("unsupported-card", f"no picture {card.view}")
        for png in [card.png] + [p for n, p in sorted(card.pictures.items()) if n != 1]:
            if png in pushed:                   # a teammate's re-export: the merge would build on it unseen
                raise Refused("recent", f"PNG pushed in the last {RECENT.seconds // 60} min")
        card.png_sha = snapshot.sha(card.png)
        old_psd, old_png = join(folder, "old", basename(card.rel)), join(folder, "old", basename(card.png))
        snapshot.fetch(card.psd_sha, old_psd)
        snapshot.fetch(card.png_sha, old_png)
        for n, png in card.pictures.items():
            if n != 1:
                snapshot.fetch(snapshot.sha(png), join(folder, "old", basename(png)))

        psd = PSDImage.open(old_psd)            # one open for the parse, classify and apply_jobs (it edits a copy)
        parsed = psd_edit.parse_card(parser, old_psd, card.rel, psd)
        template = first_type_spot(snapshot, work, card, parsed, psd, parser) if "types" in card.fields else None
        classified = psd_edit.classify(parsed, card.fields, psd, parser, template)
        card.jobs, card.redundant = dict(classified.jobs), [f for f in classified.redundant if f != "parts"]
        card.site = dict(classified.site)       # the picture's title, a stat past its rows: website only (owner 10-26)
        card.parts = classified.parts
        card.partial = [r for r in classified.refusals if r.field in psd_edit.ICON_FIELDS]
        settled = _parts_settled(card)          # redundant or skipped boxes: out of the entry, baked or not
        if len(card.partial) < len(classified.refusals) or (card.partial and not card.jobs and not card.redundant
                                                            and not settled):
            first, *rest = classified.refusals      # per card all-or-nothing (section 10), but for stars and types
            card.partial = []
            raise Refused(first.code, "; ".join([first.why] + [f"{r.code}: {r.why}" for r in rest]))
        if not card.jobs and not card.redundant and not settled:
            if card.parts is not None and "waiting" in card.parts.outcome.values():
                raise Refused("unsupported-card", f"ability: {psd_edit.ADDED_LATER}")
            return                              # (a website-only field alone: share_of marks it)

        card.expect = {card.rel: card.psd_sha, card.png: card.png_sha,
                       **{png: snapshot.sha(png) for png in card.pictures.values()}}
        try:                                    # a stars edit on a card starred by its folder moves it
            bake = plan_move(snapshot, card, parsed, psd, parser)
        except Refused as refused:              # that field's alone, as a stars refusal is
            if not (set(card.jobs) - {"stars"}) and not card.redundant and not settled:
                raise
            card.partial.append(refused.refusal)
            del card.jobs["stars"]
            bake = dict(card.jobs)
        try:                                    # the name's files too, when asked (owner 10-25)
            plan_rename(snapshot, card, claimed)
        except Refused as refused:              # the name stays in the file, refused; the rest bakes
            if not (set(card.jobs) - {"name"}) and not [f for f in card.redundant if f != "name"] and not settled:
                raise
            card.partial.append(refused.refusal)
            card.jobs.pop("name", None)
            bake.pop("name", None)
            card.redundant = [f for f in card.redundant if f != "name"]
        if bake:
            texty = psd_edit.TEXT_FIELDS + ("parts",)
            if any(f in texty for f in card.jobs) and not psd_edit.text_engine():
                raise Transient("this runner has no text engine (psd-tools[text])")
            if any(f in texty for f in card.jobs) and not psd_edit.story_sync():
                raise Transient("this runner's psd-tools fork predates Txt2 story sync: Photoshop would undo the text")
            if any(psd_edit.glow_tokens(str(card.jobs[f])) for f in texty if f in card.jobs) \
                    and not psd_edit.glow_halos():
                raise Transient("this runner's psd-tools predates the pin's glow halos: reinstall it")
            if any(f in psd_edit.ICON_FIELDS for f in card.jobs) and not psd_edit.icon_engine():
                raise Transient("this runner's psd-tools cannot copy or make layers (the fork's duplicate and fromarray)")
            colors = chart() if {"ability", "parts"} & set(bake) else None
            if {"ability", "parts"} & set(bake) and colors is None:
                raise Transient(f"no status chart ({psd_edit.STATUS_CHART}) in the tree")
            new_dir = join(folder, "new")
            makedirs(new_dir, exist_ok=True)
            types_dir = types_icons(snapshot, work, parser, card.jobs["types"], parsed["card"].get("types")) \
                if "types" in card.jobs else None
            target = card.moves.get(card.rel, card.rel) if "stars" in bake else card.rel
            added = card_template(snapshot, work, card, parsed) \
                if psd_edit.ADD_ABILITY in psd_edit.parts_jobs(card.jobs) else None
            while True:                         # a stars or types refusal leaves the card's other fields to bake
                try:
                    if card.pictures:
                        card.baked = bake_pictures(card, psd, parsed, parser, folder, fonts_dir, colors, types_dir,
                                                   bake, target, template, added)
                    else:
                        card.baked = psd_edit.apply_jobs(old_psd, old_png, target, bake, parsed, parser, new_dir,
                                                         fonts_dir, colors, update_preview=True, types_dir=types_dir,
                                                         psd=psd, template=template, ability_from=added,
                                                         title_anchor=card.title_anchor)
                    break
                except Refused as refused:
                    if refused.field not in card.jobs or (len(card.jobs) == 1 and not card.redundant):
                        if not card.partial:
                            raise
                        earlier, card.partial = card.partial, []    # the card's reason names them too
                        raise Refused(refused.code, "; ".join([refused.why] + [f"{r.code}: {r.why}" for r in earlier])) from refused
                    card.partial.append(refused.refusal)
                    del card.jobs[refused.field]
                    bake.pop(refused.field, None)
                    if refused.field == "stars":    # not baked: not moved either
                        card.moves, card.moved_to, card.move_notes = {}, "", []
                    if refused.field == "name":     # not baked: not renamed either
                        card.renames, card.renamed = {}, ""
                    if not bake:
                        break
        if card.baked is not None:
            titled = "name" in card.jobs or any(isinstance(part, dict) and part.get("field") == "name"
                                                for part in (card.jobs.get("parts") or {}).values())
            if card.title_anchor and not titled:    # the title stays moved: its verdict keeps saying so
                card.baked.fx_notes.append(f"overflow: name: moved {card.title_anchor:+d} px to stay on the card")
            card.shared = shared_creatures(parsed["card"].get("creatures"), card.baked.reparse["card"].get("creatures"),
                                           card.jobs, card.listed)
            with open(card.baked.psd_path, "rb") as fh:
                psd_bytes = fh.read()
            card.new_psd_sha = blob_sha(psd_bytes)
            card.files = {card.rel: psd_bytes, card.png: _smaller(card.baked.png_path)}
            if card.views:                      # each picture the edit shows in: its own PNG
                card.files = {card.rel: psd_bytes}
                for n, baked in card.views.items():
                    card.files[card.pictures[n]] = _smaller(baked.png_path)
                old_png = join(folder, "old", basename(card.pictures[next(iter(card.views))]))
            crops = psd_edit.artifacts(Image.open(old_png), Image.open(card.baked.png_path), card.baked.region)
            for name, image in crops.items():
                image.save(join(folder, f"{name}.png"))
            with open(join(folder, "parse.json"), "w", encoding="utf-8") as fh:
                fh.write(dumps({"old": parsed["card"], "new": card.baked.reparse["card"], "jobs": card.jobs,
                                "validator": card.baked.reparse["validator"], "fx": card.baked.fx}, indent=1, ensure_ascii=False))
            card.verdict = {"state": "baked", "for": card.jobs, "psd": card.new_psd_sha[:8],
                            "reason": _describe(card) + _fx_notes(card), "at": _now()}
        _relocate(card, folder)
        card.dropped = [f for f in list(card.jobs) + card.redundant if f != "parts"]   # a baked or printed name goes too
        if card.partial:                        # what stays in the file: refused on the PSD this commit leaves; FX notes told
            card.verdict = {"state": "refused", "for": {r.field: card.fields[r.field] for r in card.partial},
                            "psd": (card.new_psd_sha or card.psd_sha)[:8], "rules": RULES,
                            "reason": "; ".join(f"{r.code}: {r.why}" for r in card.partial) + _fx_notes(card), "at": _now()}
        _part_verdict(card)                     # per layer: the boxes leave the entry, skipped ones told
    except Refused as refused:
        card.refusal = refused.refusal
        if refused.code == "font":              # retried once MDW/Fonts changes (pending)
            card.fonts = snapshot.fonts_key()
    except Transient as error:                  # the runner's or GitHub's: reported, no verdict
        card.refusal, card.transient = Refusal("error", str(error)), True
    except Exception as error:                  # noqa: BLE001 - one bad card never blocks another
        card.refusal = Refusal("error", f"{type(error).__name__}: {error}")
        card.transient = isinstance(error, (RequestException, ImportError))   # a 5xx, or the runner lacks scipy / the text extra
        card.traceback = format_exc()
    finally:
        card.seconds = perf_counter() - started


def _relocate(card: Card, folder: str) -> None:
    """A stars move's or rename's files: the PSD and its PNGs added at the new folder or name, deleted here."""
    card.relocated = {old: _moved_to(card, old) for old in dict.fromkeys(list(card.moves) + list(card.renames))}
    if not card.relocated:
        return
    if card.baked is None:
        card.new_psd_sha = card.psd_sha
        card.verdict = {"state": "baked", "for": card.jobs, "psd": card.psd_sha[:8], "reason": _describe(card), "at": _now()}
    for old, new in card.relocated.items():
        if old not in card.files:
            with open(join(folder, "old", basename(old)), "rb") as fh:
                card.files[old] = fh.read()
        card.files[new], card.files[old] = card.files[old], None
        card.expect[new] = ""                   # nothing there at the commit
    if card.moves:
        card.verdict["moved"] = card.moved_to
    if card.renames:
        card.verdict["renamed"] = card.renamed


def _kept(card: Card) -> bool:
    """A bake's flags (FX notes, skipped layers, a folder move): its verdict stays once no field is left."""
    return bool(_fx_notes(card)) or bool((card.verdict or {}).get("skipped")) or bool(card.moved_to) or bool(card.renamed)


def share_of(card: Card) -> Optional[Share]:
    """
    A card's part of the run's commit: a baked card's PSD, PNG and dropped fields, a redundant card's dropped fields,
    an old `_psd` taken out of its entry (the verdicts live on the data branch). None for any refusal (a refusal makes
    no main commit: its verdict is records()') or a card with nothing to write. At H' a baked or redundant card whose
    edit changed stays out (edit-changed: never remove a value you did not render).
    """
    if card.refusal is not None or (not card.dropped and not card.part_drop and not card.site):
        return None

    def drop(text: str) -> Optional[str]:
        if _wanted_at(text, card.name) != card.fields:
            raise Refused("edit-changed", f"the edit changed on {CARD_BRANCH} meanwhile: not touched")
        out = rewrite_edits(text, card.name, card.dropped, None, False, card.part_drop, _creature_one(card), card.site)
        gone = _gone_fields(card)
        for n, png in sorted(card.pictures.items()):    # the other pictures' entries: their derived name/ability
            if n != 1 and gone:
                site = CardEdits.parse(out or text).site(CardAuthors.key(card_name(png)))
                mine = [f for f in gone if f not in site]    # a website-only one stays
                out = (rewrite_edits(out or text, card_name(png), mine) or out) if mine else out
        return out
    return _rename_share(card, Share(card, card.files, card.expect, drop))


def shared_creatures(before: Optional[list], after: Optional[list], jobs: dict, listed=()) -> dict:
    """Creature n (1-based, 2...) -> {stat: value} for the other creatures a stat bake changed as well (a bar they share
    with creature 1: psd_edit._creatures_after's rule, read off the re-parse), but where the entry's creatures list
    (listed) sets the stat apart from what the card printed: the editor's, website only, it stays."""
    if not before or not after or len(before) != len(after):
        return {}
    out: dict = {}
    for stat in jobs:
        if stat in psd_edit.STAT_FIELDS:
            for n in range(1, len(after)):
                mine = listed[n] if n < len(listed) and isinstance(listed[n], dict) else {}
                if after[n].get(stat) != before[n].get(stat) and mine.get(stat) in (None, before[n].get(stat)):
                    out.setdefault(n + 1, {})[stat] = after[n].get(stat)
    return out


def _creature_one(card: Card) -> list:
    """The stats the card prints after the run (baked or already printed) per creature, for its creatures list: creature
    1's, then any other a shared bar changed ([{stat: value}, ...]; [] for a card without one)."""
    if not card.creatures:
        return []
    one = {f: card.fields[f] for f in card.dropped if f in psd_edit.STAT_FIELDS and f in card.fields}
    count = max([1] + list(card.shared))
    return [one] + [card.shared.get(n, {}) for n in range(2, count + 1)]


def refusal_verdict(card: Card) -> dict:
    """A refused card's verdict on the PSD it was tried on."""
    return {"state": "refused", "for": card.fields, "psd": card.psd_sha[:8], "rules": RULES,
            **({"fonts": card.fonts} if card.fonts else {}),
            **({"pngs": card.pngs} if card.pngs else {}),
            **({"printed": False} if card.pngs and card.rel else {}),    # the redundancy check ran first
            "reason": f"{card.refusal.code}: {card.refusal.why}", "at": _now()}


def records(report: Report, edits: CardEdits, verdicts: Optional[dict]) -> tuple:
    """
    The run's verdicts and bake events for the data branch (CARD_SYNC_VERDICTS.md), written to the work dir:
    verdicts.json {"schema": 1, "cards": {key: verdict}, "landed": {key: verdict or None}} and bake-events.json
    {"events": [...], "landed": [...]}. "cards"/"events": refusals, true without a main commit (a verdict unchanged since
    the data's is left out, so a re-check tells nothing again). "landed": the cards in the run's commit, true only once
    it lands on main: the verdict the entry would have held (None removes the data's), a `baked` event per printed card.
    Try-again refusals and runner errors write nothing.

    Returns:
        (verdicts doc, events doc)
    """
    cards, landed, events, baked = {}, {}, [], []
    for card in report.cards:
        before = verdict_of(edits, verdicts, card.key)
        if card.refusal is not None:
            if card.refusal.code in TRANSIENT or card.transient:
                continue
            card.verdict = refusal_verdict(card)
            if not _same_verdict(before, card.verdict):
                cards[card.key] = card.verdict
                events.append({"kind": "bake-refused", "card": card.name, "reason": card.verdict["reason"]})
            continue
        if not card.included:
            continue
        entry = rewrite_entry(edits.cards.get(card.key, {}), card.dropped, card.verdict, _kept(card), card.part_drop,
                              _creature_one(card), card.site)
        now = (entry or {}).get("_psd")
        if card.renamed:                        # the verdict goes with the card to its new name
            if before is not None:
                landed[card.key] = None
            landed[CardAuthors.key(card.renamed)] = now
        elif not (before is None and now is None) and not _same_verdict(before, now):
            landed[card.key] = now
        if card.jobs or card.renamed:
            pngs = sorted(p for p, data in card.files.items() if data is not None and p.lower().endswith(".png"))
            fields = [f for f in card.jobs if f != "parts"] or (["parts"] if card.jobs else ["name"])
            baked.append({"kind": "baked", "card": card.renamed or card.name, "fields": fields,
                          "reason": _describe(card) + _misses(card) + _fx_notes(card), "pngs": pngs})
    doc = {"schema": 1, "cards": cards, "landed": landed}
    told = {"events": events, "landed": baked}
    for name, data in ((VERDICTS_OUT, doc), (EVENTS_OUT, told)):
        with open(join(report.work, name), "w", encoding="utf-8") as fh:
            fh.write(dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return doc, told


def land_cards(report: Report, dry_run: bool, local: Optional[Snapshot] = None) -> None:
    """
    Land the run's cards in one commit (land; with --local's snapshot, in its clone) and record the outcome on them. A
    card left out at H' gets its try-again refusal; when the commit itself fails, the baked and redundant cards get that
    failure and the run is failed (main exits 1). Refused cards are never in it (records()). Fenced like a card's
    steps: the report always comes out.
    """
    shares = [s for s in (share_of(c) for c in report.cards) if s]
    if not shares:
        return
    try:
        report.commit = land(report.repository, shares, lambda included: _message([s.card for s in included]), dry_run,
                             join(report.work, basename(EDITS_PATH)), local)
    except Exception as error:                  # noqa: BLE001
        if isinstance(error, Refused):
            failure = error.refusal
        else:
            failure = Refusal("error", str(error) if isinstance(error, Transient) else f"{type(error).__name__}: {error}")
            if not isinstance(error, (Transient, RequestException)):
                report.traceback = format_exc()
        report.note, report.commit_failed = f"Nothing committed: {failure.code}: {failure.why}", True
        for s in shares:
            s.card.refusal, s.card.transient = failure, True
        return
    if local is not None and report.commit:
        try:
            bare = git(local.local, "rev-parse", "--is-bare-repository", env={**environ, **NO_LAZY}).decode().strip() == "true"
            report.checkout = "" if bare else checkout_head(local.local, {**environ, **NO_LAZY})
        except (CalledProcessError, TimeoutExpired, OSError):
            report.checkout = "?"
    included = [s for s in shares if s.included]
    if included:
        report.subject = _message([s.card for s in included]).split("\n", 1)[0]
    for s in shares:
        if s.out is not None:
            s.card.refusal, s.card.transient = s.out, True
        elif s.included:
            s.card.included, s.card.commit = True, report.commit


# ========================
# Report
# ========================
def _card_line(card: Card, dry_run: bool) -> str:
    where = f" `{card.rel}`" if card.rel else ""
    fields = _describe(card) if card.jobs else ", ".join(f"{f} {card.fields[f]!r}" for f in card.fields)
    line = f"**{card.name}**{where}: {fields}"
    if card.refusal:
        line += f": {card.refusal.code}: {card.refusal.why}"
    if card.baked:
        parity = "; ".join("%s %.2f%% bad, mean %.2f" % (f, m["bad_pct"], m["mean"]) for f, m in card.baked.parity.items())
        line += f"; parity {parity}{_misses(card)}; {card.baked.flag_bytes} flag bytes, {card.baked.changed_pixels} pixels"
        if card.baked.fx:
            line += f"; FX layers: {', '.join(card.baked.fx)}"
        line += _fx_notes(card)
        icons = [f"{f}: {how}" for f, how in getattr(card.baked, "icons", {}).items() if how]
        if icons:
            line += f"; {'; '.join(icons)}"
    elif card.partial:
        line += _misses(card)
    if card.redundant:
        line += f"; already on the card: {', '.join(card.redundant)}"
    if card.site:
        line += f"; website only: {'; '.join(card.site.values())}"
    if card.commit:
        line += f"; committed `{card.commit[:8]}`"
    elif not card.refusal and (card.dropped or card.baked or card.site):
        line += "; not committed (dry run)" if dry_run else "; nothing to commit"
    if card.note:
        line += f"; {card.note}"
    return line + f" ({card.seconds:.1f} s)"


def render(report: Report) -> str:
    """The markdown report (tts_save._render style)."""
    lines = ["# Card edits", "",
             f"- Repo: `{report.repository}` {CARD_BRANCH} `{report.head[:8]}`" + (" (dry run)" if report.dry_run else "")]
    if report.local:
        lines.append(f"- Local: `{report.local}`: the commit is made there, nothing pushed")
    lines += report.lines
    if report.note:
        lines += ["", f"**{report.note}**"]
    baked = [c for c in report.cards if c.baked and not c.refusal]
    redundant = [c for c in report.cards if c.redundant and not c.jobs and not c.refusal]
    refused = [c for c in report.cards if c.refusal and c.refusal.code not in ("error", "capped")]
    errors = [c for c in report.cards if c.refusal and c.refusal.code == "error"]
    capped = [c for c in report.cards if c.refusal and c.refusal.code == "capped"]
    if report.cards and not baked:
        lines += ["", "**Nothing baked.**"]

    def section(title, cards):
        if cards:
            lines.extend(["", f"## {title} ({len(cards)})"] + [f"- {_card_line(c, report.dry_run)}" for c in cards])
    section("Baked", baked)
    section("Redundant (dropped from the file)", redundant)
    section("Refused", refused)
    section("Errors", errors)
    section("Next run (capped)", capped)
    for card in errors:
        if card.traceback:
            lines += ["", f"<details><summary>{card.name}</summary>", "", "```", card.traceback.rstrip(), "```", "</details>"]
    if report.commit and report.local:
        lines += ["", f"One local commit: `{report.commit}` {report.subject} (not pushed)."]
        if report.checkout == report.commit:
            lines.append("The checkout is on it.")
        elif report.checkout == "?":
            lines.append("**The checkout's HEAD couldn't be read.**")
        elif report.checkout:
            lines.append(f"**The checkout is not on it:** HEAD is `{report.checkout[:8]}` (not --head), so check out "
                         f"`{report.commit[:8]}` before reading its tree.")
        else:
            lines.append("No checkout moved (a bare clone or an unborn HEAD).")
    elif report.commit:
        lines += ["", f"One commit: `{report.commit[:8]}` {report.subject}."]
    elif report.subject and report.dry_run:
        lines += ["", f"Would commit: {report.subject}."]
    if report.traceback:
        lines += ["", "<details><summary>commit</summary>", "", "```", report.traceback.rstrip(), "```", "</details>"]
    lines += ["", f"Took {report.seconds:.0f} s."]
    return "\n".join(lines) + "\n"


# ========================
# Public API Functions
# ========================
def failure(report: Report) -> str:
    """Why the run needs a person (main exits 1; card-sync.yml tells Discord): the commit failed, or a card errored."""
    if report.error or report.commit_failed:
        return report.error or report.note
    errors = [c.name for c in report.cards if c.refusal and c.refusal.code == "error"]
    return f"{plural(len(errors), 'card')} with an error: {', '.join(errors)}" if errors else ""


def run(repository: str = CARD_REPO, head: str = "", dry_run: bool = False, card: Optional[str] = None,
        pending_only: bool = False, local: str = "", verdicts: str = "") -> tuple:
    """
    Bake the PSD-kind edits at head and land everything in one commit on the branch's current head (run_report).

    Returns:
        (markdown report, commit shas: one at most); raises on a failure outside a card's steps and the commit
    """
    report = run_report(repository, head, dry_run, card, pending_only, local, verdicts)
    return report.text, [report.commit] if report.commit else []


def run_report(repository: str = CARD_REPO, head: str = "", dry_run: bool = False, card: Optional[str] = None,
               pending_only: bool = False, local: str = "", verdicts: str = "") -> Report:
    """
    run() as a Report: at most MAX_BAKES cards baked and none started after BUDGET; the rest are refused `capped`
    (try-again: pending next run).

    Args:
        repository: owner/name of the card repo
        head: The commit whose tree and edits file are the snapshot ("" = the branch head)
        dry_run: Write the artifacts, commit nothing
        card: Restrict the run to one card (matched like the authors file)
        pending_only: Skip the cards with a final verdict (the schedule: pending()'s cards only, so refused cards
            cost no time from BUDGET); a dispatch re-checks them
        local: A clone of the card repo (blobless is enough): the tree and blobs from it, head "" = its HEAD, and the
            commit B made there on head, not pushed (a checkout at head moves to it)
        verdicts: The data's db/verdicts.json ("" or missing: none yet, the entries' old `_psd` only)

    Returns:
        The Report, its text rendered and written to the work dir's report.md; the verdicts and bake events beside it
        (records())
    """
    started = perf_counter()
    token_from_env()                            # the Action has no config.py: github_api's fetches take the token from the environment
    work = _work_dir()
    for name in (VERDICTS_OUT, EVENTS_OUT):    # a reused work dir: no earlier run's records
        if exists(join(work, name)):
            remove(join(work, name))
    local = abspath(expanduser(local)) if local else ""
    try:
        snapshot = local_snapshot(repository, local, head) if local else snapshot_at(repository, head or ref_head(repository))
    except Transient as error:
        if not local:
            raise
        snapshot = None                         # card-sync goes on from main's head: no B
        report = Report(repository, head, dry_run, work, local=local)
        report.error = report.note = f"Nothing baked: {error}"
    if snapshot is not None:
        head = snapshot.head
        report = Report(repository, head, dry_run, work, local=local)
    report.lines.append(f"- Work dir: `{work}`")
    if snapshot is None:
        pass
    elif snapshot.edits_path is None:
        report.note = f"No {EDITS_PATH} at `{head[:8]}`."
    else:
        text = snapshot.fetch(snapshot.sha(snapshot.edits_path)).decode("utf-8")
        try:
            edits = CardEdits.parse(text)
        except ValueError as error:
            edits = None
            report.note = f"{snapshot.edits_path} isn't JSON: {error}"
        if edits is not None:
            known = load_verdicts(verdicts)
            cards = select(edits, snapshot, card)
            for one in cards:
                one.title_anchor = title_anchor(verdict_of(edits, known, one.key), one.psd_sha)
            named = len(cards)
            if pending_only:
                cards = [c for c in cards if not final(edits, snapshot, c, known)]
            report.lines.append(f"- Edits: `{snapshot.edits_path}` `{snapshot.sha(snapshot.edits_path)[:8]}`: "
                                f"{plural(len(edits.cards), 'card')}, {named} with PSD fields"
                                + (f" (--card {card})" if card else "")
                                + (f", {named - len(cards)} refused before (not re-checked)" if named > len(cards) else ""))
            if card and not cards:
                report.note = f"No entry with a PSD field for {card!r}."
            if cards:
                fonts = _fonts_dir(snapshot)
                engine = ("no text engine" if not psd_edit.text_engine()
                          else "text engine" if psd_edit.story_sync() else "text engine without Txt2 story sync")
                report.lines.append(f"- Runner: psd-tools {PSD_TOOLS_VERSION} ({engine}); fonts `{fonts}`"
                                    + ("" if isdir(fonts) else " (missing)"))
                parser = vocabulary(snapshot)
                colors: list = []               # the chart's fills, read once

                def chart() -> Optional[dict]:
                    if not colors:
                        if psd_edit.STATUS_CHART not in snapshot.tree:
                            colors.append(None)
                        else:
                            path = join(work, basename(psd_edit.STATUS_CHART))
                            snapshot.fetch(snapshot.sha(psd_edit.STATUS_CHART), path)
                            colors.append(psd_edit.status_fills(PSDImage.open(path)))
                    return colors[0]
                pushed: list = []               # recent_paths(), read once
                claimed: set = set()            # the card keys earlier cards rename to: one name, one card

                def recent() -> Optional[set]:
                    if not pushed:
                        pushed.append(recent_paths(repository, head, datetime.now(timezone.utc), local))
                    return pushed[0]
                for one in cards:
                    baked = sum(1 for c in cards if c.baked)
                    if baked >= MAX_BAKES or perf_counter() - started >= BUDGET.total_seconds():
                        one.refusal = Refusal("capped", f"{baked} cards baked in {(perf_counter() - started) / 60:.0f} min "
                                                        f"(at most {MAX_BAKES} or {BUDGET.seconds // 60} min a run)")
                        continue
                    bake_card(snapshot, one, parser, fonts, chart, work, recent, frozenset(claimed))
                    if one.renames and one.refusal is None:
                        claimed.update(CardAuthors.key(card_name(new)) for new in one.renames.values())
                waiting = sum(1 for c in cards if c.refusal and c.refusal.code == "capped")
                if waiting:
                    report.lines.append(f"- Capped: {plural(waiting, 'card')} wait for the next run")
                report.cards = cards
                land_cards(report, dry_run, snapshot if local else None)
                records(report, edits, known)
    report.seconds = perf_counter() - started
    report.text = render(report)
    with open(join(work, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.text)
    return report


def _pending(repository: str, head: str, local: str, verdicts: str = "") -> tuple:
    """(snapshot, pending's Cards) at head."""
    token_from_env()
    snapshot = local_snapshot(repository, local, head) if local else snapshot_at(repository, head or ref_head(repository))
    if snapshot.edits_path is None:
        return snapshot, []
    edits = CardEdits.parse(snapshot.fetch(snapshot.sha(snapshot.edits_path)).decode("utf-8"))
    known = load_verdicts(verdicts)
    return snapshot, [card for card in select(edits, snapshot) if not final(edits, snapshot, card, known)]


def pending(repository: str = CARD_REPO, head: str = "", local: str = "", verdicts: str = "") -> list:
    """
    The schedule's change gate: the cards a live run has work for at head, i.e. a PSD-kind edit
    without a final verdict (final(); build_site.apply_edits shows a refusal by the same rule); a baked field leaves the
    file. So new, changed and redundant edits, try-again refusals (recent, head-moved, edit-changed, capped, runner
    errors), refused cards whose PSD changed since, `font` refusals whose MDW/Fonts changed and `duplicate` or no-PNG
    refusals whose PNGs changed are pending, and after a complete live run nothing is.

    Args:
        local: A clone of the card repo to read the tree from (local_snapshot), else the API
        verdicts: The data's db/verdicts.json (load_verdicts)

    Returns:
        Card names in the file's order ([] when nothing is pending or the file is missing); raises ValueError when the
        file isn't JSON (the gate step fails and Discord is told)
    """
    return [card.name for card in _pending(repository, head, local, verdicts)[1]]


def due(repository: str = CARD_REPO, head: str = "", local: str = "", verdicts: str = "") -> list:
    """
    pending()'s cards a run can bake now: PSD and PNG not pushed within RECENT (Card sync's Pending edits step). []
    while the pushes can't be told (the bake would refuse them all `recent`).
    """
    snapshot, cards = _pending(repository, head, local, verdicts)
    if not cards:
        return []
    pushed = recent_paths(repository, snapshot.head, datetime.now(timezone.utc), snapshot.local)
    if pushed is None:
        return []
    return [card.name for card in cards if not card.rel or not {card.rel, snapshot.png_for(card.rel)[0]} & pushed]


def main(args: Optional[list] = None) -> int:
    """The command line: run() with the Action's flags, the report on stdout; exit 1 when failure() names a reason."""
    parser = ArgumentParser(prog="python -m commands.card_bake",
                            description="Bake the PSD-kind edits of site/Card Edits.json into the card PSDs and PNGs, "
                                        "in one commit (docs/EDIT_PIPELINE_DESIGN.md).")
    parser.add_argument("--repo", default=CARD_REPO, help=f"owner/name of the card repo (default {CARD_REPO})")
    parser.add_argument("--head", default="", help=f"commit to read the tree and edits file at (default: {CARD_BRANCH}'s head)")
    parser.add_argument("--dry-run", action="store_true", help="write the artifacts, commit nothing")
    parser.add_argument("--card", default=None, help="only this card (matched like the authors file)")
    parser.add_argument("--pending", action="store_true", help="skip cards whose refused verdict still holds (the schedule)")
    parser.add_argument("--local", default="", metavar="CLONE",
                        help="read the tree and blobs from this clone of the card repo (--head default: its HEAD) and make "
                             "the commit there, pushing nothing; a checkout at --head moves to it (the card sync)")
    parser.add_argument("--verdicts", default="", metavar="FILE",
                        help="the data branch's db/verdicts.json (the verdicts --pending keeps to); the run's own go to the "
                             "work dir's verdicts.json and bake-events.json")
    options = parser.parse_args(args)
    report = run_report(options.repo, options.head, options.dry_run, options.card, options.pending, options.local,
                        options.verdicts)
    print(report.text, end="")
    why = failure(report)
    if why:
        print(f"card_bake: {why}", file=stderr)
    return 1 if why else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'Card',
    'ORDER',
    'Report',
    'Share',
    'Snapshot',
    'Transient',
    'bake_card',
    'due',
    'dump_edits',
    'editors',
    'failure',
    'fetch_verified',
    'final',
    'land',
    'land_cards',
    'load_verdicts',
    'local_snapshot',
    'main',
    'pending',
    'psd_fields',
    'recent_paths',
    'records',
    'refusal_verdict',
    'render',
    'rewrite_edits',
    'rewrite_entry',
    'run',
    'run_report',
    'select',
    'share_of',
    'verdict_of',
    'vocabulary',
]
