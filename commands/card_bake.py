"""
The edit pipeline's driver (docs/EDIT_PIPELINE_DESIGN.md sections 4, 5, 8-11): bake the PSD-kind edits of
TTSCardMaker's site/Card Edits.json into the card PSDs and PNGs, one commit per run.

    python -m commands.card_bake --repo MichaelJSr/TTSCardMaker --head <sha> [--dry-run] [--card <Name>] [--pending]
                                 [--local <clone>]
    run(repository, head, dry_run, card, pending_only, local) -> (markdown report, commit shas: one at most)
    pending(repository, head) -> the cards a live run has work for (card-edits.yml's schedule gate)

Input: one tree snapshot at head and blobs by sha (the card folders are never checked out). Per card named in the
file with a PSD-kind field: refuse `recent` (PSD or PNG pushed in the last 30 min), fetch its PSD and official PNG (gate
`blob`), classify the edits against a fresh offline parse (psd_edit.classify), apply them with every gate
(psd_edit.apply_jobs; an ability edit also regenerates the status words' FX layers from the chart, section 7 idea 14).
Then one commit on the branch's current head (land): every baked card's PSD and PNG and the file rewritten with the
baked and redundant fields removed and the `_psd` verdicts written (a Python port of edit.js editsFile), through
git_commit (the GitHub API, or git objects in the history clone pushed over SSH when the token can't write). A card
whose PSD, PNG or edit changed meanwhile is left out of the commit as a try-again refusal; the others still land.
--local (card-sync.yml's stage 1, docs/Campaigns/DATA_BUILD_ACTION.md 6.3): the tree and blobs come from a local clone
(blobless is enough; a blob it lacks by REST) and the run's commit B is made there on head, pushed by no one here; a
checkout at head moves to B.
Refusals are data: one bad card never blocks another. A run bakes at most MAX_BAKES cards and starts none after BUDGET
(the rest are refused `capped` and wait for the next run). The report goes to stdout, the artifacts (edited PSD, merged
PNG, before/after/diff crops, the rewritten file) to the work dir. The runner supplies the token
(GITHUB_TOKEN, or config.GIT_TOKEN on the bot), the fonts (PSD_TOOLS_FONT_DIRS; else the checkout's MDW/Fonts, else the
fonts cache by sha, font_check.refresh_cache) and the work dir (RUNNER_TEMP).
No Discord or config imports; never the pickles; --dry-run commits nothing. GitHub, git and runner failures (a 5xx,
a rate limit, no text engine) are reported but never written as a verdict; such a card error, or a commit that failed,
exits 1 (failure()). The `font` gate refuses only where a face resolves to nothing: on a Mac the platform fonts stand
in for a missing or empty fonts dir (psd_edit.OK_FONTS); a `font` verdict holds until MDW/Fonts changes.
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from json import dumps, loads
from os import environ, makedirs, pathsep
from os.path import abspath, basename, dirname, expanduser, isdir, join, splitext
from re import IGNORECASE, escape, fullmatch
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import stderr
from tempfile import mkdtemp
from time import perf_counter
from traceback import format_exc
from typing import Callable, Optional
from unicodedata import combining, normalize

from PIL import Image
from psd_tools import PSDImage, __version__ as PSD_TOOLS_VERSION
from requests import RequestException

import commands.github_api as github_api
import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import card_name
from commands.git_commit import (API_TIMEOUT, COMMIT_TRIES, GIT_TIMEOUT, NO_LAZY, Transient, api, commit_files, commit_info,
                                 git, local_commit, plural, ref_head)
from commands.github_api import blob_sha, token_from_env
from commands.psd_edit import Refusal, Refused
from global_config import CARD_BRANCH, CARD_REPO, EDITS_PATHS, FONT_DIRS_ENV, ICON_MANIFEST_PATH, LOCAL_DIR_LOC, STARS_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
RECENT = timedelta(minutes=30)                  # a PSD or PNG pushed more recently is its author's: don't race them
PUSHES = 100                                    # activity entries per request (GitHub's max)
COMPARE_FILES = 300                             # a compare lists at most this many files
SUBJECT = "Card edits"                          # the run's commit: "Card edits: 5 baked, 20 redundant, 4 refused"
EDIT_BY = "Edit-by"                             # trailer key per editor credited (TTSCardMaker build_site.EDIT_BY)
NO_PNG = "no PNG beside the PSD"
MAX_BAKES = 40                                  # PSD edits per run; the rest wait for the next run (a title took ~30 s)
BUDGET = timedelta(minutes=40)                  # no card starts later: the commit lands within the job's 60 min
TRANSIENT = ("recent", "head-moved", "edit-changed", "capped")   # try-again refusals: not a verdict for the editor
EXCLUDE_FOLDERS = ("Markers", "MDW")            # not cards (repo_traverser.EXCLUDE_FOLDERS)
WORK_ENV = "RUNNER_TEMP"                        # the Action's temp dir; the work dir is card-bake/ in it
WORK_NAME = "card-bake"
# edit.js FIELDS, in order: how a rewritten entry lists its fields
ORDER = ("name", "type", "subtype", "series", "subseries", "stars", "types", "hp", "def", "atk", "spd", "ability",
         "note", "aliases", "dup_of", "hidden", "problems_ok", "allow_high_stats", "creatures")
FIELDS = frozenset(CardEdits.TYPES)
# what localeCompare(…, {sensitivity: "base"}) puts before digits and letters, in CLDR root order
PUNCTUATION = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"
KIND = CardEdits.KIND


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
    dropped: list = field(default_factory=list)     # fields removed from the file
    creatures: bool = False                     # the entry has a creatures list (it sets the stats)
    by: object = None                           # the entry's _by: who asked for its fields (Edit-by trailers)
    verdict: Optional[dict] = None              # the _psd meta written for it (baked or refused)
    fonts: str = ""                             # a `font` refusal's Snapshot.fonts_key: final only while MDW/Fonts holds
    pngs: str = ""                              # a PNG refusal's Snapshot.pngs_key: final only while the PNGs hold
    note: str = ""
    traceback: str = ""
    seconds: float = 0.0


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


class Snapshot:
    """The card repo at one commit: the tree, the PSDs by card key and where the edits file is."""

    def __init__(self, repository: str, head: str, info: dict, tree: dict, local: str = ""):
        self.repository, self.head, self.info, self.tree, self.local = repository, head, info, tree, local
        self.lower = {path.lower() for path in tree}
        self.psds: dict = {}
        for path in tree:
            if path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/")):
                self.psds[CardAuthors.key(card_name(path))] = path
        self.edits_path = next((p for p in EDITS_PATHS if p in tree), None)

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


def recent_paths(repository: str, head: str, now: datetime) -> Optional[set]:
    """
    The paths that pushes within RECENT of now changed, up to head: by push time, not commit date (two requests at most).

    Returns:
        The paths, or None when there were too many pushes or files to tell
    """
    since = (now - RECENT).strftime("%Y-%m-%dT%H:%M:%SZ")
    pushes = api(repository, "GET", "activity", params={"ref": CARD_BRANCH, "time_period": "day", "per_page": PUSHES}).json()
    window = [push for push in pushes if (push.get("timestamp") or "") >= since]
    if not window:
        return set()
    base = min(window, key=lambda push: push["timestamp"]).get("before") or ""
    if len(window) >= PUSHES or not base.strip("0"):  # a new branch has no before
        return None
    files = api(repository, "GET", f"compare/{base}...{head}", params={"per_page": 1}).json().get("files") or []
    if len(files) >= COMPARE_FILES:
        return None
    return {f["filename"] for f in files} | {f["previous_filename"] for f in files if f.get("previous_filename")}


# ========================
# Card Edits.json (the edit.js editsFile port)
# ========================
def _json(value) -> str:
    """JSON.stringify: compact, UTF-8 as is."""
    return dumps(value, ensure_ascii=False, separators=(",", ":"))


def _collate(key: str) -> list:
    """How localeCompare with sensitivity "base" orders keys: case and accents ignored; spaces, punctuation, symbols, digits, letters."""
    out = []
    for char in normalize("NFKD", key):
        if combining(char):
            continue
        if char.isspace():
            out.append((0, 0))
        elif char in PUNCTUATION:
            out.append((1, PUNCTUATION.index(char)))
        elif char.isdigit():
            out.append((3, ord(char)))
        elif char.isalpha():
            out.append((4, ord(char.lower())))
        else:
            out.append((2, ord(char)))
    return out


def dump_edits(data: dict) -> str:
    """
    Card Edits.json as edit.js editsFile writes it: the other top-level keys first, cards sorted, one line per card.

    Args:
        data: The parsed file

    Returns:
        The text (with its final newline)
    """
    cards = data.get("cards") or {}
    rest = "".join(f"  {_json(k)}: {_json(v)},\n" for k, v in data.items() if k != "cards")
    keys = sorted(cards, key=_collate)
    body = ",\n".join(f"    {_json(k)}: {_json(cards[k])}" for k in keys)
    return "{\n" + rest + '  "cards": {' + (f"\n{body}\n  " if keys else "") + "}\n}\n"


def rewrite_entry(entry: dict, drop=(), psd: Optional[dict] = None) -> Optional[dict]:
    """
    One card's entry as the pipeline leaves it: the dropped fields gone, "_psd" replaced (None removes it), fields
    in ORDER, then the other keys in their order ("_psd" where it was, else last), "_by", "_at" (edit.js editsFile's
    layout; _by/_at are kept as they are).

    Args:
        entry: The card's entry as parsed from the file
        drop: Fields to remove
        psd: The "_psd" verdict to write, None to remove it

    Returns:
        The entry, or None when no field is left: the entry goes whole, its _-keys with it (section 5 step 10)
    """
    current = {k: v for k, v in entry.items() if k not in drop and k not in ("_by", "_at")}
    fields = [f for f in ORDER if f in current]
    if not fields:
        return None
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


def rewrite_edits(text: str, name: str, drop=(), psd: Optional[dict] = None) -> Optional[str]:
    """
    Card Edits.json with one card's entry rewritten (matched like edit.js: keys compared as CardAuthors.key).

    Args:
        text: The file
        name: Card name
        drop: Fields to remove
        psd: The "_psd" verdict to write, None to remove it

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
    entry = rewrite_entry(cards[key], drop, psd)
    if entry == cards[key]:
        return None
    if entry is None:
        del cards[key]
    else:
        cards[key] = entry
    return dump_edits(data)


def psd_fields(fields: dict) -> dict:
    """The PSD-kind fields of an entry with a value (the job key, section 5)."""
    return {k: v for k, v in fields.items() if KIND.get(k) == "psd" and v is not None}


def _entry_at(text: str, name: str) -> Optional[dict]:
    """A card's parsed entry in a file text (fields valid for CardEdits), or None."""
    return CardEdits.parse(text).cards.get(CardAuthors.key(name))


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
        edits_path = next((p for p in EDITS_PATHS if p in tree), None)
        if edits_path is None:
            raise Refused("head-moved", f"no {EDITS_PATHS[0]} on {CARD_BRANCH} `{head[:8]}`")
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
        if out_path:
            makedirs(dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
        if dry_run:
            return None
        payload = {**files, **({edits_path: new_text.encode("utf-8")} if new_text != text else {})}
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
    """
    FONT_DIRS_ENV's first dir, else the checkout's MDW/Fonts, else the fonts cache brought to the snapshot's MDW/Fonts.

    Raises Transient when a font file failed to download: a missing face would be refused as a `font` verdict.
    """
    dirs = environ.get(FONT_DIRS_ENV, "")
    if dirs:
        return dirs.split(pathsep)[0]
    checkout = join(snapshot.local if snapshot is not None and snapshot.local else expanduser(LOCAL_DIR_LOC), psd_edit.FONTS_DIR)
    if isdir(checkout) or snapshot is None:
        return checkout
    from commands.font_check import refresh_cache, tree_fonts      # lazy: not in the Action's files
    fonts = refresh_cache(snapshot.repository, tree_fonts({path: sha for path, (_, sha) in snapshot.tree.items()}))
    if fonts.failed:
        name = min(fonts.failed)
        raise Transient(f"fonts cache: {plural(len(fonts.failed), 'font file')} failed to download ({name}: {fonts.failed[name]})")
    return fonts.directory


def vocabulary(snapshot: Snapshot):
    """The icon vocabulary from the tree (Types/, Types/Stars/, the stoogedings manifest), as repo_traverser builds it."""
    all_types = [splitext(basename(p))[0].lower() for p in snapshot.tree
                 if p.startswith(TYPES_DIR) and "Stars" not in p and "." in p]
    stars = [splitext(basename(p))[0].lower().replace(" ", "_") for p in snapshot.tree
             if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
    manifest = None
    if ICON_MANIFEST_PATH in snapshot.tree:
        manifest = loads(snapshot.fetch(snapshot.sha(ICON_MANIFEST_PATH)).decode("utf-8"))
    return psd_edit.offline_parser(all_types, stars, manifest)


def select(edits: CardEdits, snapshot: Snapshot, only: Optional[str] = None) -> list:
    """
    The cards the run looks at: every entry with a PSD-kind field (one card with --card).

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
        wanted = psd_fields(fields)
        if not wanted:
            continue
        rel = snapshot.psds.get(key, "")
        name = card_name(rel) if rel else " ".join(word[:1].upper() + word[1:] for word in key.split())
        cards.append(Card(name, key, wanted, rel, snapshot.sha(rel) if rel else "", creatures=bool(fields.get("creatures")),
                          by=edits.meta.get(key, {}).get("_by")))
    return cards


def final(edits: CardEdits, snapshot: Snapshot, card: Card) -> bool:
    """
    Whether the card's verdict is final (pending's rule): refused for exactly its PSD-kind values on the PSD blob the
    tree holds; a `font` verdict only while MDW/Fonts holds, a `duplicate` or no-PNG one only while its PNGs do.
    """
    verdict = edits.meta.get(card.key, {}).get("_psd")
    if not (isinstance(verdict, dict) and verdict.get("state") == "refused" and verdict.get("for") == card.fields
            and verdict.get("psd") == card.psd_sha[:8]):
        return False
    reason = str(verdict.get("reason", ""))
    if reason.startswith("font:") and verdict.get("fonts") != snapshot.fonts_key():
        return False
    if (reason.startswith("duplicate:") or reason == f"unsupported-card: {NO_PNG}") \
            and verdict.get("pngs") != snapshot.pngs_key(card.rel, card.key):
        return False
    return True


def _describe(card: Card) -> str:
    """'hp 5 -> 8, name 'White CLot' -> 'White Clot'': the jobs for the commit subject and the report."""
    parts = []
    for name, value in card.jobs.items():
        old = next((edit.old for edit in (card.baked.edits if card.baked else []) if edit.field == name), None)
        if name == "ability":
            parts.append("ability")
        elif name == "name":
            parts.append(f"name {old!r} -> {value!r}" if old is not None else f"name -> {value!r}")
        else:
            parts.append(f"{name} {old} -> {value}" if old is not None else f"{name} -> {value}")
    return ", ".join(parts)


def editors(card: Card) -> list:
    """The logins whose baked fields the commit prints: _by (one login, or {field: login}), sorted, deduplicated."""
    if not card.jobs:                           # nothing printed (redundant only): no credit
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
    baked = [c for c in cards if c.jobs and not c.refusal]
    redundant = [c for c in cards if not c.jobs and not c.refusal]
    refused = [c for c in cards if c.refusal]
    counts = [f"{len(group)} {word}" for group, word in ((baked, "baked"), (redundant, "redundant"), (refused, "refused")) if group]
    lines = [f"- {c.name}: {_describe(c)}" for c in baked]
    trailers = [f"{EDIT_BY}: {login}" for login in sorted(login for c in baked for login in editors(c))]
    return "\n\n".join(part for part in (f"{SUBJECT}: " + ", ".join(counts), "\n".join(lines), "\n".join(trailers)) if part)


def bake_card(snapshot: Snapshot, card: Card, parser, fonts_dir: str, chart: Callable[[], Optional[dict]], work: str,
              recent: Callable[[], Optional[set]]) -> None:
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
        stats = [f for f in card.fields if f in psd_edit.STAT_FIELDS]
        if stats and card.creatures:            # CardEdits.apply sets the stats from creature 1: a baked stat would be undone
            raise Refused("unsupported-card", f"{', '.join(stats)}: the entry's creatures list sets the stats")
        pushed = recent()
        if pushed is None:
            raise Refused("recent", f"too many pushes in the last {RECENT.seconds // 60} min to tell")
        if card.rel in pushed:
            raise Refused("recent", f"PSD pushed in the last {RECENT.seconds // 60} min")
        card.png, duplicates = snapshot.png_for(card.rel)
        if not card.png or duplicates:
            card.pngs = snapshot.pngs_key(card.rel, card.key)
        if not card.png:
            raise Refused("unsupported-card", NO_PNG)
        if duplicates:
            raise Refused("duplicate", "numbered PNGs share the PSD: " + ", ".join(basename(p) for p in duplicates))
        if card.png in pushed:                  # a teammate's re-export: the merge would build on it unseen
            raise Refused("recent", f"PNG pushed in the last {RECENT.seconds // 60} min")
        card.png_sha = snapshot.sha(card.png)
        old_psd, old_png = join(folder, "old", basename(card.rel)), join(folder, "old", basename(card.png))
        snapshot.fetch(card.psd_sha, old_psd)
        snapshot.fetch(card.png_sha, old_png)

        parsed = psd_edit.parse_card(parser, old_psd, card.rel)
        classified = psd_edit.classify(parsed, card.fields, PSDImage.open(old_psd))
        card.jobs, card.redundant = dict(classified.jobs), list(classified.redundant)
        if classified.refusals:                 # per card all-or-nothing (section 10)
            first, *rest = classified.refusals
            raise Refused(first.code, "; ".join([first.why] + [f"{r.code}: {r.why}" for r in rest]))
        if not card.jobs and not card.redundant:
            return

        card.expect = {card.rel: card.psd_sha, card.png: card.png_sha}
        if card.jobs:
            if any(f in psd_edit.TEXT_FIELDS for f in card.jobs) and not psd_edit.text_engine():
                raise Transient("this runner has no text engine (psd-tools[text])")
            colors = chart() if "ability" in card.jobs else None
            if "ability" in card.jobs and colors is None:
                raise Transient(f"no status chart ({psd_edit.STATUS_CHART}) in the tree")
            new_dir = join(folder, "new")
            makedirs(new_dir, exist_ok=True)
            card.baked = psd_edit.apply_jobs(old_psd, old_png, card.rel, card.jobs, parsed, parser, new_dir, fonts_dir,
                                             colors, update_preview=True)
            with open(card.baked.psd_path, "rb") as fh:
                psd_bytes = fh.read()
            with open(card.baked.png_path, "rb") as fh:
                png_bytes = fh.read()
            card.new_psd_sha = blob_sha(psd_bytes)
            card.files = {card.rel: psd_bytes, card.png: png_bytes}
            crops = psd_edit.artifacts(Image.open(old_png), Image.open(card.baked.png_path), card.baked.region)
            for name, image in crops.items():
                image.save(join(folder, f"{name}.png"))
            with open(join(folder, "parse.json"), "w", encoding="utf-8") as fh:
                fh.write(dumps({"old": parsed["card"], "new": card.baked.reparse["card"], "jobs": card.jobs,
                                "validator": card.baked.reparse["validator"], "fx": card.baked.fx}, indent=1, ensure_ascii=False))
            card.verdict = {"state": "baked", "for": card.jobs, "psd": card.new_psd_sha[:8], "reason": _describe(card),
                            "at": _now()}
        card.dropped = list(card.jobs) + card.redundant     # a baked or printed name goes too: the bot reads the title
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


def share_of(card: Card) -> Optional[Share]:
    """
    A card's part of the run's commit: a baked card's PSD, PNG and file entry, a redundant card's entry, a refused card's
    verdict (section 9's _psd). None for a try-again refusal (no verdict for the editor) or a card with nothing to write.
    At H' a baked or redundant card whose edit changed stays out (edit-changed: never remove a value you did not
    render); a refused card whose edit changed gets no verdict, and an unchanged verdict writes nothing.
    """
    if card.refusal is None:
        if not card.dropped:
            return None

        def drop(text: str) -> Optional[str]:
            if psd_fields(_entry_at(text, card.name) or {}) != card.fields:
                raise Refused("edit-changed", f"the edit changed on {CARD_BRANCH} meanwhile: not touched")
            return rewrite_edits(text, card.name, card.dropped, card.verdict)
        return Share(card, card.files, card.expect, drop)
    if card.refusal.code in TRANSIENT or card.transient:
        return None
    card.verdict = {"state": "refused", "for": card.fields, "psd": card.psd_sha[:8], **({"fonts": card.fonts} if card.fonts else {}),
                    **({"pngs": card.pngs} if card.pngs else {}),
                    "reason": f"{card.refusal.code}: {card.refusal.why}", "at": _now()}

    def refuse(text: str) -> Optional[str]:
        entry = _entry_at(text, card.name)
        if entry is None or psd_fields(entry) != card.fields:
            raise Refused("edit-changed", "edit changed meanwhile: verdict not written")
        if _same_verdict(CardEdits.parse(text).meta.get(card.key, {}).get("_psd"), card.verdict):
            return None
        return rewrite_edits(text, card.name, (), card.verdict)
    return Share(card, {}, {}, refuse)


def land_cards(report: Report, dry_run: bool, local: Optional[Snapshot] = None) -> None:
    """
    Land the run's cards in one commit (land; with --local's snapshot, in its clone) and record the outcome on them. A
    card left out at H' gets its try-again refusal (a refused card a note); when the commit itself fails, the baked and
    redundant cards get that failure, the refused ones a note that their verdict was not written, and the run is failed
    (main exits 1). Fenced like a card's steps: the report always comes out.
    """
    shares = [s for s in (share_of(c) for c in report.cards) if s]
    if not shares:
        return
    try:
        report.commit = land(report.repository, shares, lambda included: _message([s.card for s in included]), dry_run,
                             join(report.work, basename(EDITS_PATHS[0])), local)
    except Exception as error:                  # noqa: BLE001
        if isinstance(error, Refused):
            failure = error.refusal
        else:
            failure = Refusal("error", str(error) if isinstance(error, Transient) else f"{type(error).__name__}: {error}")
            if not isinstance(error, (Transient, RequestException)):
                report.traceback = format_exc()
        report.note, report.commit_failed = f"Nothing committed: {failure.code}: {failure.why}", True
        for s in shares:
            if s.card.refusal is None:
                s.card.refusal, s.card.transient = failure, True
            else:
                s.card.note = "verdict not written: nothing committed"
        return
    included = [s for s in shares if s.included]
    if included:
        report.subject = _message([s.card for s in included]).split("\n", 1)[0]
    for s in shares:
        if s.out is not None and s.card.refusal is not None:   # a refused card whose edit changed: no verdict
            s.card.note = s.out.why
        elif s.out is not None:
            s.card.refusal, s.card.transient = s.out, True
        elif s.included:
            s.card.commit = report.commit


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
        line += f"; parity {parity}; {card.baked.flag_bytes} flag bytes, {card.baked.changed_pixels} pixels"
        if card.baked.fx:
            line += f"; FX layers: {', '.join(card.baked.fx)}"
    if card.redundant:
        line += f"; already on the card: {', '.join(card.redundant)}"
    if card.commit:
        line += f"; committed `{card.commit[:8]}`"
    elif not card.refusal and (card.dropped or card.baked):
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
    """Why the run needs a person (main exits 1; card-edits.yml tells Discord): the commit failed, or a card errored."""
    if report.commit_failed:
        return report.note
    errors = [c.name for c in report.cards if c.refusal and c.refusal.code == "error"]
    return f"{plural(len(errors), 'card')} with an error: {', '.join(errors)}" if errors else ""


def run(repository: str = CARD_REPO, head: str = "", dry_run: bool = False, card: Optional[str] = None,
        pending_only: bool = False, local: str = "") -> tuple:
    """
    Bake the PSD-kind edits at head and land everything in one commit on the branch's current head (run_report).

    Returns:
        (markdown report, commit shas: one at most); raises on a failure outside a card's steps and the commit
    """
    report = run_report(repository, head, dry_run, card, pending_only, local)
    return report.text, [report.commit] if report.commit else []


def run_report(repository: str = CARD_REPO, head: str = "", dry_run: bool = False, card: Optional[str] = None,
               pending_only: bool = False, local: str = "") -> Report:
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

    Returns:
        The Report, its text rendered and written to the work dir's report.md
    """
    started = perf_counter()
    token_from_env()                            # the Action has no config.py: github_api's fetches take the token from the environment
    work = _work_dir()
    local = abspath(expanduser(local)) if local else ""
    snapshot = local_snapshot(repository, local, head) if local else snapshot_at(repository, head or ref_head(repository))
    head = snapshot.head
    report = Report(repository, head, dry_run, work, local=local)
    report.lines.append(f"- Work dir: `{work}`")
    if snapshot.edits_path is None:
        report.note = f"No {EDITS_PATHS[0]} at `{head[:8]}`."
    else:
        text = snapshot.fetch(snapshot.sha(snapshot.edits_path)).decode("utf-8")
        try:
            edits = CardEdits.parse(text)
        except ValueError as error:
            edits = None
            report.note = f"{snapshot.edits_path} isn't JSON: {error}"
        if edits is not None:
            cards = select(edits, snapshot, card)
            named = len(cards)
            if pending_only:
                cards = [c for c in cards if not final(edits, snapshot, c)]
            report.lines.append(f"- Edits: `{snapshot.edits_path}` `{snapshot.sha(snapshot.edits_path)[:8]}`: "
                                f"{plural(len(edits.cards), 'card')}, {named} with PSD fields"
                                + (f" (--card {card})" if card else "")
                                + (f", {named - len(cards)} refused before (not re-checked)" if named > len(cards) else ""))
            if card and not cards:
                report.note = f"No entry with a PSD field for {card!r}."
            if cards:
                fonts = _fonts_dir(snapshot)    # only now: no card to bake fills no fonts cache
                engine = "text engine" if psd_edit.text_engine() else "no text engine"
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

                def recent() -> Optional[set]:
                    if not pushed:
                        pushed.append(recent_paths(repository, head, datetime.now(timezone.utc)))
                    return pushed[0]
                for one in cards:
                    baked = sum(1 for c in cards if c.baked)
                    if baked >= MAX_BAKES or perf_counter() - started >= BUDGET.total_seconds():
                        one.refusal = Refusal("capped", f"{baked} cards baked in {(perf_counter() - started) / 60:.0f} min "
                                                        f"(at most {MAX_BAKES} or {BUDGET.seconds // 60} min a run)")
                        continue
                    bake_card(snapshot, one, parser, fonts, chart, work, recent)
                waiting = sum(1 for c in cards if c.refusal and c.refusal.code == "capped")
                if waiting:
                    report.lines.append(f"- Capped: {plural(waiting, 'card')} wait for the next run")
                report.cards = cards
                land_cards(report, dry_run, snapshot if local else None)
    report.seconds = perf_counter() - started
    report.text = render(report)
    with open(join(work, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.text)
    return report


def pending(repository: str = CARD_REPO, head: str = "") -> list:
    """
    The schedule's change gate (card-edits.yml): the cards a live run has work for at head, i.e. a PSD-kind edit
    without a final verdict (final(); build_site.apply_edits shows a refusal by the same rule); a baked field leaves the
    file. So new, changed and redundant edits, try-again refusals (recent, head-moved, edit-changed, capped, runner
    errors), refused cards whose PSD changed since, `font` refusals whose MDW/Fonts changed and `duplicate` or no-PNG
    refusals whose PNGs changed are pending, and after a complete live run nothing is.

    Returns:
        Card names in the file's order ([] when nothing is pending or the file is missing); raises ValueError when the
        file isn't JSON (the gate step fails and Discord is told)
    """
    token_from_env()
    snapshot = snapshot_at(repository, head or ref_head(repository))
    if snapshot.edits_path is None:
        return []
    edits = CardEdits.parse(snapshot.fetch(snapshot.sha(snapshot.edits_path)).decode("utf-8"))
    return [card.name for card in select(edits, snapshot) if not final(edits, snapshot, card)]


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
    options = parser.parse_args(args)
    report = run_report(options.repo, options.head, options.dry_run, options.card, options.pending, options.local)
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
    'dump_edits',
    'editors',
    'failure',
    'fetch_verified',
    'final',
    'land',
    'land_cards',
    'local_snapshot',
    'main',
    'pending',
    'psd_fields',
    'recent_paths',
    'render',
    'rewrite_edits',
    'rewrite_entry',
    'run',
    'run_report',
    'select',
    'share_of',
    'vocabulary',
]
