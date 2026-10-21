"""
The edit pipeline's driver (docs/EDIT_PIPELINE_DESIGN.md sections 4, 5, 8-11): bake the PSD-kind edits of
TTSCardMaker's site/Card Edits.json into the card PSDs and PNGs, one commit per card.

    python -m commands.card_bake --repo MichaelJSr/TTSCardMaker --head <sha> [--dry-run] [--card <Name>]
    run(repository, head, dry_run, card) -> (markdown report, commit shas)

Input: one tree snapshot at head and blobs by sha (the card folders are never checked out). Per card named in the
file with a PSD-kind field: refuse `recent` (PSD committed in the last 30 min), fetch its PSD and official PNG (gate
`blob`), classify the edits against a fresh offline parse (psd_edit.classify), apply them with every gate
(psd_edit.apply_jobs), rewrite the file with the baked and redundant fields removed and the `_psd` verdict written
(a Python port of edit.js editsFile), and commit PSD + PNG + file on the branch's current head: GitHub API, or git
objects in the history clone pushed over SSH when the token can't write. Refusals are data: one bad card never
blocks another; refused verdicts land in one file-only commit. The report goes to stdout, the artifacts (edited
PSD, merged PNG, before/after/diff crops, the rewritten file) to the work dir. The runner supplies the token
(GITHUB_TOKEN, or config.GIT_TOKEN on the bot), the fonts (PSD_TOOLS_FONT_DIRS) and the work dir (RUNNER_TEMP).
No Discord or config imports; never the pickles; --dry-run commits nothing. GitHub, git and runner failures (a 5xx,
a rate limit, no text engine) are reported but never written as a verdict. The `font` gate refuses only where a face
resolves to nothing: on a Mac the platform fonts stand in for a missing or empty fonts dir (psd_edit.OK_FONTS).
"""
from argparse import ArgumentParser
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from json import dumps, loads
from os import environ, makedirs, pathsep
from os.path import basename, dirname, expanduser, isdir, join, splitext
from re import IGNORECASE, escape, fullmatch
from subprocess import CalledProcessError, run as run_process
from tempfile import mkdtemp
from time import perf_counter
from traceback import format_exc
from typing import Callable, Optional
from unicodedata import combining, normalize

from PIL import Image
from psd_tools import PSDImage, __version__ as PSD_TOOLS_VERSION
from requests import RequestException, request

import commands.github_api as github_api
import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import card_name
from commands.github_api import api_url, fetch_blob, get_with_retry
from commands.psd_edit import Refusal, Refused
from commands.psd_parser import CardValidator
from commands.repo_history import refresh
from global_config import (CARD_BRANCH, CARD_REPO, EDITS_PATHS, HISTORY_CLONE, ICON_MANIFEST_PATH, LOCAL_DIR_LOC,
                           STARS_DIR, TYPES_DIR)

# ========================
# Configuration
# ========================
RECENT = timedelta(minutes=30)                  # a PSD committed more recently is its author's: don't race them
COMMIT_TRIES = 3                                # main moved meanwhile: redo step 10 on the new head
CONFLICT = (409, 422)                           # ref update refused (not a fast forward)
OWN_SUBJECT = "Card edit"                       # the pipeline's own commits start so: nothing new to bake on them
AUTHOR = {"name": "AutoCroissant", "email": "autocroissant@gmail.com"}      # as the PICKLE and TTS save commits
GIT_IDENTITY = {"GIT_AUTHOR_NAME": AUTHOR["name"], "GIT_AUTHOR_EMAIL": AUTHOR["email"],
                "GIT_COMMITTER_NAME": AUTHOR["name"], "GIT_COMMITTER_EMAIL": AUTHOR["email"]}
TRANSIENT = ("recent", "head-moved", "edit-changed")   # try-again refusals: not a verdict for the editor
GIT_TIMEOUT = 300
API_TIMEOUT = 300                               # a PSD blob is up to 85 MB (113 MB as base64)
EXCLUDE_FOLDERS = ("Markers", "MDW")            # not cards (repo_traverser.EXCLUDE_FOLDERS)
FONT_DIRS_ENV = "PSD_TOOLS_FONT_DIRS"           # the runner's fonts dir(s), pathsep-separated; else the checkout's MDW/Fonts
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
# Global State
# ========================
_cant_write = False                             # POST git/blobs got a 403: commit over SSH until a restart


# ========================
# Data
# ========================
class Transient(RuntimeError):
    """A GitHub, git or runner failure: the card's error is reported, not written as a verdict (the next run retries)."""


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
    commit: Optional[str] = None
    dropped: list = field(default_factory=list)     # fields removed from the file
    verdict: Optional[dict] = None              # the _psd meta written for it (refused cards)
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
    note: str = ""                              # why the run stopped early, if it did
    cards: list = field(default_factory=list)
    verdicts: Optional[str] = None              # the refused verdicts' commit
    traceback: str = ""                         # when the verdicts commit failed outside a card's steps
    seconds: float = 0.0


class Snapshot:
    """The card repo at one commit: the tree, the PSDs by card key and where the edits file is."""

    def __init__(self, repository: str, head: str, info: dict, tree: dict):
        self.repository, self.head, self.info, self.tree = repository, head, info, tree
        self.lower = {path.lower() for path in tree}
        self.psds: dict = {}
        for path in tree:
            if path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/")):
                self.psds[CardAuthors.key(card_name(path))] = path
        self.edits_path = next((p for p in EDITS_PATHS if p in tree), None)

    def sha(self, path: str) -> str:
        return self.tree[path][1]

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
def _token() -> str:
    return github_api.GIT_TOKEN or environ.get("GITHUB_TOKEN", "")


def _authenticate() -> None:
    """The Action has no config.py: github_api's fetches take the token from the environment."""
    if not github_api.GIT_TOKEN and environ.get("GITHUB_TOKEN"):
        github_api.HEADERS["Authorization"] = f"token {environ['GITHUB_TOKEN']}"


def _status(reply) -> str:
    try:
        message = reply.json().get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return f"GitHub {reply.status_code}" + (f": {message}" if message else "")


def _api(repository: str, method: str, path: str, check: bool = True, **kwargs):
    """A GitHub API call on the card repo; check: raise on an error status, with GitHub's message."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if _token():
        headers["Authorization"] = f"token {_token()}"
    if method == "GET":                         # 429 and 5xx retried as the bot's other GETs are
        reply = get_with_retry(api_url(repository, path), headers=headers, timeout=API_TIMEOUT, **kwargs)
    else:
        reply = request(method, api_url(repository, path), headers=headers, timeout=API_TIMEOUT, **kwargs)
    if check and not reply.ok:
        raise Transient(f"{_status(reply)} (on {method} {path})")
    return reply


def _ref_head(repository: str) -> str:
    return _api(repository, "GET", f"git/ref/heads/{CARD_BRANCH}").json()["object"]["sha"]


def _commit_info(repository: str, sha: str) -> dict:
    """GET git/commits/{sha}: its tree, author and message."""
    return _api(repository, "GET", f"git/commits/{sha}").json()


def _tree(repository: str, tree_sha: str) -> dict:
    """{path: (mode, blob sha)} of a whole tree."""
    reply = get_with_retry(api_url(repository, f"git/trees/{tree_sha}?recursive=1"), headers=github_api.HEADERS)
    if reply.status_code != 200:
        raise Transient(f"{_status(reply)} (on GET git/trees/{tree_sha[:8]})")
    data = reply.json()
    if data.get("truncated"):
        raise Transient(f"tree {tree_sha[:8]} is truncated")
    return {entry["path"]: (entry.get("mode", "100644"), entry["sha"]) for entry in data.get("tree", [])
            if entry.get("type") == "blob"}


def _own_commit(info: dict) -> bool:
    return (info.get("author") or {}).get("name") == AUTHOR["name"] and (info.get("message") or "").startswith(OWN_SUBJECT)


def blob_sha(data: bytes) -> str:
    """git's blob sha: sha1("blob N\\0" + data)."""
    return sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_verified(repository: str, sha: str, path: str = "") -> bytes:
    """
    A blob's bytes by sha, checked against the tree (gate blob) and written to path when given.

    Returns:
        The bytes; raises Refused("blob") when they don't hash to sha
    """
    data = fetch_blob(repository, sha, timeout=API_TIMEOUT).content
    if blob_sha(data) != sha:
        raise Refused("blob", f"blob {sha[:8]} hashes to {blob_sha(data)[:8]}")
    if path:
        makedirs(dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
    return data


def _recent(repository: str, path: str, now: datetime) -> Optional[str]:
    """The sha of a commit touching path within RECENT of now, else None (one small request)."""
    since = (now - RECENT).strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = _api(repository, "GET", "commits", params={"path": path, "since": since, "per_page": 1}).json()
    return commits[0].get("sha", "?") if commits else None


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
def _git(where: str, *args: str, stdin: Optional[bytes] = None, env: Optional[dict] = None) -> bytes:
    """git in the blobless history clone: refs, trees and our own objects, never diff or log."""
    return run_process(["git", "-C", where, *args], input=stdin, capture_output=True, check=True, timeout=GIT_TIMEOUT,
                       env=env).stdout


def _git_env() -> dict:
    env = {**environ, "GIT_TERMINAL_PROMPT": "0", **GIT_IDENTITY}
    if "GIT_SSH" not in env and "GIT_SSH_COMMAND" not in env:
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
    return env


def _git_tree(clone: str, tree: Optional[str], changes: dict, env: dict) -> str:
    """
    Write a tree: `tree` (None for a new folder) with changes applied, folder by folder.

    Args:
        changes: {entry name: blob sha, or {nested changes} for a folder}

    Returns:
        The new tree's sha (--missing: siblings' blobs aren't in the blobless clone)
    """
    entries: dict = {}
    if tree:
        for line in _git(clone, "ls-tree", "-z", tree, env=env).decode("utf-8").split("\0"):
            if line:
                meta, _, name = line.partition("\t")
                mode, kind, sha = meta.split(" ")
                entries[name] = (mode, kind, sha)
    for name, change in changes.items():
        old = entries.get(name)
        if isinstance(change, dict):
            entries[name] = ("040000", "tree", _git_tree(clone, old[2] if old and old[1] == "tree" else None, change, env))
        else:
            entries[name] = (old[0] if old and old[1] == "blob" else "100644", "blob", change)
    data = "".join(f"{mode} {kind} {sha}\t{name}\0" for name, (mode, kind, sha) in entries.items())
    return _git(clone, "mktree", "-z", "--missing", stdin=data.encode("utf-8"), env=env).decode().strip()


def _git_commit(repository: str, head: str, files: dict, message: str) -> Optional[str]:
    """
    Commit files on head in the history clone and push to the branch over SSH (no force).

    Args:
        files: {repo path: bytes}

    Returns:
        The new commit's sha, or None when the push was rejected (the branch moved)
    """
    if repository != CARD_REPO:
        raise Transient(f"the history clone mirrors {CARD_REPO}, not {repository}")
    refresh()                                   # the clone exists and knows head
    clone, env = expanduser(HISTORY_CLONE), _git_env()
    changes: dict = {}
    for path, data in files.items():
        node = changes
        *folders, name = path.split("/")
        for folder in folders:
            node = node.setdefault(folder, {})
        node[name] = _git(clone, "hash-object", "-w", "--stdin", stdin=data, env=env).decode().strip()
    tree = _git_tree(clone, _git(clone, "rev-parse", f"{head}^{{tree}}", env=env).decode().strip(), changes, env)
    commit = _git(clone, "commit-tree", tree, "-p", head, "-m", message, env=env).decode().strip()
    push = run_process(["git", "-C", clone, "push", "--porcelain", "--no-thin", f"git@github.com:{repository}.git",
                        f"{commit}:refs/heads/{CARD_BRANCH}"], capture_output=True, timeout=GIT_TIMEOUT, env=env)
    if push.returncode:
        out = push.stdout.decode("utf-8", "replace")
        if any(line.startswith("!") and "[rejected]" in line for line in out.splitlines()):
            return None
        err = [line for line in (push.stderr.decode("utf-8", "replace") or out).splitlines() if line.strip()]
        raise Transient(f"git push failed: {' / '.join(err[max(0, len(err) - 3):])}"
                           + ("" if environ.get("SSH_AUTH_SOCK") else " (no SSH_AUTH_SOCK: no SSH agent)"))
    refresh()
    return commit


def _upload(repository: str, data: bytes) -> Optional[str]:
    """POST git/blobs: the blob's sha, or None when the token can't write (403: the SSH path takes over)."""
    global _cant_write
    reply = _api(repository, "POST", "git/blobs", check=False,
                 json={"content": b64encode(data).decode("ascii"), "encoding": "base64"})
    if reply.status_code == 403:
        limited = (reply.headers.get("retry-after") or reply.headers.get("x-ratelimit-remaining") == "0"
                   or "rate limit" in reply.text.lower())
        if limited:                             # this run's: not the token's rights
            raise Transient(f"{_status(reply)}: rate limited (on POST git/blobs)")
        _cant_write = True
        return None
    if not reply.ok:
        raise Transient(f"{_status(reply)} (on POST git/blobs)")
    sha = reply.json()["sha"]
    if sha != blob_sha(data):
        raise Transient(f"GitHub stored blob {sha[:8]} for data hashing to {blob_sha(data)[:8]}")
    return sha


def _api_commit(repository: str, head: str, base_tree: str, modes: dict, blobs: dict, message: str) -> Optional[str]:
    """
    One commit through the Git Data API: tree on base_tree, commit on head, non-force ref update.

    Args:
        modes: {repo path: mode} of the paths as the tree has them
        blobs: {repo path: uploaded blob sha}

    Returns:
        The commit's sha, or None on a 409/422 (the branch moved)
    """
    tree = _api(repository, "POST", "git/trees", json={"base_tree": base_tree, "tree": [
        {"path": path, "mode": modes.get(path, "100644"), "type": "blob", "sha": sha} for path, sha in blobs.items()]}).json()["sha"]
    commit = _api(repository, "POST", "git/commits", json={"message": message, "tree": tree, "parents": [head],
                                                           "author": AUTHOR, "committer": AUTHOR}).json()["sha"]
    try:
        reply = _api(repository, "PATCH", f"git/refs/heads/{CARD_BRANCH}", check=False, json={"sha": commit, "force": False})
    except RequestException:                    # no reply: the ref may have moved anyway
        if _ref_head(repository) != commit:
            raise
        return commit
    if reply.status_code in CONFLICT:
        return None
    if not reply.ok:
        raise Transient(f"{_status(reply)} (on PATCH git/refs/heads/{CARD_BRANCH})")
    return commit


def land(repository: str, files: dict, expect: dict, rewrite: Callable[[str], Optional[str]], message: str,
         dry_run: bool, out_path: str = "") -> Optional[str]:
    """
    Step 10 for one commit: re-read the branch head H' and its tree, check the files we replace still hold the blobs
    we read (head-moved), rewrite Card Edits.json from the file at H' (the rewrite raises edit-changed itself), and
    commit on H'; on 409/422 or a rejected push, again from H' up to COMMIT_TRIES times.

    Args:
        files: {repo path: bytes} the PSD and PNG to commit ({} for a file-only commit)
        expect: {repo path: blob sha} what those paths must still hold at H'
        rewrite: edits file text at H' -> new text, or None when nothing is left to write
        message: Commit subject
        dry_run: Write out_path, commit nothing
        out_path: Where the rewritten file goes for the artifacts

    Returns:
        The commit's sha, None when nothing was committed (dry run, or the rewrite returned None);
        raises Refused("head-moved") after COMMIT_TRIES, Refused("edit-changed") from the rewrite
    """
    global _cant_write
    uploaded: dict = {}
    for attempt in range(COMMIT_TRIES):
        head = _ref_head(repository)
        info = _commit_info(repository, head)
        tree = _tree(repository, info["tree"]["sha"])
        for path, sha in expect.items():
            if tree.get(path, ("", ""))[1] != sha:
                raise Refused("head-moved", f"{path} changed on {CARD_BRANCH} `{head[:8]}` meanwhile")
        edits_path = next((p for p in EDITS_PATHS if p in tree), None)
        if edits_path is None:
            raise Refused("head-moved", f"no {EDITS_PATHS[0]} on {CARD_BRANCH} `{head[:8]}`")
        text = fetch_verified(repository, tree[edits_path][1]).decode("utf-8")
        new_text = rewrite(text)
        if new_text is None:
            return None
        if out_path:
            makedirs(dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
        if dry_run:
            return None
        payload = {**files, edits_path: new_text.encode("utf-8")}
        if _token() and not _cant_write:
            for path, data in payload.items():
                if path not in uploaded or path == edits_path:
                    sha = _upload(repository, data)
                    if sha is None:
                        break
                    uploaded[path] = sha
            else:
                modes = {path: tree[path][0] for path in payload if path in tree}
                commit = _api_commit(repository, head, info["tree"]["sha"], modes, uploaded, message)
                if commit:
                    return commit
                continue
        try:
            commit = _git_commit(repository, head, payload, message)
        except CalledProcessError as error:
            raise Transient(f"git {error.cmd[3]} in the history clone failed: "
                               f"{(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
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


def _fonts_dir() -> str:
    dirs = environ.get(FONT_DIRS_ENV, "")
    return dirs.split(pathsep)[0] if dirs else join(expanduser(LOCAL_DIR_LOC), psd_edit.FONTS_DIR)


def _text_engine() -> bool:
    try:
        import psd_tools.composite.text  # noqa: F401
    except ImportError:
        return False
    return True


def vocabulary(snapshot: Snapshot):
    """The icon vocabulary from the tree (Types/, Types/Stars/, the stoogedings manifest), as repo_traverser builds it."""
    all_types = [splitext(basename(p))[0].lower() for p in snapshot.tree
                 if p.startswith(TYPES_DIR) and "Stars" not in p and "." in p]
    stars = [splitext(basename(p))[0].lower().replace(" ", "_") for p in snapshot.tree
             if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
    manifest = None
    if ICON_MANIFEST_PATH in snapshot.tree:
        manifest = loads(fetch_verified(snapshot.repository, snapshot.sha(ICON_MANIFEST_PATH)).decode("utf-8"))
    return psd_edit.offline_parser(all_types, stars, manifest)


def select(edits: CardEdits, snapshot: Snapshot, only: Optional[str] = None) -> list:
    """
    The cards the run looks at: every entry with a PSD-kind field (one card with --card).

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
        cards.append(Card(name, key, wanted, rel, snapshot.sha(rel) if rel else ""))
    return cards


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


def _subject(card: Card) -> str:
    if card.jobs:
        return f"{OWN_SUBJECT}: {card.name} {_describe(card)}"
    return f"{OWN_SUBJECT}: {card.name} " + ", ".join(f"{f} {card.fields[f]!r}" for f in card.redundant) + " already on the card"


def bake_card(snapshot: Snapshot, card: Card, parser, fonts_dir: str, chart: Callable[[], Optional[dict]], work: str,
              dry_run: bool, now: datetime) -> None:
    """
    Section 5 for one card: the recent, duplicate and blob gates, classify, apply with every gate, commit.
    Fills the Card in place; a Refused becomes its refusal, any other exception its "error".
    """
    started = perf_counter()
    folder = join(work, card.name.replace("/", "_"))
    try:
        if not card.rel:
            original = snapshot.duplicate_of(card.key)
            if original:
                raise Refused("duplicate", f"a numbered PNG exported from `{original}`")
            raise Refused("unsupported-card", "no PSD in the card repo")
        recent = _recent(snapshot.repository, card.rel, now)
        if recent:
            raise Refused("recent", f"PSD committed in the last {RECENT.seconds // 60} min (`{recent[:8]}`)")
        card.png, duplicates = snapshot.png_for(card.rel)
        if not card.png:
            raise Refused("unsupported-card", "no PNG beside the PSD")
        if duplicates:
            raise Refused("duplicate", "numbered PNGs share the PSD: " + ", ".join(basename(p) for p in duplicates))
        card.png_sha = snapshot.sha(card.png)
        old_psd, old_png = join(folder, "old", basename(card.rel)), join(folder, "old", basename(card.png))
        fetch_verified(snapshot.repository, card.psd_sha, old_psd)
        fetch_verified(snapshot.repository, card.png_sha, old_png)

        parsed = psd_edit.parse_card(parser, old_psd, card.rel)
        sorted_out = psd_edit.classify(parsed, card.fields, PSDImage.open(old_psd),
                                       card.name in CardValidator.EXCESSIVE_STAT_EXCLUSIONS)
        card.jobs, card.redundant = dict(sorted_out.jobs), list(sorted_out.redundant)
        if sorted_out.refusals:                 # per card all-or-nothing (section 10)
            first, *rest = sorted_out.refusals
            raise Refused(first.code, "; ".join([first.why] + [f"{r.code}: {r.why}" for r in rest]))
        if not card.jobs and not card.redundant:
            return

        verdict = None
        files, expect = {}, {card.rel: card.psd_sha, card.png: card.png_sha}
        if card.jobs:
            if any(f in psd_edit.TEXT_FIELDS for f in card.jobs) and not _text_engine():
                raise Transient("this runner has no text engine (psd-tools[text])")
            colours = chart() if "ability" in card.jobs else None
            if "ability" in card.jobs and colours is None:
                raise Transient(f"no status chart ({psd_edit.STATUS_CHART}) in the tree")
            new_dir = join(folder, "new")
            makedirs(new_dir, exist_ok=True)
            card.baked = psd_edit.apply_jobs(old_psd, old_png, card.rel, card.jobs, parsed, parser, new_dir, fonts_dir,
                                             colours, update_preview=True)
            with open(card.baked.psd_path, "rb") as fh:
                psd_bytes = fh.read()
            with open(card.baked.png_path, "rb") as fh:
                png_bytes = fh.read()
            card.new_psd_sha = blob_sha(psd_bytes)
            files = {card.rel: psd_bytes, card.png: png_bytes}
            crops = psd_edit.artifacts(Image.open(old_png), Image.open(card.baked.png_path), card.baked.region)
            for name, image in crops.items():
                image.save(join(folder, f"{name}.png"))
            with open(join(folder, "parse.json"), "w", encoding="utf-8") as fh:
                fh.write(dumps({"old": parsed["card"], "new": card.baked.reparse["card"], "jobs": card.jobs,
                                "validator": card.baked.reparse["validator"]}, indent=1, ensure_ascii=False))
            verdict = {"state": "baked", "for": card.jobs, "psd": card.new_psd_sha[:8], "reason": _describe(card),
                       "at": _now()}
        card.dropped = [f for f in list(card.jobs) + card.redundant if f != "name"]

        def rewrite(text: str) -> Optional[str]:
            entry = _entry_at(text, card.name)
            if psd_fields(entry or {}) != card.fields:
                raise Refused("edit-changed", f"the edit changed on {CARD_BRANCH} meanwhile: not touched")
            if not files and not card.dropped:  # a name already printed: the entry stays as it is (idempotent reruns)
                meta = CardEdits.parse(text).meta.get(card.key, {}).get("_psd")
                if (meta or {}).get("state") != "refused":
                    return None
            new_text = rewrite_edits(text, card.name, card.dropped, verdict)
            return new_text if new_text is not None or not files else text

        card.commit = land(snapshot.repository, files, expect, rewrite, _subject(card), dry_run,
                           join(folder, basename(EDITS_PATHS[0])))
    except Refused as refused:
        card.refusal = refused.refusal
    except Transient as error:                  # the runner's or GitHub's: reported, no verdict
        card.refusal, card.transient = Refusal("error", str(error)), True
    except Exception as error:                  # noqa: BLE001 - one bad card never blocks another
        card.refusal = Refusal("error", f"{type(error).__name__}: {error}")
        card.transient = isinstance(error, (RequestException, ImportError))   # a 5xx, or the runner lacks scipy / the text extra
        card.traceback = format_exc()
    finally:
        card.seconds = perf_counter() - started


def land_verdicts(snapshot: Snapshot, cards: list, dry_run: bool, work: str) -> Optional[str]:
    """
    The refused verdicts (section 9's _psd) in one file-only commit; unchanged verdicts and transient refusals
    write nothing, cards whose edit changed meanwhile are left out.

    Returns:
        The commit's sha, or None
    """
    refused = [c for c in cards if c.refusal and c.refusal.code not in TRANSIENT and not c.transient]
    for card in refused:
        card.verdict = {"state": "refused", "for": card.fields, "psd": card.psd_sha[:8],
                        "reason": f"{card.refusal.code}: {card.refusal.why}", "at": _now()}
    if not refused:
        return None

    def rewrite(text: str) -> Optional[str]:
        changed = False
        for card in refused:
            entry = _entry_at(text, card.name)
            if entry is None or psd_fields(entry) != card.fields:
                card.note = "edit changed meanwhile: verdict not written"
                continue
            meta = CardEdits.parse(text).meta.get(card.key, {}).get("_psd")
            if _same_verdict(meta, card.verdict):
                continue
            new_text = rewrite_edits(text, card.name, (), card.verdict)
            if new_text is not None:
                text, changed = new_text, True
        return text if changed else None

    return land(snapshot.repository, {}, {}, rewrite, f"{OWN_SUBJECT}s: {len(refused)} refused", dry_run,
                join(work, "verdicts", basename(EDITS_PATHS[0])))


# ========================
# Report
# ========================
def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _card_line(card: Card, dry_run: bool) -> str:
    where = f" `{card.rel}`" if card.rel else ""
    fields = _describe(card) if card.jobs else ", ".join(f"{f} {card.fields[f]!r}" for f in card.fields)
    line = f"**{card.name}**{where}: {fields}"
    if card.refusal:
        line += f": {card.refusal.code}: {card.refusal.why}"
    if card.baked:
        parity = "; ".join("%s %.2f%% bad, mean %.2f" % (f, m["bad_pct"], m["mean"]) for f, m in card.baked.parity.items())
        line += f"; parity {parity}; {card.baked.flag_bytes} flag bytes, {card.baked.changed_pixels} pixels"
    if card.redundant:
        line += f"; already on the card: {', '.join(card.redundant)}" + (" (name stays)" if "name" in card.redundant else "")
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
    lines += report.lines
    if report.note:
        lines += ["", f"**{report.note}**"]
    baked = [c for c in report.cards if c.baked and not c.refusal]
    redundant = [c for c in report.cards if c.redundant and not c.jobs and not c.refusal]
    refused = [c for c in report.cards if c.refusal and c.refusal.code != "error"]
    errors = [c for c in report.cards if c.refusal and c.refusal.code == "error"]
    if report.cards and not baked:
        lines += ["", "**Nothing baked.**"]

    def section(title, cards):
        if cards:
            lines.extend(["", f"## {title} ({len(cards)})"] + [f"- {_card_line(c, report.dry_run)}" for c in cards])
    section("Baked", baked)
    section("Redundant (dropped from the file)", redundant)
    section("Refused", refused)
    section("Errors", errors)
    for card in errors:
        if card.traceback:
            lines += ["", f"<details><summary>{card.name}</summary>", "", "```", card.traceback.rstrip(), "```", "</details>"]
    if report.verdicts:
        lines += ["", f"Refused verdicts committed `{report.verdicts[:8]}`."]
    if report.traceback:
        lines += ["", "<details><summary>verdicts</summary>", "", "```", report.traceback.rstrip(), "```", "</details>"]
    lines += ["", f"Took {report.seconds:.0f} s."]
    return "\n".join(lines) + "\n"


# ========================
# Public API Functions
# ========================
def run(repository: str = CARD_REPO, head: str = "", dry_run: bool = False, card: Optional[str] = None) -> tuple:
    """
    Bake the PSD-kind edits at head: one commit per card on the branch's current head, the refused verdicts in one.

    Args:
        repository: owner/name of the card repo
        head: The commit whose tree and edits file are the snapshot ("" = the branch head)
        dry_run: Write the artifacts, commit nothing
        card: Restrict the run to one card (matched like the authors file)

    Returns:
        (markdown report, commit shas); raises on a failure outside a card's steps
    """
    started = perf_counter()
    _authenticate()
    work = _work_dir()
    head = head or _ref_head(repository)
    info = _commit_info(repository, head)
    report = Report(repository, head, dry_run, work)
    report.lines.append(f"- Work dir: `{work}`")
    commits: list = []
    if _own_commit(info):
        report.note = "The head is the pipeline's own commit: nothing new to bake."
    else:
        snapshot = Snapshot(repository, head, info, _tree(repository, info["tree"]["sha"]))
        if snapshot.edits_path is None:
            report.note = f"No {EDITS_PATHS[0]} at `{head[:8]}`."
        else:
            text = fetch_verified(repository, snapshot.sha(snapshot.edits_path)).decode("utf-8")
            try:
                edits = CardEdits.parse(text)
            except ValueError as error:
                edits = None
                report.note = f"{snapshot.edits_path} isn't JSON: {error}"
            if edits is not None:
                cards = select(edits, snapshot, card)
                report.lines.append(f"- Edits: `{snapshot.edits_path}` `{snapshot.sha(snapshot.edits_path)[:8]}`: "
                                    f"{_plural(len(edits.cards), 'card')}, {len(cards)} with PSD fields"
                                    + (f" (--card {card})" if card else ""))
                fonts = _fonts_dir()
                engine = "text engine" if _text_engine() else "no text engine"
                report.lines.append(f"- Runner: psd-tools {PSD_TOOLS_VERSION} ({engine}); fonts `{fonts}`"
                                    + ("" if isdir(fonts) else " (missing)"))
                if card and not cards:
                    report.note = f"No entry with a PSD field for {card!r}."
                if cards:
                    parser = vocabulary(snapshot)
                    colours: list = []

                    def chart() -> Optional[dict]:
                        if not colours:
                            if psd_edit.STATUS_CHART not in snapshot.tree:
                                colours.append(None)
                            else:
                                path = join(work, basename(psd_edit.STATUS_CHART))
                                fetch_verified(repository, snapshot.sha(psd_edit.STATUS_CHART), path)
                                colours.append(psd_edit.status_fills(PSDImage.open(path)))
                        return colours[0]
                    now = datetime.now(timezone.utc)
                    for one in cards:
                        bake_card(snapshot, one, parser, fonts, chart, work, dry_run, now)
                        if one.commit:
                            commits.append(one.commit)
                    report.cards = cards
                    try:                        # fenced like a card's steps: the bake commits above have landed
                        report.verdicts = land_verdicts(snapshot, cards, dry_run, work)
                    except Refused as refused:
                        report.note = f"Refused verdicts not committed: {refused.code}: {refused.why}"
                    except Exception as error:  # noqa: BLE001
                        report.note = f"Refused verdicts not committed: {type(error).__name__}: {error}"
                        if not isinstance(error, (Transient, RequestException)):
                            report.traceback = format_exc()
                    if report.verdicts:
                        commits.append(report.verdicts)
    report.seconds = perf_counter() - started
    text = render(report)
    with open(join(work, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text, commits


def main(args: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.card_bake",
                            description="Bake the PSD-kind edits of site/Card Edits.json into the card PSDs and PNGs, "
                                        "one commit per card (docs/EDIT_PIPELINE_DESIGN.md).")
    parser.add_argument("--repo", default=CARD_REPO, help=f"owner/name of the card repo (default {CARD_REPO})")
    parser.add_argument("--head", default="", help=f"commit to read the tree and edits file at (default: {CARD_BRANCH}'s head)")
    parser.add_argument("--dry-run", action="store_true", help="write the artifacts, commit nothing")
    parser.add_argument("--card", default=None, help="only this card (matched like the authors file)")
    options = parser.parse_args(args)
    text, _ = run(options.repo, options.head, options.dry_run, options.card)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'Card',
    'ORDER',
    'Report',
    'Snapshot',
    'Transient',
    'bake_card',
    'blob_sha',
    'dump_edits',
    'fetch_verified',
    'land',
    'land_verdicts',
    'main',
    'psd_fields',
    'render',
    'rewrite_edits',
    'rewrite_entry',
    'run',
    'select',
    'vocabulary',
]
