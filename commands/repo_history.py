from functools import lru_cache
from json import dump, load
from os import environ, remove, replace
from os.path import abspath, exists, expanduser, isdir
from shutil import rmtree
from subprocess import CalledProcessError, SubprocessError, run
from threading import Lock
from time import monotonic
from typing import Optional
from urllib.parse import quote

from global_config import CARD_BRANCH, CARD_REPO, HISTORY_CLONE, HISTORY_JSON, RAW_BASE

# ========================
# Configuration
# ========================
REPOSITORY = CARD_REPO                          # the card repo the history clone mirrors
GIT_TIMEOUT = 300                               # seconds per git call
BATCH_MODE = "-o BatchMode=yes"                 # ssh fails at once instead of prompting for a key or host

# ========================
# Global State
# ========================
_paths: dict = {}                               # repo path -> (blob sha, last commit that changed it)
_head = ""                                      # main's commit _paths describes
_loaded = False
_lock = Lock()


# ========================
# Non-interactive git (every git call the bot makes)
# ========================
@lru_cache(maxsize=None)
def _configured_ssh(where: str) -> str:
    """core.sshCommand as git reads it in where (repo, global, system); '' when unset."""
    try:
        out = run(["git", *(["-C", where] if isdir(where) else []), "config", "--get", "core.sshCommand"],
                  capture_output=True, encoding="utf-8", timeout=10, env={**environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def ssh_command(where: str = "", env: Optional[dict] = None) -> Optional[str]:
    """
    The ssh command git would run in where, with BatchMode added.

    Args:
        where: The repo dir ('' or a missing dir: the current dir's config)
        env: The environment git gets (default os.environ)

    Returns:
        GIT_SSH_COMMAND, else core.sshCommand, else ssh, keeping their options (-i keys); None when GIT_SSH names a
        program, which takes no options and wins over core.sshCommand
    """
    env = environ if env is None else env
    command = env.get("GIT_SSH_COMMAND", "").strip()
    if not command:
        if env.get("GIT_SSH"):
            return None
        command = _configured_ssh(abspath(where) if where else "") or "ssh"
    if "batchmode" not in command.lower():     # an explicit BatchMode stays as set
        command = f"{command} {BATCH_MODE}"
    return command


def batch_overrides(where: str = "", env: Optional[dict] = None) -> dict:
    """The variables that make git non-interactive: no credential prompt, ssh in BatchMode (SSH_AUTH_SOCK untouched)."""
    overrides = {"GIT_TERMINAL_PROMPT": "0"}
    command = ssh_command(where, env)
    if command:
        overrides["GIT_SSH_COMMAND"] = command
    return overrides


def batch_env(where: str = "", **extra: str) -> dict:
    """os.environ plus batch_overrides(where) plus extra: the env for a git subprocess."""
    return {**environ, **batch_overrides(where), **extra}


# ========================
# Helper Functions
# ========================
def _git(*args: str) -> str:
    """Run git in the history clone (blobless: only rev-list, ls-tree, merge-base, fetch here, never diff/log)."""
    clone = expanduser(HISTORY_CLONE)
    return run(["git", "-C", clone, *args], capture_output=True, encoding="utf-8",
               check=True, timeout=GIT_TIMEOUT, env=batch_env(clone)).stdout


def _tree(commit: str) -> dict:
    """A commit's files: path -> blob sha."""
    files = {}
    for entry in _git("ls-tree", "-r", "-z", commit).split("\0"):
        meta, _, path = entry.partition("\t")
        parts = meta.split(" ")
        if len(parts) == 3 and parts[1] == "blob":
            files[path] = parts[2]
    return files


def _is_ancestor(commit: str, head: str) -> bool:
    """Whether commit is in head's history (not after a force push)."""
    try:
        _git("merge-base", "--is-ancestor", commit, head)
        return True
    except CalledProcessError:
        return False


def _walk(head: str, last: str, known: dict) -> Optional[dict]:
    """
    The paths at head with the commit that last changed each, walking only commits after last.

    Args:
        head: Commit to describe
        last: Commit known describes ("" walks all of history)
        known: Map at last

    Returns:
        {path: (blob, commit)}, or None when head's history doesn't contain last (force push)
    """
    rows = [line.split() for line in _git("rev-list", "--parents", "--topo-order", "--reverse",
                                          f"{last}..{head}" if last else head).splitlines()]
    if not rows:
        return known if head == last else None
    new = {row[0] for row in rows}
    maps = {last: known} if last else {}
    children: dict = {}
    for row in rows:
        for parent in row[1:]:
            if parent not in new and parent not in maps:    # older than last: a merged branch forked before it
                if not _is_ancestor(last, head):
                    return None                 # force push: walk it all
                # a commit holding each blob is enough for a pinned link: last's where it's the same file
                maps[parent] = {path: known[path] if path in known and known[path][0] == blob else (blob, parent)
                                for path, blob in _tree(parent).items()}
            children[parent] = children.get(parent, 0) + 1

    for commit, *parents in rows:               # parents come first
        olds = [maps[p] for p in parents]
        current = {}
        for path, blob in _tree(commit).items():
            # unchanged when it matches any parent (first match, like git log); else changed here
            current[path] = next(((blob, old[path][1]) for old in olds if path in old and old[path][0] == blob),
                                 (blob, commit))
        maps[commit] = current
        for p in parents:
            children[p] -= 1
            if not children[p]:
                del maps[p]
    return maps.get(head)


def _load() -> None:
    """Read the saved map once."""
    global _paths, _head, _loaded
    _loaded = True
    if not exists(HISTORY_JSON):
        return
    try:
        with open(HISTORY_JSON, encoding="utf-8") as f:
            data = load(f)
        _paths, _head = {p: tuple(v) for p, v in data["paths"].items()}, data["head"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Could not read {HISTORY_JSON}, rebuilding it: {e}")


def _save() -> None:
    """Write the map atomically."""
    tmp = f"{HISTORY_JSON}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            dump({"head": _head, "paths": _paths}, f)
        replace(tmp, HISTORY_JSON)
    finally:
        if exists(tmp):
            remove(tmp)


# ========================
# Public API Functions
# ========================
def refresh() -> bool:
    """Fetch the history clone and walk new commits. Never raises; False on failure (the old map stays)."""
    global _paths, _head
    with _lock:
        try:
            if not _loaded:
                _load()
            clone = expanduser(HISTORY_CLONE)
            if exists(clone):
                _git("fetch", "--quiet", "origin", f"+refs/heads/{CARD_BRANCH}:refs/heads/{CARD_BRANCH}")
            else:                                   # clone beside it, then rename: no half clone left behind
                rmtree(f"{clone}.tmp", ignore_errors=True)
                run(["git", "clone", "--quiet", "--bare", "--filter=blob:none",
                     f"https://github.com/{REPOSITORY}.git", f"{clone}.tmp"], capture_output=True, encoding="utf-8",
                    check=True, timeout=GIT_TIMEOUT, env=batch_env())
                replace(f"{clone}.tmp", clone)
            head = _git("rev-parse", CARD_BRANCH).strip()
            if head == _head:
                return True
            start = monotonic()
            paths = None
            if _head:
                try:
                    paths = _walk(head, _head, _paths)
                except CalledProcessError:          # old head gone from the clone
                    pass
            if paths is None:
                paths = _walk(head, "", {})
            if paths is None:
                raise ValueError(f"no history walked to {head}")
            _paths, _head = paths, head
            _save()
            print(f"Repo history: {len(paths)} files at {head[:8]} ({monotonic() - start:.1f}s)")
            return True
        except Exception as e:
            print(f"Repo history refresh failed, keeping the old links: {e} {getattr(e, 'stderr', '') or ''}")
            return False


def pinned_ref(repository: str, path: str, blob: Optional[str] = None) -> Optional[str]:
    """The commit that last changed path, when blob is the blob it has there; else None (unknown file, or newer)."""
    found = _paths.get(path) if repository == REPOSITORY else None
    return found[1] if found and blob and found[0] == blob else None


def raw_url(repository: str, path: str, blob: Optional[str] = None) -> str:
    """A raw link pinned to the commit that last changed path, if blob matches it; else the branch link."""
    return f"{RAW_BASE}{repository}/{pinned_ref(repository, path, blob) or CARD_BRANCH}/{quote(path, safe='/')}"


# ========================
# Module Exports
# ========================
__all__ = [
    'BATCH_MODE',
    'batch_env',
    'batch_overrides',
    'pinned_ref',
    'raw_url',
    'refresh',
    'ssh_command',
]
