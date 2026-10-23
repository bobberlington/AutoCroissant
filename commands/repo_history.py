from json import dump, load
from os import environ, remove, replace
from os.path import abspath, exists, expanduser, isdir
from re import IGNORECASE, compile as re_compile
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
SSH_CACHE_SECONDS = 60                          # core.sshCommand lookups reused this long
_PLINK = re_compile(r"^(\"[^\"]*[/\\]|'[^']*[/\\]|\S*[/\\])?(tortoise)?plink(\.exe)?[\"']?(\s|$)", IGNORECASE)   # takes no -o
_HAS_BATCH = re_compile(r"batchmode(\s*=|\s+(yes|no)\b)", IGNORECASE)     # BatchMode set on purpose

# ========================
# Global State
# ========================
_paths: dict = {}                               # repo path -> (blob sha, last commit that changed it)
_head = ""                                      # main's commit _paths describes
_loaded = False
_lock = Lock()
_ssh_cache: dict = {}                           # dir -> (time, core.sshCommand)


# ========================
# Non-interactive git (every git call the bot makes)
# ========================
def _configured_ssh(where: str) -> Optional[str]:
    """core.sshCommand as git reads it in where (repo, global, system); '' when unset, None when git config fails."""
    now = monotonic()
    hit = _ssh_cache.get(where)
    if hit and now - hit[0] < SSH_CACHE_SECONDS:
        return hit[1]
    exists_dir = isdir(where)
    try:
        out = run(["git", *(["-C", where] if exists_dir else []), "config", "--get", "core.sshCommand"],
                  capture_output=True, encoding="utf-8", timeout=10, env={**environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, SubprocessError):
        return None                             # unknown: not cached
    command = out.stdout.strip() if out.returncode == 0 else ""
    if exists_dir or not where:                 # a missing dir isn't cached: its repo config comes later
        _ssh_cache[where] = (now, command)
    return command


def ssh_command(where: str = "", env: Optional[dict] = None) -> Optional[str]:
    """
    The ssh command git would run in where, with BatchMode added.

    Args:
        where: The repo dir ('' or a missing dir: the current dir's config)
        env: The environment git gets (default os.environ)

    Returns:
        Git's order: GIT_SSH_COMMAND, else core.sshCommand, else GIT_SSH, else ssh, keeping their options (-i keys);
        None for a GIT_SSH program (takes no options) or a failed config lookup;
        a plink command or an explicit BatchMode is left as is
    """
    env = environ if env is None else env
    command = env.get("GIT_SSH_COMMAND", "").strip() or _configured_ssh(abspath(where) if where else "")
    if command is None:                         # lookup failed: git keeps its own config
        return None
    if not command:
        if env.get("GIT_SSH"):
            return None
        command = "ssh"
    if _PLINK.match(command) or _HAS_BATCH.search(command):
        return command
    return f"{command} {BATCH_MODE}"


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
def _git(*args: str, **extra: str) -> str:
    """Run git in the history clone (blobless: only rev-list, ls-tree, merge-base, fetch here, never diff/log)."""
    clone = expanduser(HISTORY_CLONE)
    return run(["git", "-C", clone, *args], capture_output=True, encoding="utf-8",
               check=True, timeout=GIT_TIMEOUT, env=batch_env(clone, **extra)).stdout


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
    return raw_url_at(repository, path, pinned_ref(repository, path, blob) or CARD_BRANCH)


def raw_url_at(repository: str, path: str, ref: str) -> str:
    """A raw link to path at a commit (or branch)."""
    return f"{RAW_BASE}{repository}/{ref}/{quote(path, safe='/')}"


def commit_before(repository: str, path: str, end: float) -> Optional[tuple]:
    """
    The newest commit on the card branch that changed path before end (/query as_of's image of that day), from the
    history clone (rev-list over its trees: no blob is read, none fetched).

    Returns:
        (commit sha, its committer time, whether a later commit changed path), or None: no commit that old, another
        repository than the clone's, or no readable clone (one console line)
    """
    if repository != REPOSITORY or not isdir(expanduser(HISTORY_CLONE)):
        return None
    try:
        rows = _git("rev-list", "--timestamp", CARD_BRANCH, "--", path, GIT_NO_LAZY_FETCH="1").split()
    except (OSError, SubprocessError) as e:
        print(f"Repo history: no commit of {path} looked up ({e})")
        return None
    for i in range(0, len(rows) - 1, 2):
        if float(rows[i]) < end and _has(rows[i + 1], path):      # not the commit that deleted it
            return (rows[i + 1], float(rows[i]), i > 0)
    return None


def _has(commit: str, path: str) -> bool:
    """Whether path is in commit's tree (ls-tree: no blob read)."""
    try:
        return bool(_git("ls-tree", "-z", commit, "--", path, GIT_NO_LAZY_FETCH="1").strip("\0"))
    except (OSError, SubprocessError):
        return False


# ========================
# Module Exports
# ========================
__all__ = [
    'BATCH_MODE',
    'batch_env',
    'batch_overrides',
    'commit_before',
    'pinned_ref',
    'raw_url',
    'raw_url_at',
    'refresh',
    'ssh_command',
]
