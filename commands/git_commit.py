"""
One commit on the card repo's branch, shared by the edit pipeline (card_bake) and the TTS save updater (tts_save):
blobs, tree and commit through the Git Data API with a non-force ref update; when the token can't write (NO_WRITE on
POST git/blobs, or no token) the same objects are written into the blobless history clone and pushed over SSH.

    commit_files(repository, head, base_tree, modes, {path: bytes}, message) -> sha, or None when the branch moved

A rate-limited 403 is this run's, not the token's: it raises Transient (ssh_on_limit: SSH for the rest of the run). No Discord or
config imports (the token: config through github_api, else GITHUB_TOKEN); never the pickles.
"""
from base64 import b64encode
from os import environ
from os.path import expanduser
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from typing import Optional

from requests import RequestException, request

import commands.github_api as github_api
from commands.github_api import api_url, blob_sha, get_with_retry
from commands.repo_history import batch_env, refresh
from global_config import CARD_BRANCH, CARD_REPO, HISTORY_CLONE

# ========================
# Configuration
# ========================
AUTHOR = {"name": "AutoCroissant", "email": "autocroissant@gmail.com"}      # as the PICKLE commits
GIT_IDENTITY = {"GIT_AUTHOR_NAME": AUTHOR["name"], "GIT_AUTHOR_EMAIL": AUTHOR["email"],
                "GIT_COMMITTER_NAME": AUTHOR["name"], "GIT_COMMITTER_EMAIL": AUTHOR["email"]}
COMMIT_TRIES = 3                                # the branch moved meanwhile: callers redo their step on the new head
CONFLICT = (409, 422)                           # ref update refused (not a fast forward)
NO_WRITE = (401, 403, 404)                      # blob upload refused: the token can't write (classic PATs get 404)
GIT_TIMEOUT = 300
API_TIMEOUT = 300                               # a PSD blob is up to 85 MB (113 MB as base64)

# ========================
# Global State
# ========================
cant_write = False                              # POST git/blobs got a NO_WRITE status: commit over SSH until a restart
limited = False                                 # rate-limited with ssh_on_limit: SSH for the rest of the run (the caller resets it)


# ========================
# Data
# ========================
class Transient(RuntimeError):
    """A GitHub, git or runner failure: reported, never written as a verdict (the next run retries)."""


# ========================
# GitHub API
# ========================
def token() -> str:
    return github_api.GIT_TOKEN or environ.get("GITHUB_TOKEN", "")


def status(reply) -> str:
    """'GitHub 422: <its message>'."""
    try:
        message = reply.json().get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return f"GitHub {reply.status_code}" + (f": {message}" if message else "")


def api(repository: str, method: str, path: str, check: bool = True, raw: bool = False, timeout: int = API_TIMEOUT,
        **kwargs):
    """
    A GitHub API call on a repo.

    Args:
        repository: owner/name
        method: GET (retried on 429 and 5xx as the bot's other GETs are), POST, PATCH...
        path: After /repos/<repository>/
        check: Raise Transient on an error status, with GitHub's message
        raw: Accept a blob's bytes instead of its JSON envelope

    Returns:
        The response
    """
    headers = {"Accept": github_api.RAW_MEDIA if raw else "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token():
        headers["Authorization"] = f"token {token()}"
    if method == "GET":
        reply = get_with_retry(api_url(repository, path), headers=headers, timeout=timeout, **kwargs)
    else:
        reply = request(method, api_url(repository, path), headers=headers, timeout=timeout, **kwargs)
    if check and not reply.ok:
        raise Transient(f"{status(reply)} (on {method} {path})")
    return reply


def ref_head(repository: str) -> str:
    """The branch's current commit."""
    return api(repository, "GET", f"git/ref/heads/{CARD_BRANCH}").json()["object"]["sha"]


def commit_info(repository: str, sha: str) -> dict:
    """GET git/commits/{sha}: its tree, author and message."""
    return api(repository, "GET", f"git/commits/{sha}").json()


def upload_blob(repository: str, data: bytes, ssh_on_limit: bool = False) -> Optional[str]:
    """
    POST git/blobs.

    Args:
        ssh_on_limit: A rate-limited 403 answers None and sets `limited` (SSH now, cant_write untouched) instead of raising

    Returns:
        The blob's sha, or None when the token can't write (NO_WRITE: the SSH path takes over until a restart);
        raises Transient on a rate limit (unless ssh_on_limit) or any other failure
    """
    global cant_write, limited
    reply = api(repository, "POST", "git/blobs", check=False,
                json={"content": b64encode(data).decode("ascii"), "encoding": "base64"})
    if reply.status_code in NO_WRITE:
        limited = (reply.headers.get("retry-after") or reply.headers.get("x-ratelimit-remaining") == "0"
                   or "rate limit" in reply.text.lower())     # a secondary limit may send no header
        if limited:                             # this run's: not the token's rights
            if ssh_on_limit:
                limited = True
                return None
            raise Transient(f"{status(reply)}: rate limited (on POST git/blobs)")
        cant_write = True
        return None
    if not reply.ok:
        raise Transient(f"{status(reply)} (on POST git/blobs)")
    sha = reply.json()["sha"]
    if sha != blob_sha(data):
        raise Transient(f"GitHub stored blob {sha[:8]} for data hashing to {blob_sha(data)[:8]}")
    return sha


def api_commit(repository: str, head: str, base_tree: str, modes: dict, blobs: dict, message: str) -> Optional[str]:
    """
    One commit through the Git Data API: tree on base_tree, commit on head, non-force ref update.

    Args:
        modes: {repo path: mode} of the paths as the tree has them (100644 when absent)
        blobs: {repo path: uploaded blob sha}

    Returns:
        The commit's sha, or None on a 409/422 when the branch moved; raises Transient when it didn't (a rule refused it)
    """
    tree = api(repository, "POST", "git/trees", json={"base_tree": base_tree, "tree": [
        {"path": path, "mode": modes.get(path, "100644"), "type": "blob", "sha": sha} for path, sha in blobs.items()]}).json()["sha"]
    commit = api(repository, "POST", "git/commits", json={"message": message, "tree": tree, "parents": [head],
                                                          "author": AUTHOR, "committer": AUTHOR}).json()["sha"]
    try:
        reply = api(repository, "PATCH", f"git/refs/heads/{CARD_BRANCH}", check=False, json={"sha": commit, "force": False})
    except RequestException:                    # no reply: the ref may have moved anyway
        if ref_head(repository) != commit:
            raise
        return commit
    if reply.status_code in CONFLICT and ref_head(repository) != head:
        return None
    if not reply.ok:
        raise Transient(f"{status(reply)} (on PATCH git/refs/heads/{CARD_BRANCH})")
    return commit


# ========================
# git over SSH (the history clone)
# ========================
def git(where: str, *args: str, stdin: Optional[bytes] = None, env: Optional[dict] = None) -> bytes:
    """git in a repo dir, no prompts. In the blobless history clone only refs, trees and our own objects: never diff or log."""
    return run_process(["git", "-C", where, *args], input=stdin, capture_output=True, check=True, timeout=GIT_TIMEOUT,
                       env=batch_env(where) if env is None else env).stdout


def git_env() -> dict:
    """No prompts, ssh in BatchMode (SSH_AUTH_SOCK comes from the service), the bot's identity."""
    return batch_env(expanduser(HISTORY_CLONE), **GIT_IDENTITY)


def git_tree(clone: str, tree: Optional[str], changes: dict, env: dict) -> str:
    """
    Write a tree: `tree` (None for a new folder) with changes applied, folder by folder.

    Args:
        changes: {entry name: blob sha, or {nested changes} for a folder}

    Returns:
        The new tree's sha (--missing: siblings' blobs aren't in the blobless clone)
    """
    entries: dict = {}
    if tree:
        for line in git(clone, "ls-tree", "-z", tree, env=env).decode("utf-8").split("\0"):
            if line:
                meta, _, name = line.partition("\t")
                mode, kind, sha = meta.split(" ")
                entries[name] = (mode, kind, sha)
    for name, change in changes.items():
        old = entries.get(name)
        if isinstance(change, dict):
            entries[name] = ("040000", "tree", git_tree(clone, old[2] if old and old[1] == "tree" else None, change, env))
        else:
            entries[name] = (old[0] if old and old[1] == "blob" else "100644", "blob", change)
    data = "".join(f"{mode} {kind} {sha}\t{name}\0" for name, (mode, kind, sha) in entries.items())
    return git(clone, "mktree", "-z", "--missing", stdin=data.encode("utf-8"), env=env).decode().strip()


def git_commit(repository: str, head: str, files: dict, message: str) -> Optional[str]:
    """
    Commit files on head in the history clone and push to the branch over SSH (no force).

    Args:
        files: {repo path: bytes}

    Returns:
        The new commit's sha, or None when the push was rejected (the branch moved); raises Transient on any
        other git failure
    """
    if repository != CARD_REPO:
        raise Transient(f"the history clone mirrors {CARD_REPO}, not {repository}")
    refresh()                                   # the clone exists and knows head
    clone, env = expanduser(HISTORY_CLONE), git_env()
    try:
        changes: dict = {}
        for path, data in files.items():
            node = changes
            *folders, name = path.split("/")
            for folder in folders:
                node = node.setdefault(folder, {})
            node[name] = git(clone, "hash-object", "-w", "--stdin", stdin=data, env=env).decode().strip()
        tree = git_tree(clone, git(clone, "rev-parse", f"{head}^{{tree}}", env=env).decode().strip(), changes, env)
        commit = git(clone, "commit-tree", tree, "-p", head, "-m", message, env=env).decode().strip()
    except CalledProcessError as error:
        raise Transient(f"git {error.cmd[3]} in the history clone failed: "
                        f"{(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:  # slower than GIT_TIMEOUT, or no git
        raise Transient(f"git in the history clone failed: {error}") from error
    try:                                        # --no-thin: a thin pack would fetch the old blobs as its delta bases
        push = run_process(["git", "-C", clone, "push", "--porcelain", "--no-thin", f"git@github.com:{repository}.git",
                            f"{commit}:refs/heads/{CARD_BRANCH}"], capture_output=True, timeout=GIT_TIMEOUT, env=env)
    except (TimeoutExpired, OSError) as error:
        raise Transient(f"git push failed: {error}") from error
    if push.returncode:
        out = push.stdout.decode("utf-8", "replace")
        if any(line.startswith("!") and "[rejected]" in line for line in out.splitlines()):
            return None
        err = [line for line in (push.stderr.decode("utf-8", "replace") or out).splitlines() if line.strip()]
        raise Transient(f"git push failed: {' / '.join(err[max(0, len(err) - 3):])}"
                        + ("" if environ.get("SSH_AUTH_SOCK") else " (no SSH_AUTH_SOCK: no SSH agent)"))
    refresh()
    return commit


# ========================
# Public API Functions
# ========================
def commit_files(repository: str, head: str, base_tree: str, modes: dict, files: dict, message: str,
                 uploaded: Optional[set] = None, ssh_on_limit: bool = False) -> Optional[str]:
    """
    One commit of files on head: the API when the token can write, else git objects pushed over SSH.

    Args:
        head: The branch commit to build on (its tree is base_tree)
        modes: {repo path: mode} as the tree has them
        files: {repo path: bytes}
        message: Commit subject
        uploaded: Blob shas already uploaded this run (filled here): a retry re-uploads only what changed
        ssh_on_limit: A rate-limited API falls to SSH (the bot's choice) instead of raising (the Action's); `limited` then
            keeps the run on SSH until the caller resets it

    Returns:
        The commit's sha, or None when the branch moved (409/422, or the push was rejected): try again from the
        new head; raises Transient on any other failure
    """
    if token() and not cant_write and not (ssh_on_limit and limited):
        uploaded = set() if uploaded is None else uploaded
        for data in files.values():
            sha = blob_sha(data)
            if sha not in uploaded:
                if upload_blob(repository, data, ssh_on_limit) is None:
                    break
                uploaded.add(sha)
        else:
            return api_commit(repository, head, base_tree, modes, {path: blob_sha(data) for path, data in files.items()}, message)
    return git_commit(repository, head, files, message)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ========================
# Module Exports
# ========================
__all__ = [
    'API_TIMEOUT',
    'AUTHOR',
    'COMMIT_TRIES',
    'CONFLICT',
    'GIT_IDENTITY',
    'GIT_TIMEOUT',
    'NO_WRITE',
    'Transient',
    'api',
    'api_commit',
    'commit_files',
    'commit_info',
    'git',
    'git_commit',
    'git_env',
    'git_tree',
    'limited',
    'plural',
    'ref_head',
    'status',
    'token',
    'upload_blob',
]
