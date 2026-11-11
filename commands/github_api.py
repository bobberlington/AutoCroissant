"""GitHub access for the card repo: the token, auth headers, a Session per thread, tree and blob fetches (sha-verified),
and the retry policy. No psd-tools: the bot loads it (RemotePSD is psd_remote's)."""
from hashlib import sha1
from http.cookiejar import DefaultCookiePolicy
from os import environ
from os.path import exists, getmtime
from requests import Session
from requests.exceptions import RequestException
from subprocess import TimeoutExpired, run
from threading import local
from time import sleep, time
from urllib.parse import quote

try:
    from fcntl import LOCK_EX, flock                # the two checks side by side: one REST read of a blob
except ImportError:                                 # Windows: no claim
    flock = None
try:
    import config
except ImportError:                                 # standalone use (python -m commands.tts_save) without config.py
    config = None
from global_config import API_BASE, CARD_BRANCH, RAW_BASE

# ========================
# Configuration
# ========================
GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")
HEADERS: dict = {"Authorization": f"token {GIT_TOKEN}"} if GIT_TOKEN else {}
RAW_MEDIA = "application/vnd.github.raw"            # Accept: a blob's bytes instead of its JSON envelope
RETRY_WAITS = (1, 2)                                # seconds between the 3 tries of get_with_retry
RANGE_READS = True                                  # read card PSDs by range requests when psd-tools can (the data build,
                                                    # card_check --after); False: whole downloads
PENDING_ENV = "CARD_BLOBS_PENDING"                  # card-guards.yml's fetch list while it fetches (<list>.done: it ended)
PENDING_POLL, PENDING_CAP = 0.2, 300                # s: a listed blob looked for; the fetch's cap from the list's write

# ========================
# Global State
# ========================
_local = local()                                    # one requests.Session per thread, never shared
_listed: dict = {}                                  # PENDING_ENV's list -> (its shas, the wait's end): read once


# ========================
# Public API Functions
# ========================
def print_token_status() -> None:
    """One line at startup: is there a token (5000 requests/hour) or not (60)."""
    print("Git token found, API limited to 5000 requests/hour." if GIT_TOKEN
          else "No git token in config, API limited to 60 requests/hour.")


def token_from_env() -> None:
    """The Actions' GITHUB_TOKEN when config.py supplied none (the bot's own token wins)."""
    if not GIT_TOKEN and environ.get("GITHUB_TOKEN"):
        HEADERS["Authorization"] = f"token {environ['GITHUB_TOKEN']}"


def session() -> Session:
    """This thread's Session: its connections are kept alive between requests. It keeps no cookies (like requests.get)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = Session()
        s.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        _local.session = s
    return s


def get(url: str, **kwargs):
    """requests.get on this thread's Session: same headers, auth and timeouts, without a new connection each time."""
    return session().get(url, **kwargs)


def api_url(repository: str, path: str) -> str:
    """https://api.github.com/repos/<repository>/<path>."""
    return f"{API_BASE}{repository}/{path}"


def get_with_retry(url: str, **kwargs):
    """
    GET with up to 3 tries, 1 s then 2 s apart, on a connection error, 429 or 5xx.

    Returns:
        The last response (check its status); the last connection error is raised
    """
    kwargs.setdefault("timeout", 30)
    for wait in RETRY_WAITS + (None,):
        try:
            reply = get(url, **kwargs)
        except RequestException:
            if wait is None:
                raise
        else:
            if wait is None or (reply.status_code != 429 and reply.status_code < 500):
                return reply
            _discard(reply)
        sleep(wait)


def fetch_tree(repository: str, etag: str = "", retry: bool = True, timeout: int = 30):
    """
    GET the recursive tree of CARD_BRANCH. With etag (If-None-Match), GitHub answers 304 when nothing changed.

    Returns:
        The response: callers handle 304, 200 and errors themselves (each keeps its own ETag)
    """
    headers = {**HEADERS, **({"If-None-Match": etag} if etag else {})}
    url = api_url(repository, f"git/trees/{CARD_BRANCH}?recursive=1")
    if retry:
        return get_with_retry(url, headers=headers, timeout=timeout)
    return get(url, headers=headers, timeout=timeout)


def fetch_blob(repository: str, sha: str, stream: bool = False, timeout: int = 30):
    """
    A file's exact bytes by blob sha (raw.githubusercontent can lag behind the tree), tried 3 times on a 5xx, 429 or
    connection error (get_with_retry).

    Returns:
        The response; raises HTTPError on an error status, or the last connection error
    """
    reply = get_with_retry(api_url(repository, f"git/blobs/{sha}"), stream=stream, timeout=timeout,
                           headers={**HEADERS, "Accept": RAW_MEDIA})
    if reply.status_code >= 400 and stream:
        reply.close()                               # hand the connection back
    reply.raise_for_status()
    return reply


def _discard(reply) -> None:
    """Read and close a reply that gets retried, so a stream's connection goes back to the pool."""
    try:
        reply.content                               # a short error body
    except RequestException:
        pass
    reply.close()


def blob_sha(data: bytes) -> str:
    """git's blob sha: sha1("blob N\\0" + data)."""
    return sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git_blob(clone: str, *args: str, data: bytes = None):
    """git cat-file / hash-object in a clone (never a lazy fetch); None when it fails."""
    try:
        done = run(["git", "-C", clone, *args], input=data, capture_output=True, timeout=300,
                   env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    except (OSError, TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def pending_blob(clone: str, sha: str):
    """A blob the guards' background fetch is bringing (PENDING_ENV's list): looked for every PENDING_POLL s until it
    lands or the fetch ends (<list>.done, or PENDING_CAP); None when not listed or not brought."""
    path = environ.get(PENDING_ENV, "")
    if not path:
        return None
    if path not in _listed:
        try:
            with open(path, encoding="ascii") as fh:
                _listed[path] = frozenset(fh.read().split()), getmtime(path) + PENDING_CAP
        except OSError:
            _listed[path] = frozenset(), 0.0
    shas, end = _listed[path]
    if sha not in shas:
        return None
    while True:
        ended = exists(f"{path}.done") or time() >= end     # before the look: the last pack is seen
        data = _git_blob(clone, "cat-file", "blob", sha)
        if data is not None or ended:
            return data
        sleep(PENDING_POLL)


def _claim(sha: str):
    """With PENDING_ENV: a lock on sha's REST read (held until closed), so the other check waits and reads the clone;
    None without it."""
    path = environ.get(PENDING_ENV, "")
    if not path or flock is None:
        return None
    try:
        held = open(f"{path}.{sha}.lock", "a")
    except OSError:
        return None
    flock(held, LOCK_EX)
    return held


def fetch_verified(repository: str, sha: str, timeout: int = 30, clone: str = "") -> bytes:
    """A blob's bytes by sha, checked against it; raises ValueError when they don't hash to sha. With clone: its object
    store first (a blob in PENDING_ENV's list waited for), and a fetched blob is kept there (the next reader of that
    clone does not download it again; with PENDING_ENV one reader at a time)."""
    data = _git_blob(clone, "cat-file", "blob", sha) if clone else None
    if data is None and clone:
        data = pending_blob(clone, sha)
    if data is not None and blob_sha(data) == sha:
        return data
    claim = _claim(sha) if clone else None
    try:
        if claim:
            data = _git_blob(clone, "cat-file", "blob", sha)            # the other check read it meanwhile
            if data is not None and blob_sha(data) == sha:
                return data
        data = fetch_blob(repository, sha, timeout=timeout).content
        if blob_sha(data) != sha:
            raise ValueError(f"blob {sha[:8]} hashes to {blob_sha(data)[:8]}")
        if clone:
            _git_blob(clone, "hash-object", "-w", "--stdin", data=data)
        return data
    finally:
        if claim:
            claim.close()


def blob_text(repository: str, sha: str, timeout: int = 30) -> str:
    """A small text file by blob sha, read as UTF-8."""
    reply = fetch_blob(repository, sha, timeout=timeout)
    reply.encoding = "utf-8"
    return reply.text


def range_reads() -> bool:
    """Whether range reads are on (RANGE_READS): the fork opens a PSD lazily over a RangeReader."""
    return RANGE_READS


def raw_url(repository: str, ref: str, path: str) -> str:
    """raw.githubusercontent's URL of a file at a commit (a commit sha never lags behind, a branch name can)."""
    return f"{RAW_BASE}{repository}/{ref}/{quote(path)}"


# ========================
# Module Exports
# ========================
__all__ = [
    'GIT_TOKEN',
    'HEADERS',
    'RANGE_READS',
    'api_url',
    'blob_sha',
    'blob_text',
    'fetch_blob',
    'fetch_tree',
    'fetch_verified',
    'get',
    'get_with_retry',
    'pending_blob',
    'print_token_status',
    'range_reads',
    'raw_url',
    'session',
    'token_from_env',
]
