"""GitHub access for the card repo: the token, auth headers, a Session per thread, tree and blob fetches (sha-verified),
range reads of a PSD at a commit (RemotePSD), and the retry policy."""
from hashlib import sha1
from http.cookiejar import DefaultCookiePolicy
from os import environ
from re import compile as re_compile
from requests import Session
from requests.exceptions import RequestException
from threading import local
from time import sleep
from urllib.parse import quote

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
CONTENT_RANGE = re_compile(r"bytes (\d+)-(\d+)/(\d+)")   # a 206 reply's Content-Range

# ========================
# Global State
# ========================
_local = local()                                    # one requests.Session per thread, never shared


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


def fetch_verified(repository: str, sha: str, timeout: int = 30) -> bytes:
    """A blob's bytes by sha, checked against it; raises ValueError when they don't hash to sha."""
    data = fetch_blob(repository, sha, timeout=timeout).content
    if blob_sha(data) != sha:
        raise ValueError(f"blob {sha[:8]} hashes to {blob_sha(data)[:8]}")
    return data


def blob_text(repository: str, sha: str, timeout: int = 30) -> str:
    """A small text file by blob sha, read as UTF-8."""
    reply = fetch_blob(repository, sha, timeout=timeout)
    reply.encoding = "utf-8"
    return reply.text


def range_reads() -> bool:
    """Whether the installed psd-tools opens a PSD lazily over a RangeReader (the fork's feat/lazy-records)."""
    try:
        from psd_tools.range_reader import RangeReader  # noqa: F401
    except ImportError:
        return False
    return True


def raw_url(repository: str, ref: str, path: str) -> str:
    """raw.githubusercontent's URL of a file at a commit (a commit sha never lags behind, a branch name can)."""
    return f"{RAW_BASE}{repository}/{ref}/{quote(path)}"


class RangeError(OSError):
    """A range request answered with something else than its range (status is the HTTP code, 0 for a bad reply)."""
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class RemotePSD:
    """
    A card PSD at a commit, opened by HTTP range requests to raw.githubusercontent: psd-tools' lazy open fetches the
    header, resources and layer records (about 1% of a card), and a layer's pixels only when the parse reads them
    (stars, effect layers). Raw requests are not REST API requests.

    Args:
        repository: owner/name
        ref: A commit sha
        path: The file's repo path
        size: Its size from the tree, checked when given

    Raises:
        ImportError: psd-tools has no lazy reads (psd_tools.range_reader); RangeError: no range reply (a 404: the commit
        is not on GitHub) or another size; RequestException, or what psd-tools raises on the records
    """
    def __init__(self, repository: str, ref: str, path: str, size=None, timeout: int = 60):
        from psd_tools import PSDImage
        from psd_tools.range_reader import RangeReader
        self.url, self.timeout = raw_url(repository, ref, path), timeout
        self.failed = False                     # a fetch raised (the parse may have caught it): read the PSD whole
        self.reader = RangeReader(self._fetch)
        if size is not None and self.reader.size != size:
            raise RangeError(f"{self.reader.size} bytes at {ref[:8]}, the tree says {size}")
        self.psd = PSDImage.open(self.reader, lazy=True)

    def _fetch(self, start: int, stop: int) -> tuple:
        """Bytes [start, stop) and the file's size: one range request (get_with_retry's tries)."""
        try:
            reply = get_with_retry(self.url, stream=True, timeout=self.timeout,
                                   headers={**HEADERS, "Range": f"bytes={start}-{stop - 1}"})
            if reply.status_code != 206:
                reply.close()                   # a 200 would be the whole file: never read it here
                raise RangeError(f"HTTP {reply.status_code}, not a range", reply.status_code)
            found = CONTENT_RANGE.fullmatch(reply.headers.get("Content-Range", ""))
            if found is None or int(found.group(1)) != start:
                raise RangeError(f"Content-Range {reply.headers.get('Content-Range')!r} for bytes {start}-{stop - 1}")
            return reply.content, int(found.group(3))
        except Exception:
            self.failed = True
            raise

    @property
    def fetched(self) -> int:
        """Bytes fetched so far."""
        return self.reader.fetched

    def close(self) -> None:
        self.reader.close()


# ========================
# Module Exports
# ========================
__all__ = [
    'GIT_TOKEN',
    'HEADERS',
    'RangeError',
    'RemotePSD',
    'api_url',
    'blob_sha',
    'blob_text',
    'fetch_blob',
    'fetch_tree',
    'fetch_verified',
    'get',
    'get_with_retry',
    'print_token_status',
    'range_reads',
    'raw_url',
    'session',
    'token_from_env',
]
