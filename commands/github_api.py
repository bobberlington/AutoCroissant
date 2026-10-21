"""GitHub access for the card repo: the token, auth headers, tree and blob fetches (sha-verified), and the retry policy."""
from hashlib import sha1
from os import environ
from requests import get
from requests.exceptions import RequestException
from time import sleep

try:
    import config
except ImportError:                                 # standalone use (python -m commands.tts_save) without config.py
    config = None
from global_config import API_BASE, CARD_BRANCH

# ========================
# Configuration
# ========================
GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")
HEADERS: dict = {"Authorization": f"token {GIT_TOKEN}"} if GIT_TOKEN else {}
RAW_MEDIA = "application/vnd.github.raw"            # Accept: a blob's bytes instead of its JSON envelope
RETRY_WAITS = (1, 2)                                # seconds between the 3 tries of get_with_retry


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
    """A file's exact bytes by blob sha (raw.githubusercontent can lag behind the tree). Raises on an error status."""
    reply = get(api_url(repository, f"git/blobs/{sha}"), stream=stream, timeout=timeout,
                headers={**HEADERS, "Accept": RAW_MEDIA})
    reply.raise_for_status()
    return reply


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


# ========================
# Module Exports
# ========================
__all__ = [
    'GIT_TOKEN',
    'HEADERS',
    'api_url',
    'blob_sha',
    'blob_text',
    'fetch_blob',
    'fetch_tree',
    'fetch_verified',
    'get_with_retry',
    'print_token_status',
    'token_from_env',
]
