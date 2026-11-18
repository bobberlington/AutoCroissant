"""
The `previews` branch of TTSCardMaker (docs/EDIT_PIPELINE_DESIGN.md section 16): a card preview's out dir (card_preview's
result.json and images) published as previews/<request>/ with its index.json row, one commit per run on the branch head,
retried on a moved head; when the branch would hold more than KEEP previews or one older than DAYS, the commit is a fresh
orphan with the KEEP_AFTER newest under DAYS_AFTER old (pushed with a lease). Also result.json's blank shape and clean(),
which keeps a dispatched value on one line of the job's log.

    python -m commands.card_preview_branch --out <dir> --request <id> [--git <checkout>]
    python -m commands.card_preview_branch --warm [--git <checkout>]
    python -m commands.card_preview_branch --take <dir>
    python -m commands.card_preview_branch --hand <dir> [--lane <dir>]   (stdin: ACTIONS_RUNTIME_TOKEN, ACTIONS_RESULTS_URL)

--warm reads the branch while the job waits for the preview: publish's first try then builds on that head, no network
until the push (a refused push: the usual tries from the head then). --take is the job's wait: this run's hand-over
(the preview job's artifact) listed by conditional GETs on one connection, then its zip unzipped into <dir> (take_over);
`go=` and `took=` lines on stdout for $GITHUB_OUTPUT. --hand is that hand-over made by the preview job itself, from a
process of its own (hand(); TTSCardMaker .github/actions/hand-over), hand.done beside <dir> saying ok, none or fail;
with --lane, then the warm lane's presses (card_preview_lane), each <lane>/<request>/ as LANE_ARTIFACT-<request>, until
<lane>.end (hand_lane).

Standard library and git only (never PIL, psd-tools, requests or card_bake): card-preview.yml's publish job runs it with
the runner's own python3 and no pip install, so a preview that never ran (a failed install, a timeout) is still published,
as a try-again error. It uses the checkout's git and credentials (actions/checkout's); blobless is enough (pushes
without delta search). SPARSE_FILES is its import closure (the publish job's checkout).
"""
from argparse import ArgumentParser
from base64 import urlsafe_b64decode
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from io import BytesIO
from json import dumps, loads
from os import environ, listdir, makedirs, replace
from os.path import abspath, dirname, isdir, isfile, join
from random import uniform
from re import compile as re_compile
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import argv, stderr, stdin
from time import monotonic, sleep
from typing import Optional
from unicodedata import category
from urllib.parse import urlsplit
from zipfile import ZIP_STORED, BadZipFile, ZipFile
from zlib import error as ZlibError

from commands.repo_history import batch_env

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_preview_branch.py", "commands/repo_history.py", "global_config.py")
REQUEST = re_compile(r"[A-Za-z0-9-]{8,64}")     # the client's id: a path on the branch (card-preview.yml checks it too)
MAX_CARD = 200                                  # characters of a card input (longer: an input error, kept clipped)
RESULT, PNG = "result.json", "preview.png"
SMALL, BEFORE = "preview.avif", "before.avif"  # 720 px; before: the card's PNG before the edit, made as SMALL (the Diff's)
STATES = ("baked", "redundant", "refused", "error")
VERSION = 1                                     # result.json's and index.json's "v"
BRANCH = "previews"
DIR = "previews"                                # previews/<request>/ on the branch
INDEX, README = "index.json", "README.md"
LOCAL_REF = "refs/card-preview/previews"        # where publish fetches the branch head (no remote-tracking ref moved)
WARM_REF = "refs/card-preview/warm"             # where warm() fetches it, for publish's first try
KEEP, DAYS = 200, timedelta(days=7)             # more previews or an older one: the next commit is a fresh orphan...
KEEP_AFTER, DAYS_AFTER = 150, timedelta(days=6)  # ... with the newest this many under this age (a rewrite about daily)
TRIES = 5                                       # attempts, each from the head then; 1-2 s, 2-4 s, 3-6 s ... between them
GIT_TIMEOUT = 300
IDENTITY = {"GIT_AUTHOR_NAME": "AutoCroissant", "GIT_AUTHOR_EMAIL": "autocroissant@gmail.com",
            "GIT_COMMITTER_NAME": "AutoCroissant", "GIT_COMMITTER_EMAIL": "autocroissant@gmail.com"}  # git_commit's
BREAKS = ("Cc", "Zl", "Zp")                     # control characters and line or paragraph separators
ARTIFACT = "card-preview"                       # the preview job's hand-over (card-preview.yml's Hand over)
HANDED = (RESULT, SMALL, BEFORE)                # the only names taken from its zip
ZIP_CAP = 16 << 20                              # bytes: the zip, and the files taken from it
POLL, FAST = 0.25, 30                           # s between listings (a 304 isn't counted) for the first FAST s...
POLL_LATE, LATE = 3, 55                         # ... then 1 s, POLL_LATE from LATE s on (as the gh loop before)
JOB_FIRST, JOB_EVERY = 10, 15                   # s: the preview job looked at (ended without a hand-over?)
WAIT = 715                                      # s at most (the step's 5 s before it: 12 minutes)
LOW = 100                                       # rate limit left under which listings slow to POLL_LATE
HAND_POLL, HAND_WAIT = 0.02, 600                # s between looks for result.json; s at most (the preview's 480 s, then)
HAND_DONE = "hand.done"                         # beside the out dir: ok, none or fail
LANE_ARTIFACT = "card-preview-"                 # + the request: a warm lane's hand-over (card_preview_lane)
LANE_END, LANE_HANDED = ".end", "hand.lane"     # <lane>.end: the lane stopped; hand.lane beside it: its hand-overs ended
LANE_WAIT = 3600                                # s at most: <lane>.end comes first (Start makes it if the lane dies)
RPC = "/twirp/github.actions.results.api.v1.ArtifactService/"     # upload-artifact v7's (Twirp JSON, proto names)
BLOB_VERSION = "2026-02-06"                     # x-ms-version: its @azure/storage-blob's
BRANCH_README = ("# Card previews\n\nMade by TTSCardMaker's `.github/workflows/card-preview.yml` (AutoCroissant "
                 "`commands/card_preview.py`): `previews/<request>/` holds `result.json`, `preview.avif` and "
                 "`before.avif` (the card before the edit, for the Diff); "
                 "`index.json` lists their dates. One commit per preview; rewritten as one fresh commit when it holds more "
                 f"than {KEEP} previews or one older than {DAYS.days} days. Never merged into main.\n")


# ========================
# Data
# ========================
class Unpublished(RuntimeError):
    """A git or GitHub failure: publish tries again from the branch head, TRIES times."""


def clean(text, limit: int = 0) -> str:
    """
    A value on one line: control characters and line separators as escapes (a newline as \\n), so a dispatched value
    never starts a line of the job's log (the runner reads a line starting `::` as a command).

    Args:
        limit: At most this many characters, the last an ellipsis (0: all)
    """
    out = "".join(c if category(c) not in BREAKS else "\\n" if c == "\n" else
                  f"\\x{ord(c):02x}" if ord(c) < 256 else f"\\u{ord(c):04x}" for c in str(text))
    return out if not limit or len(out) <= limit else out[:limit - 1] + "…"


def stamp() -> str:
    """Now in UTC, as result.json's `at`."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_url() -> Optional[str]:
    if environ.get("GITHUB_RUN_ID") and environ.get("GITHUB_REPOSITORY"):
        return f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{environ['GITHUB_REPOSITORY']}/actions/runs/{environ['GITHUB_RUN_ID']}"
    return None


def blank(request: str, card: str) -> dict:
    """result.json before anything is known: an error until the preview says otherwise (the card clean and clipped)."""
    return {"v": VERSION, "request": request if REQUEST.fullmatch(request or "") else None, "card": clean(card, MAX_CARD),
            "path": "", "png": "", "base": {"repo": "", "commit": "", "psd_blob": "", "png_blob": ""}, "edit": {},
            "state": "error", "transient": False, "fields": {}, "reason": "", "changes": [], "notes": [], "parity": {},
            "region": None, "image": None, "files": {}, "runner": {}, "run": _run_url(), "took_s": 0.0, "at": stamp()}


# ========================
# The previews branch
# ========================
def _git(where: str, *args: str, stdin: Optional[bytes] = None, lazy: bool = False) -> str:
    """git in the checkout as the bot (IDENTITY), no prompts; lazy: a blobless checkout may fetch a missing blob."""
    env = {**batch_env(where), **IDENTITY}
    if lazy:
        env.pop("GIT_NO_LAZY_FETCH", None)
    try:
        return run_process(["git", "-C", where, *args], input=stdin, capture_output=True, check=True, timeout=GIT_TIMEOUT,
                           env=env).stdout.decode("utf-8")
    except CalledProcessError as error:
        raise Unpublished(f"git {_verb(args)} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Unpublished(f"git {_verb(args)} failed: {error}") from error


def _verb(args: tuple) -> str:
    """The git command past any -c option."""
    return args[2] if args[0] == "-c" else args[0]


def _fetch(where: str, ref: str = LOCAL_REF) -> str:
    """The branch's head from origin into ref (one commit, its trees, no blobs); "" when there is no branch."""
    listed = _git(where, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}").split()
    if not listed:
        return ""
    _git(where, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", "--filter=blob:none", "origin",
         f"+refs/heads/{BRANCH}:{ref}")
    return _git(where, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def _listing(where: str, head: str) -> tuple:
    """({request: entry tree sha}, index rows) at the branch head; an unreadable index.json counts as empty."""
    if not head:
        return {}, {}
    root = {line.split("\t", 1)[1]: line.split()[2] for line in _git(where, "ls-tree", "-z", head).split("\0") if "\t" in line}
    entries = {}
    if DIR in root:
        for line in _git(where, "ls-tree", "-z", f"{head}:{DIR}").split("\0"):
            if "\t" in line and line.split()[1] == "tree":
                entries[line.split("\t", 1)[1]] = line.split()[2]
    index = {}
    if INDEX in root:
        try:
            data = loads(_git(where, "cat-file", "blob", root[INDEX], lazy=True))
            index = data.get("previews", {}) if isinstance(data, dict) and isinstance(data.get("previews"), dict) else {}
        except ValueError:
            index = {}
    return entries, index


def warm(where: str = ".") -> str:
    """publish's reads ahead of it: the head into WARM_REF, index.json local. Returns the head ("": no branch)."""
    head = _fetch(where, WARM_REF)
    _listing(where, head)
    return head


def _warmed(where: str) -> str:
    """warm()'s head; "" when it never ran or failed."""
    try:
        return _git(where, "rev-parse", "--verify", "-q", f"{WARM_REF}^{{commit}}").strip()
    except Unpublished:
        return ""


def _age(at, now: datetime) -> Optional[timedelta]:
    try:
        return now - datetime.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def kept(entries: dict, index: dict, request: str, now: datetime) -> Optional[list]:
    """
    The previews a rewrite keeps, or None when none is due: due when the branch would hold more than KEEP previews, or
    one older than DAYS (or of no known age); it keeps this request and the newest others under DAYS_AFTER, KEEP_AFTER in
    all.
    """
    ages = {r: _age((index.get(r) or {}).get("at"), now) for r in entries if r != request}
    if len(ages) + 1 <= KEEP and all(age is not None and age <= DAYS for age in ages.values()):
        return None
    fresh = sorted((age, r) for r, age in ages.items() if age is not None and age <= DAYS_AFTER)
    return [request] + [r for _, r in fresh[:KEEP_AFTER - 1]]


def _blob(where: str, data: bytes) -> str:
    return _git(where, "hash-object", "-w", "--stdin", stdin=data).strip()


def _tree(where: str, rows: list) -> str:
    """mktree of (mode, kind, sha, name) rows."""
    return _git(where, "mktree", stdin="".join(f"{m} {k} {s}\t{n}\n" for m, k, s, n in sorted(rows, key=lambda r: r[3])).encode()).strip()


def _push(where: str, commit: str, lease: Optional[str]) -> None:
    """The commit to the branch: a fast forward, or with lease (the head it was built from) a forced update; raises
    Unpublished when refused (the branch moved) or failed."""
    lease_arg = [f"--force-with-lease=refs/heads/{BRANCH}:{lease}"] if lease else []
    # no delta search: it would want a replaced file's old blob (HTTPS pushes thin whatever --no-thin says)
    args = ["-c", "pack.window=0", "push", "-q", "--no-thin", *lease_arg]
    _git(where, *args, "origin", f"{commit}:refs/heads/{BRANCH}")


def _result_of(out: str, request: str) -> dict:
    """The preview's result.json, or the stand-in a preview that never finished leaves (an error: try again)."""
    try:
        with open(join(out, RESULT), encoding="utf-8") as fh:
            result = loads(fh.read())
        if isinstance(result, dict) and result.get("request") == request and result.get("state") in STATES:
            return result
    except (OSError, ValueError):
        pass
    result = blank(request, "")
    result.update(transient=True, reason="error: the preview left no result (see the run)")
    return result


def publish(out: str, request: str, where: str = ".", now: Optional[datetime] = None) -> dict:
    """
    One commit on the previews branch with previews/<request>/ and its index row, on warm()'s head if it ran, then on
    the branch's head (TRIES times from a moved head); a rewrite as a fresh orphan commit when kept() says so, pushed
    with a lease.

    Args:
        out: The preview's out dir
        request: The client's id (REQUEST)
        where: A checkout of the card repo (its git and credentials; blobless is enough); nothing in it moves but LOCAL_REF
            (warm() moves WARM_REF)

    Returns:
        {"commit", "parent" (None for an orphan), "rewritten", "kept", "dropped", "result"}; raises Unpublished when no
        push landed
    """
    if not REQUEST.fullmatch(request or ""):
        raise ValueError("the request must be 8-64 of A-Z a-z 0-9 -")
    result = _result_of(out, request)
    files = {}
    for name in (SMALL, BEFORE):
        if name in (result.get("files") or {}).values() and isfile(join(out, name)):
            with open(join(out, name), "rb") as fh:
                files[name] = fh.read()
    result["files"] = {kind: name for kind, name in (result.get("files") or {}).items() if name in files}   # what is published
    files[RESULT] = (dumps(result, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    card = clean(result.get("card") or "", MAX_CARD)
    row = {"at": result.get("at")}            # kept()'s
    entry = _tree(where, [("100644", "blob", _blob(where, data), name) for name, data in files.items()])
    readme = _blob(where, BRANCH_README.encode("utf-8"))
    why = ""
    head = _warmed(where)
    for attempt in range(-1 if head else 0, TRIES):     # -1: on warm()'s head, an extra try; 0 follows it with no sleep
        if attempt > 0:
            sleep(uniform(1, 2) * attempt)
        try:
            if attempt >= 0:
                head = _fetch(where)
            entries, index = _listing(where, head)
            entries[request], index[request] = entry, row
            keep = kept(entries, index, request, now or datetime.now(timezone.utc)) if head else None
            dropped = sorted(set(entries) - set(keep)) if keep is not None else []
            entries = {r: t for r, t in entries.items() if r not in dropped}
            index = {r: index[r] for r in sorted(entries) if r in index}
            listing = _blob(where, (dumps({"v": VERSION, "previews": index}, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())
            previews = _tree(where, [("040000", "tree", t, r) for r, t in entries.items()])
            root = _tree(where, [("100644", "blob", listing, INDEX), ("100644", "blob", readme, README), ("040000", "tree", previews, DIR)])
            message = f"Preview {request}: {card or '?'} ({result.get('state')})"
            if keep is not None:
                message += (f"\n\nRewritten as one commit: {len(entries)} of {len(entries) + len(dropped)} previews kept "
                            f"(more than {KEEP}, or one older than {DAYS.days} days).")
            parent = head if head and keep is None else None
            commit = _git(where, "commit-tree", root, *(["-p", parent] if parent else []), "-m", message).strip()
            _push(where, commit, head if keep is not None else None)
        except Unpublished as error:            # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_preview_branch: attempt {attempt + 1} of {TRIES}: {clean(why, 2000)}", file=stderr)
            continue
        return {"commit": commit, "parent": parent, "rewritten": keep is not None, "kept": len(entries), "dropped": dropped,
                "result": result}
    raise Unpublished(f"not published after {TRIES} tries: {why}")


# ========================
# The hand-over
# ========================
def _local(url) -> bool:
    """A plain-HTTP stand-in on this machine (the tests' and a local lane's)."""
    return url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost")


class _Api:
    """GETs on one kept HTTPS connection to the API host (the token's only host), conditional where asked."""

    def __init__(self, base: str, token: str):
        url = urlsplit(base)
        self.host, self.root, self.plain = url.netloc, url.path.rstrip("/"), _local(url)
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "card-preview-publish"}
        self.connection = None
        self.tags = {}                          # path -> ETag
        self.used, self.left, self.sent, self.unchanged = [], None, 0, 0

    def get(self, path: str, conditional: bool = False) -> tuple:
        """(status, reply, body); one retry on a fresh connection."""
        headers = {**self.headers, **({"If-None-Match": self.tags[path]} if conditional and path in self.tags else {})}
        for retry in (False, True):
            if self.connection is None:
                self.connection = (HTTPConnection if self.plain else HTTPSConnection)(self.host, timeout=10)
            try:
                self.connection.request("GET", self.root + path, headers=headers)
                reply = self.connection.getresponse()
                body = reply.read()
                break
            except (OSError, HTTPException):
                self.close()
                if retry:
                    raise
        self.sent += 1
        self.unchanged += reply.status == 304
        if reply.getheader("X-RateLimit-Used"):
            used = int(reply.getheader("X-RateLimit-Used"))
            self.used = [(self.used or [used])[0], used]       # first, last
            self.left = int(reply.getheader("X-RateLimit-Remaining") or 0)
        if conditional and reply.status == 200 and reply.getheader("ETag"):
            self.tags[path] = reply.getheader("ETag")
        return reply.status, reply, body

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None


def _unzip(data: bytes) -> dict:
    """HANDED's files in a zip, {name: bytes}; raises ValueError on a bad zip or past ZIP_CAP."""
    files, total = {}, 0
    try:
        with ZipFile(BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.filename in HANDED and info.filename not in files:
                    total += info.file_size
                    if total > ZIP_CAP:
                        raise ValueError("past the size cap")
                    files[info.filename] = archive.read(info)
    except (BadZipFile, OSError, EOFError, ZlibError) as error:
        raise ValueError(f"unreadable zip: {error}") from error
    return files


def _found(status: int, reply, body: bytes) -> list:
    """ARTIFACT's rows in a 200 listing ([] for any other reply)."""
    return [a for a in loads(body).get("artifacts", []) if a.get("name") == ARTIFACT and not a.get("expired")] if status == 200 else []


def _take(api: _Api, repo: str, row: dict, out: str) -> dict:
    """The listed artifact's zip: its 302 on the API's connection, the blob store's URL without the token, checked
    against the row's digest; HANDED's files written into out. Returns {name: size}; raises ValueError or OSError."""
    status, reply, _ = api.get(f"/repos/{repo}/actions/artifacts/{int(row['id'])}/zip")
    to = urlsplit(reply.getheader("Location") or "")
    if status != 302 or to.scheme != "https" or not to.netloc:
        raise ValueError(f"the zip answered {status}")
    store = HTTPSConnection(to.netloc, timeout=30)
    try:
        store.request("GET", to.path + (f"?{to.query}" if to.query else ""), headers={"User-Agent": "card-preview-publish"})
        reply = store.getresponse()
        data = reply.read(ZIP_CAP + 1)
    except HTTPException as error:
        raise ValueError(f"the blob store: {error}") from error
    finally:
        store.close()
    if reply.status != 200 or len(data) > ZIP_CAP:
        raise ValueError(f"the blob store answered {reply.status} ({len(data)} bytes)")
    digest = str(row.get("digest") or "")
    if digest.startswith("sha256:") and digest[7:] != sha256(data).hexdigest():
        raise ValueError("the zip's digest isn't the listing's")
    files = _unzip(data)
    makedirs(out, exist_ok=True)
    for name, body in files.items():
        with open(join(out, name), "wb") as fh:
            fh.write(body)
    return {name: len(body) for name, body in files.items()}


def take_over(out: str, repo: str, run: str, token: str, base: str = "https://api.github.com", limit: float = WAIT) -> str:
    """
    card-preview.yml's wait: the run's ARTIFACT listed (every POLL s, If-None-Match: a 304 isn't counted against the
    rate limit), then its zip from the same connection (_take); the preview job looked at from JOB_FIRST s, every
    JOB_EVERY s, for an end without a hand-over. Never raises for GitHub's or the network's sake.

    Returns:
        "taken" (HANDED's files in out), "listed" (listed, not taken: the fallback download-artifact's), "ended" (the
        job validated, then ended without a hand-over: Publish publishes a try-again error), "invalid" (it never
        validated: nothing to publish) or "late" (none of those in limit seconds)
    """
    api, runs, start = _Api(base, token), f"/repos/{repo}/actions/runs/{run}", monotonic()
    listing, job_at, state = f"{runs}/artifacts?name={ARTIFACT}", JOB_FIRST, "late"
    try:
        while (now := monotonic() - start) < limit:
            pause = POLL_LATE if now >= LATE or (api.left is not None and api.left < LOW) else POLL if now < FAST else 1
            try:
                status, reply, body = api.get(listing, conditional=True)
                found = _found(status, reply, body)
                if found:
                    state = "listed"
                    break
                if status in (403, 429):            # rate limited
                    pause = max(POLL_LATE, int(reply.getheader("Retry-After") or 0))
                elif status not in (200, 304):
                    pause = max(pause, 1)
                if monotonic() - start >= job_at:
                    job_at = monotonic() - start + JOB_EVERY
                    status, _, body = api.get(f"{runs}/jobs", conditional=True)
                    job = next((j for j in loads(body).get("jobs", []) if j.get("name") == "preview"), None) if status == 200 else None
                    if job and job.get("status") == "completed":
                        valid = any(s.get("name") == "Validate" and s.get("conclusion") == "success" for s in job.get("steps") or [])
                        found = _found(*api.get(listing)) if valid else []     # a hand-over since the listing
                        state = "listed" if found else "ended" if valid else "invalid"
                        break
            except (OSError, HTTPException, ValueError, AttributeError, TypeError) as error:
                print(f"card_preview_branch: a read failed ({clean(type(error).__name__)}): tried again", file=stderr)
                pause = max(pause, 1)
            sleep(pause)
        took = monotonic()
        if state == "listed":
            try:
                sizes = _take(api, repo, found[0], out)
                state = "taken"
                print(f"Hand-over taken in {monotonic() - took:.2f} s: " + ", ".join(f"{n} ({s} bytes)" for n, s in sizes.items()), file=stderr)
            except (OSError, HTTPException, ValueError, KeyError, TypeError) as error:
                print(f"card_preview_branch: the hand-over not taken ({clean(error, 300)}): download-artifact's", file=stderr)
    finally:
        api.close()
    used = f"rate limit used {api.used[0]} -> {api.used[-1]}" if api.used else "no rate limit read"
    print(f"Waited {took - start:.1f} s: {state}; {api.sent} API reads, {api.unchanged} not modified; {used}.", file=stderr)
    return state


class _Kept:
    """One HTTPS connection kept to a host, its TLS made at once; a request on one the host closed goes once more on a
    new one."""

    def __init__(self, host: str, plain: bool = False):
        self.host, self.plain, self.connection = host, plain, None
        try:
            self.open()
        except OSError:                         # made at the first request
            self.close()

    def open(self) -> None:
        self.close()
        self.connection = (HTTPConnection if self.plain else HTTPSConnection)(self.host, timeout=30)
        self.connection.connect()

    def send(self, method: str, path: str, body: bytes, headers: dict) -> tuple:
        """(status, reply, body)."""
        try:
            return self._once(method, path, body, headers)
        except (OSError, HTTPException):
            self.open()
            return self._once(method, path, body, headers)

    def _once(self, method: str, path: str, body: bytes, headers: dict) -> tuple:
        if self.connection is None:
            self.open()
        self.connection.request(method, path, body=body, headers=headers)
        reply = self.connection.getresponse()
        return reply.status, reply, reply.read()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None


def _backend_ids(token: str) -> tuple:
    """(run, job): the runtime token's scp claim `Actions.Results:<run>:<job>`, as upload-artifact reads it."""
    try:
        part = token.split(".")[1]
        scopes = str(loads(urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("scp") or "").split(" ")
    except (IndexError, ValueError, AttributeError) as error:
        raise ValueError("an unreadable runtime token") from error
    for scope in scopes:
        bits = scope.split(":")
        if bits[0] == "Actions.Results" and len(bits) == 3:
            return bits[1], bits[2]
    raise ValueError("no Actions.Results scope in the runtime token")


def _rpc(api: _Kept, token: str, method: str, body: dict) -> dict:
    """One call of the artifact service, as upload-artifact v7's; raises ValueError unless it answers ok."""
    status, _, data = api.send("POST", RPC + method, dumps(body).encode(), {
        "Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": "card-preview-hand"})
    try:
        reply = loads(data)
    except ValueError:
        reply = None
    if status != 200 or not isinstance(reply, dict) or reply.get("ok") is not True:
        raise ValueError(f"{method} answered {status} {clean(reply.get('msg', '') if isinstance(reply, dict) else '', 200)}")
    return reply


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def hand(out: str, token: str, results: str, limit: float = HAND_WAIT, name: str = ARTIFACT) -> str:
    """
    The preview job's hand-over as upload-artifact v7 makes it, from a process of its own started while the preview
    bakes: ARTIFACT created at once and both hosts' connections made; the moment result.json is in out (looked for every
    HAND_POLL s), HANDED's files there zipped (stored, at the root), the zip put on the blob store's connection and the
    artifact finalized (size, sha256: --take's digest) on the service's, a failed put or finalize once more. A preview
    that ended (<out>.code, card-preview.yml's) or ran past limit with no result.json: nothing handed.

    Args:
        token, results: ACTIONS_RUNTIME_TOKEN and ACTIONS_RESULTS_URL (never logged; the token to the service alone)
        name: The artifact's (a lane's press: LANE_ARTIFACT + its request)

    Returns:
        "ok" (finalized) or "none" (no result.json); raises ValueError, OSError or HTTPException otherwise
    """
    started = monotonic()
    ids = dict(zip(("workflow_run_backend_id", "workflow_job_run_backend_id"), _backend_ids(token)), name=name)
    service = urlsplit(results)
    if (service.scheme != "https" and not _local(service)) or not service.netloc:
        raise ValueError("no results URL")
    api = _Kept(service.netloc, _local(service))
    store = None
    try:
        expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        made = _rpc(api, token, "CreateArtifact", {**ids, "expires_at": expires, "version": 7, "mime_type": "application/zip"})
        upload = urlsplit(str(made.get("signed_upload_url") or made.get("signedUploadUrl") or ""))
        if (upload.scheme != "https" and not _local(upload)) or not upload.netloc:
            raise ValueError("CreateArtifact gave no upload URL")
        store = _Kept(upload.netloc, _local(upload))
        created = monotonic()
        result = join(out, RESULT)
        while not isfile(result) and not isfile(out.rstrip("/") + ".code") and monotonic() - started < limit:
            sleep(HAND_POLL)
        if not isfile(result):
            print(f"No result.json at {_utc()}: nothing handed.", file=stderr)
            return "none"
        seen, seen_at = monotonic(), _utc()
        zipped = BytesIO()
        with ZipFile(zipped, "w", ZIP_STORED) as archive:
            for name in HANDED:
                if isfile(join(out, name)):
                    archive.write(join(out, name), name)
        data = zipped.getvalue()
        headers = {"x-ms-blob-type": "BlockBlob", "x-ms-version": BLOB_VERSION, "x-ms-blob-content-type": "application/zip"}
        for _ in (1, 2):
            status, reply = store.send("PUT", upload.path + (f"?{upload.query}" if upload.query else ""), data, headers)[:2]
            if status == 201:
                break
            print(f"card_preview_branch: the blob store answered {status} {clean(reply.getheader('x-ms-error-code') or '', 80)}",
                  file=stderr)
        else:
            raise ValueError("the zip was not put")
        put = monotonic()
        body = {**ids, "size": str(len(data)), "hash": "sha256:" + sha256(data).hexdigest()}
        try:
            done = _rpc(api, token, "FinalizeArtifact", body)
        except (ValueError, OSError, HTTPException) as error:
            print(f"card_preview_branch: {clean(error, 300)}: once more", file=stderr)
            done = _rpc(api, token, "FinalizeArtifact", body)
        end = monotonic()
        print(f"Handed over {ids['name']} {len(data)} bytes (artifact {clean(done.get('artifact_id', '?'), 40)}): created "
              f"{created - started:.3f} s after start; result.json seen {seen_at} UTC, put {put - seen:.3f} s on, "
              f"finalized {end - put:.3f} s later ({_utc()} UTC).", file=stderr)
        return "ok"
    finally:
        api.close()
        if store is not None:
            store.close()


def hand_lane(lane: str, token: str, results: str, limit: float = LANE_WAIT) -> int:
    """
    The warm lane's hand-overs (card_preview_lane), one at a time: each new <lane>/<request>/ (made as the lane takes the
    press) is handed over as LANE_ARTIFACT + request (hand(): created at once, its files the moment result.json is in;
    <lane>/<request>.code with no result.json: none), until <lane>.end or limit s. Returns how many were handed over.
    """
    started, done, handed = monotonic(), set(), 0
    while monotonic() - started < limit:
        ended = isfile(lane + LANE_END)         # before the listing: a press made before the end is handed over
        for request in sorted(listdir(lane)) if isdir(lane) else []:
            if request in done or not REQUEST.fullmatch(request) or not isdir(join(lane, request)):
                continue
            done.add(request)
            try:
                handed += hand(join(lane, request), token, results, limit - (monotonic() - started),
                               LANE_ARTIFACT + request) == "ok"
            except (ValueError, OSError, HTTPException) as error:
                why = clean(error, 300) if isinstance(error, (ValueError, OSError)) else type(error).__name__
                print(f"card_preview_branch: {request} not handed over ({why})", file=stderr)
        if ended:
            break
        sleep(HAND_POLL)
    return handed


# ========================
# Public API Functions
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line (module docstring); one line on stdout."""
    parser = ArgumentParser(prog="python -m commands.card_preview_branch",
                            description=f"Publish a card preview's out dir on the {BRANCH} branch, one commit "
                                        "(docs/EDIT_PIPELINE_DESIGN.md section 16).")
    parser.add_argument("--out", help="the preview's out dir")
    parser.add_argument("--request", help="the client's id (8-64 of A-Z a-z 0-9 -)")
    parser.add_argument("--git", default=".", help="a checkout of the card repo (its git and credentials)")
    parser.add_argument("--warm", action="store_true", help="only read the branch, for a publish later (no --out or --request)")
    parser.add_argument("--take", metavar="DIR", help="wait for this run's hand-over, unzipped into DIR (env GITHUB_REPOSITORY, "
                                                      "GITHUB_RUN_ID, GH_TOKEN); go= and took= lines on stdout")
    parser.add_argument("--hand", metavar="DIR", help="hand DIR's preview over as the artifact once its result.json is "
                                                      "in (stdin: the runtime token, the results URL); hand.done beside it")
    parser.add_argument("--lane", metavar="DIR", help="with --hand, once that is handed over: the warm lane's presses in "
                                                      "DIR, each its own artifact, until DIR.end; hand.lane beside it")
    options = parser.parse_args(argv[1:] if args is None else args)
    if options.hand:
        token, results, state = stdin.readline().strip(), stdin.readline().strip(), "fail"
        try:
            state = hand(options.hand, token, results)
        except (ValueError, OSError, HTTPException) as error:     # never a URL (InvalidURL's would hold it)
            why = clean(error, 300) if isinstance(error, (ValueError, OSError)) else type(error).__name__
            print(f"card_preview_branch: no hand-over ({why}): upload-artifact's", file=stderr)
        finally:
            done = join(dirname(abspath(options.hand)), HAND_DONE)
            with open(done + ".tmp", "w", encoding="utf-8") as fh:
                fh.write(state + "\n")
            replace(done + ".tmp", done)
        if options.lane and state == "ok":      # the lane waits for that ok
            handed = hand_lane(abspath(options.lane), token, results)
            with open(join(dirname(abspath(options.lane)), LANE_HANDED), "w", encoding="utf-8") as fh:
                fh.write(f"{handed}\n")
        return 0
    if options.take:
        state = take_over(options.take, environ["GITHUB_REPOSITORY"], environ["GITHUB_RUN_ID"],
                          environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN", ""),
                          environ.get("GITHUB_API_URL", "https://api.github.com"))
        if state == "invalid":
            print("The request never validated: nothing to publish.", file=stderr)
        elif state == "late":
            print("::warning::the preview did not end in 12 minutes: nothing published", file=stderr)
        else:
            print("go=true" + ("\ntook=true" if state == "taken" else ""))
        return 0
    if not options.warm and (options.out is None or options.request is None):
        parser.error("--out and --request are required")
    try:
        if options.warm:
            head = warm(options.git)
            print(f"Read `{BRANCH}` at `{head[:8]}`." if head else f"No `{BRANCH}` branch yet.")
            return 0
        done = publish(options.out, options.request, options.git)
    except (ValueError, Unpublished) as error:  # a bad request, or no push landed
        print(f"card_preview_branch: not {'read' if options.warm else 'published'}: {clean(error, 2000)}", file=stderr)
        return 1
    kept_n = done["kept"]
    how = (f"on `{done['parent'][:8]}`" if done["parent"] else "as the branch's first commit" if not done["rewritten"] else
           f"as one fresh commit ({kept_n} preview{'' if kept_n == 1 else 's'} kept, {len(done['dropped'])} dropped)")
    print(f"Published `{DIR}/{options.request}/` ({done['result']['state']}) in `{done['commit'][:8]}` {how}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'SPARSE_FILES',
    'Unpublished',
    'blank',
    'clean',
    'hand',
    'hand_lane',
    'kept',
    'main',
    'publish',
    'stamp',
    'take_over',
    'warm',
]
