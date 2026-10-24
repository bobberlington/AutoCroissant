"""
The `previews` branch of TTSCardMaker (docs/EDIT_PIPELINE_DESIGN.md section 16): a card preview's out dir (card_preview's
result.json and images) published as previews/<request>/ with its index.json row, one commit per run on the branch head,
retried on a moved head; when the branch would hold more than KEEP previews or one older than DAYS, the commit is a fresh
orphan with the KEEP_AFTER newest under DAYS_AFTER old (pushed with a lease). Also result.json's blank shape and clean(),
which keeps a dispatched value on one line of the job's log.

    python -m commands.card_preview_branch --out <dir> --request <id> [--git <checkout>] [--remote <name or url>] [--png]

Standard library and git only (never PIL, psd-tools, requests or card_bake): card-preview.yml's publish job runs it with
the runner's own python3 and no pip install, so a preview that never ran (a failed install, a timeout) is still published,
as a try-again error. It uses the checkout's git and credentials (actions/checkout's); blobless is enough (--no-thin
pushes). SPARSE_FILES is its import closure (the publish job's checkout).
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from json import dumps, loads
from os import environ
from os.path import isfile, join
from random import uniform
from re import compile as re_compile
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import argv, stderr
from time import sleep
from typing import Optional
from unicodedata import category

from commands.repo_history import batch_env

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_preview_branch.py", "commands/repo_history.py", "global_config.py")
REQUEST = re_compile(r"[A-Za-z0-9-]{8,64}")     # the client's id: a path on the branch (card-preview.yml checks it too)
MAX_CARD = 200                                  # characters of a card input (longer: an input error, kept clipped)
RESULT, PNG, WEBP = "result.json", "preview.png", "preview.webp"
STATES = ("baked", "redundant", "refused", "error")
VERSION = 1                                     # result.json's and index.json's "v"
BRANCH = "previews"
DIR = "previews"                                # previews/<request>/ on the branch
INDEX, README = "index.json", "README.md"
LOCAL_REF = "refs/card-preview/previews"        # where publish fetches the branch head (no remote-tracking ref moved)
KEEP, DAYS = 200, timedelta(days=7)             # more previews or an older one: the next commit is a fresh orphan...
KEEP_AFTER, DAYS_AFTER = 150, timedelta(days=6)  # ... with the newest this many under this age (a rewrite about daily)
TRIES = 5                                       # attempts, each from the head then; 1-2 s, 2-4 s, 3-6 s ... between them
GIT_TIMEOUT = 300
IDENTITY = {"GIT_AUTHOR_NAME": "AutoCroissant", "GIT_AUTHOR_EMAIL": "autocroissant@gmail.com",
            "GIT_COMMITTER_NAME": "AutoCroissant", "GIT_COMMITTER_EMAIL": "autocroissant@gmail.com"}  # git_commit's
BREAKS = ("Cc", "Zl", "Zp")                     # control characters and line or paragraph separators
BRANCH_README = ("# Card previews\n\nMade by TTSCardMaker's `.github/workflows/card-preview.yml` (AutoCroissant "
                 "`commands/card_preview.py`): `previews/<request>/` holds `result.json` and `preview.webp`; "
                 "`index.json` lists them. One commit per preview; rewritten as one fresh commit when it holds more "
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
        raise Unpublished(f"git {args[0]} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Unpublished(f"git {args[0]} failed: {error}") from error


def _fetch(where: str, remote: str) -> str:
    """The branch's head from the remote into LOCAL_REF (one commit, its trees, no blobs); "" when there is no branch."""
    listed = _git(where, "ls-remote", "--heads", remote, f"refs/heads/{BRANCH}").split()
    if not listed:
        return ""
    _git(where, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", "--filter=blob:none", remote,
         f"+refs/heads/{BRANCH}:{LOCAL_REF}")
    return _git(where, "rev-parse", "--verify", f"{LOCAL_REF}^{{commit}}").strip()


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


def _push(where: str, remote: str, commit: str, lease: Optional[str]) -> None:
    """The commit to the branch: a fast forward, or with lease (the head it was built from) a forced update; raises
    Unpublished when refused (the branch moved) or failed."""
    args = ["push", "-q", "--no-thin"] + ([f"--force-with-lease=refs/heads/{BRANCH}:{lease}"] if lease else [])
    _git(where, *args, remote, f"{commit}:refs/heads/{BRANCH}")


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


def publish(out: str, request: str, where: str = ".", remote: str = "origin", png: bool = False,
            now: Optional[datetime] = None) -> dict:
    """
    One commit on the previews branch with previews/<request>/ and its index row, on the branch's head (TRIES times
    from a moved head); a rewrite as a fresh orphan commit when kept() says so, pushed with a lease.

    Args:
        out: The preview's out dir
        request: The client's id (REQUEST)
        where: A checkout of the card repo (its git and credentials; blobless is enough); nothing in it moves but LOCAL_REF
        remote: Its remote name or a URL
        png: Publish preview.png too (the workflow does not: the site shows the WebP)

    Returns:
        {"commit", "parent" (None for an orphan), "rewritten", "kept", "dropped", "result"}; raises Unpublished when no
        push landed
    """
    if not REQUEST.fullmatch(request or ""):
        raise ValueError("the request must be 8-64 of A-Z a-z 0-9 -")
    result = _result_of(out, request)
    files = {}
    for name in (WEBP, PNG) if png else (WEBP,):
        if name in (result.get("files") or {}).values() and isfile(join(out, name)):
            with open(join(out, name), "rb") as fh:
                files[name] = fh.read()
    result["files"] = {kind: name for kind, name in (result.get("files") or {}).items() if name in files}   # what is published
    files[RESULT] = (dumps(result, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    card = clean(result.get("card") or "", MAX_CARD)
    row = {"at": result.get("at"), "card": card, "path": clean(result.get("path") or ""), "state": result.get("state"),
           "psd_blob": (result.get("base") or {}).get("psd_blob", ""), "png_blob": (result.get("base") or {}).get("png_blob", "")}
    entry = _tree(where, [("100644", "blob", _blob(where, data), name) for name, data in files.items()])
    readme = _blob(where, BRANCH_README.encode("utf-8"))
    why = ""
    for attempt in range(TRIES):
        if attempt:
            sleep(uniform(1, 2) * attempt)
        try:
            head = _fetch(where, remote)
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
            _push(where, remote, commit, head if keep is not None else None)
        except Unpublished as error:            # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_preview_branch: attempt {attempt + 1} of {TRIES}: {clean(why, 2000)}", file=stderr)
            continue
        return {"commit": commit, "parent": parent, "rewritten": keep is not None, "kept": len(entries), "dropped": dropped,
                "result": result}
    raise Unpublished(f"not published after {TRIES} tries: {why}")


# ========================
# Public API Functions
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line (module docstring); one line on stdout."""
    parser = ArgumentParser(prog="python -m commands.card_preview_branch",
                            description=f"Publish a card preview's out dir on the {BRANCH} branch, one commit "
                                        "(docs/EDIT_PIPELINE_DESIGN.md section 16).")
    parser.add_argument("--out", required=True, help="the preview's out dir")
    parser.add_argument("--request", required=True, help="the client's id (8-64 of A-Z a-z 0-9 -)")
    parser.add_argument("--git", default=".", help="a checkout of the card repo (its git and credentials)")
    parser.add_argument("--remote", default="origin", help="its remote (name or URL)")
    parser.add_argument("--png", action="store_true", help=f"publish {PNG} too")
    options = parser.parse_args(argv[1:] if args is None else args)
    try:
        done = publish(options.out, options.request, options.git, options.remote, options.png)
    except (ValueError, Unpublished) as error:  # a bad request, or no push landed
        print(f"card_preview_branch: not published: {clean(error, 2000)}", file=stderr)
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
    'kept',
    'main',
    'publish',
    'stamp',
]
