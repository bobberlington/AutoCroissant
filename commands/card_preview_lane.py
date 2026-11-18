"""
The warm lane (owner D22, 2026-11-15): card-preview.yml's preview process, once its own press is handed over, stays up
for its editor's next presses and bakes each in process (card_preview.preview: the same bake, fonts, venv and trees, with
no setup and no imports), handed over as the artifact card-preview-<request> of this run by the --hand process
(card_preview_branch.hand_lane: the runtime token never reaches this process). Every press is still dispatched and baked
cold, and that run publishes it on the previews branch; the website shows whichever hand-over comes first (the same
pixels). The lane ends LANE_IDLE s after its last press, near its job's cap, past MAX_RSS, or at a newer press of its
editor it doesn't take.

A press names the lane in its run's title (card-preview.yml's run-name, from the website's inputs.lane):

    Card preview <request> lane <run id>-<blob>

blob = the git blob the website posts beside the dispatch (press_blob: the request, card and edit, one line each; no
ref, so no push): GitHub cuts run titles at 512 characters, so the press itself never rides in one. The lane takes a run
only when its title names this run, its editor is this run's, its blob hashes to its sha and names its request, and the
bot's main and the data's records are still the ones this process read. A press on a newer commit of the card repo is
taken there (follow: its trees fetched into the clone) when KEPT is the same there; from then on it is the lane's commit.
Any other newer run of its editor ends the lane (that run, not served, starts the next one). A run whose press its named
lane served starts none. A new run's check suite on the card repo's main, listed about 0.65 s after the run is made
where the runs list takes 3-5 s, brings it sooner (Suites: one GraphQL query for its run).
"""
from base64 import b64decode
from binascii import Error as Base64Error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha1
from http.client import HTTPException
from json import loads
from os import O_APPEND, O_CREAT, O_WRONLY, close, dup2, environ, makedirs, open as os_open, replace
from os.path import abspath, dirname, join
from re import compile as re_compile
from resource import RUSAGE_SELF, getrusage
from shutil import rmtree
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import platform, stderr, stdout
from tempfile import mkdtemp
from time import monotonic, sleep, time
from typing import Callable, NamedTuple, Optional
from urllib.parse import quote

from commands.card_preview_branch import HAND_DONE, LANE_ARTIFACT, LANE_END, _Api, clean
from commands.card_preview_reads import BLOBS_ENV, RECORDS, reads

# ========================
# Configuration
# ========================
LANE_IDLE = 240                                 # s after its last press (owner: about 4 minutes)
MIN_LEFT = 60                                   # s of the job's time a lane needs to start
POLL = 0.3                                      # s between listings of its editor's runs (a 304 isn't counted)
POLL_SLOW, LOW = 3, 100                         # s between them while the rate limit left is under LOW
LISTED = 10                                     # runs per listing
FAILING = 30                                    # s of failed listings that end the lane
HAND_WAIT = 10                                  # s for the first press's hand-over (hand.done ok)
MAX_RSS = 2 << 30                               # bytes of peak memory past which it takes no more presses
BLOB_WAIT, BLOB_POLL = 5, 0.25                  # s a press's blob may lag its run (posted beside the dispatch)
MAX_BLOB = 1 << 16                              # bytes of a press's blob (an edit is card_preview.MAX_EDIT at most)
WORKFLOW = "card-preview.yml"
RECORDS_ENV = "CARD_RECORDS_DIR"                # psd_edit.RECORDS_DIR: the data's records the first press read
TITLE = re_compile(r"Card preview ([A-Za-z0-9-]{8,64}) lane (\d{1,20})-([0-9a-f]{40})")
NAMED = re_compile(r"(\d{1,20})-[0-9a-f]{40}")  # inputs.lane
KEPT = ("MDW/Fonts", ".github/workflows/card-preview.yml", ".github/actions")    # the same at a commit the lane follows
SHA = re_compile(r"[0-9a-f]{40}")
FETCH_TIMEOUT = 20                              # s for a newer commit's trees
ACTIONS_APP, PAGE = 15368, 100                  # GitHub Actions' app id (a check suite per run); suites per listing
SUITE_LOOKS, SUITE_FAILS = 3, 3                 # looks at a suite whose run isn't read; failures that turn Suites off
SUITE_RUNS = ("query($ids:[ID!]!){nodes(ids:$ids){... on CheckSuite{databaseId status creator{login} commit{oid} "
              "workflowRun{databaseId displayTitle event createdAt file{path}}}}}")


class Press(NamedTuple):
    """A run's press as its title names it: the request, the lane, the sha of its blob."""
    request: str
    lane: int
    blob: str


# ========================
# The press
# ========================
def press_blob(request: str, card: str, edit: str) -> tuple:
    """(bytes, sha) of a press's blob: the request, card and edit, UTF-8, one line each (the website's laneBlob)."""
    data = f"{request}\n{card}\n{edit}".encode("utf-8")
    return data, sha1(b"blob %d\0" % len(data) + data).hexdigest()


def press_of(title: str) -> Optional[Press]:
    """A run title's press, None when it names no lane."""
    match = TITLE.fullmatch(title or "")
    return Press(match.group(1), int(match.group(2)), match.group(3)) if match else None


def read_press(api: _Api, repository: str, press: Press) -> tuple:
    """(card, edit, "") from the press's blob (a 404 tried again for BLOB_WAIT s), else (None, None, why)."""
    deadline = monotonic() + BLOB_WAIT
    status, _, body = api.get(f"/repos/{repository}/git/blobs/{press.blob}")
    while status == 404 and monotonic() < deadline:
        sleep(BLOB_POLL)
        status, _, body = api.get(f"/repos/{repository}/git/blobs/{press.blob}")
    if status != 200:
        return None, None, f"the press's blob not read ({status})"
    try:
        reply = loads(body)
        data = b64decode(reply["content"]) if reply.get("encoding") == "base64" and reply.get("size", 0) <= MAX_BLOB else b""
    except (ValueError, KeyError, TypeError, AttributeError, Base64Error):
        data = b""
    if sha1(b"blob %d\0" % len(data) + data).hexdigest() != press.blob:
        return None, None, "the press's blob doesn't hash to its sha"
    try:
        request, card, edit = data.decode("utf-8").split("\n", 2)
    except ValueError:
        return None, None, "the press's blob isn't a press"
    return (card, edit, "") if request == press.request else (None, None, "the press's blob names another request")


# ========================
# The lane
# ========================
def _say(text: str) -> None:
    """A line of the lane's log (lane.log: its stdout once _log_to ran)."""
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]} lane: {text}", flush=True)


def _epoch(stamp: str) -> Optional[float]:
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _read(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _handed(out: str) -> str:
    """"" once the first press's hand-over is done (hand.done ok, HAND_WAIT s at most); else why not."""
    done, deadline = join(dirname(abspath(out)), HAND_DONE), monotonic() + HAND_WAIT
    while (state := (_read(done) or b"").strip()) == b"" and monotonic() < deadline:
        sleep(0.05)
    return "" if state == b"ok" else f"the first hand-over said {clean(state.decode('utf-8', 'replace') or 'nothing', 40)}"


def served_elsewhere(api: _Api, repository: str, named: str, request: str) -> str:
    """Why this run starts no lane: the lane its press named handed it over (its artifact listed); "" otherwise."""
    match = NAMED.fullmatch(named or "")
    if not match:
        return ""
    name = LANE_ARTIFACT + request
    status, _, body = api.get(f"/repos/{repository}/actions/runs/{match.group(1)}/artifacts?name={name}")
    if status == 200 and any(a.get("name") == name and not a.get("expired") for a in loads(body).get("artifacts", [])):
        return f"lane {match.group(1)} served this press"
    return ""


def bot_moved(api: _Api, bot_repo: str, bot: str) -> str:
    """Why the bot's main is not this process's commit (bot) any more; "" while it is (a 304 isn't counted)."""
    status, _, body = api.get(f"/repos/{bot_repo}/git/ref/heads/main", conditional=True)
    if status == 304:
        return ""
    if status == 200:
        return "" if loads(body).get("object", {}).get("sha") == bot else "the bot's main moved"
    return f"the bot's main not read ({status})"


def fits(run: dict, press: Optional[Press], me: int, actor: str) -> str:
    """Why the lane doesn't take a newer run of its editor ("" when it does; its commit: follow)."""
    if press is None:
        return "a newer press named no lane"
    if press.lane != me:
        return f"a newer press named lane {press.lane}"
    if run.get("status") == "completed":
        return "a newer press was done before the lane saw it"
    if (run.get("triggering_actor") or run.get("actor") or {}).get("login") != actor:
        return "a newer press by another editor"
    return ""


def suite_run(node: dict, me: int, actor: str) -> Optional[dict]:
    """A check suite's run (a GraphQL CheckSuite) in the runs list's shape, when it is a newer press of actor's (the
    runs list's filters: this workflow, dispatched, by actor); else None."""
    run = (node or {}).get("workflowRun") or {}
    if (run.get("file") or {}).get("path") != f".github/workflows/{WORKFLOW}" or run.get("event") != "workflow_dispatch" \
            or ((node.get("creator") or {}).get("login") != actor) or int(run.get("databaseId") or 0) <= me:
        return None
    return {"id": int(run["databaseId"]), "display_title": run.get("displayTitle") or "",
            "status": str(node.get("status") or "").lower(), "head_sha": (node.get("commit") or {}).get("oid") or "",
            "triggering_actor": {"login": actor}, "created_at": run.get("createdAt"),
            "suite": int(node.get("databaseId") or 0)}


class Suites:
    """The lane's quick look for its editor's presses (no ref, no website change): GitHub Actions' check suites on the
    card repo's main (where the website dispatches), listed about 0.65 s after a run is made where the runs list takes
    3-5 s; the new ones' runs read by one GraphQL query (suite_run). Off after a refusal or SUITE_FAILS failures (the
    runs list still brings every press); never raises."""

    def __init__(self, api: _Api, repository: str, me: int, actor: str):
        self.api, self.me, self.actor, self.off, self.fails = api, me, actor, "", 0
        self.path = f"/repos/{repository}/commits/heads/main/check-suites?app_id={ACTIONS_APP}&per_page={PAGE}"
        self.looks = {}                         # node id -> looks (SUITE_LOOKS: done)

    def _list(self, page: int = 1) -> Optional[list]:
        status, _, body = self.api.get(self.path + (f"&page={page}" if page > 1 else ""), conditional=True)
        if status == 304:
            return None
        if status != 200:
            raise PermissionError(f"listed {status}") if status in (401, 403, 404, 422) else OSError(f"listed {status}")
        reply = loads(body)
        found, total = reply["check_suites"], int(reply.get("total_count") or 0)
        if page == 1 and total > PAGE:          # a busy main: its newest on the last page
            found += self._list(-(-total // PAGE)) or []
        return found

    def runs(self) -> list:
        """Newer presses of its editor's among main's new suites ([] when none or off)."""
        if self.off:
            return []
        try:
            for suite in self._list() or []:
                self.looks.setdefault(str(suite["node_id"]), 0)
            ids = [node for node, looks in self.looks.items() if looks < SUITE_LOOKS][-PAGE:]
            if not ids:
                return []
            status, _, body = self.api.post("/graphql", {"query": SUITE_RUNS, "variables": {"ids": ids}})
            if status not in (200, 401, 403):
                raise OSError(f"GraphQL {status}")
            reply = loads(body) if status == 200 else {}
            nodes, errors = (reply.get("data") or {}).get("nodes"), reply.get("errors") or []
            if not isinstance(nodes, list) or any(e.get("type") == "FORBIDDEN" for e in errors):
                raise PermissionError(f"GraphQL {status}: " + clean((errors or [{}])[0].get("message", ""), 80))
            found = []
            for node, suite in zip(ids, nodes):     # a suite not read yet: null (looked at again)
                read = bool((suite or {}).get("workflowRun"))
                self.looks[node] = SUITE_LOOKS if read else self.looks[node] + 1
                found.append(suite_run(suite, self.me, self.actor) if read else None)
            self.fails = 0
            return [run for run in found if run]
        except PermissionError as error:
            self.off = str(error)
        except Exception as error:              # noqa: BLE001 - the runs list still brings it
            self.fails += 1
            self.off = f"{type(error).__name__}: {clean(error, 80)}" if self.fails >= SUITE_FAILS else ""
        if self.off:
            _say(f"check suites off ({self.off}): the runs list alone")
        return []


def follow(clone: str, head: str, new: str) -> str:
    """Why the lane can't take a press at the card repo's commit new ("" when it can): new's trees fetched into the
    clone (shallow, blobless; the repo is public: no token) and KEPT there as at head (the cold run's fonts and steps)."""
    if new == head:
        return ""
    if not SHA.fullmatch(new or ""):
        return "the card repo moved"
    env = {**environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
    try:
        run_process(["git", "-C", clone, "fetch", "-q", "--depth", "1", "--filter=blob:none", "--no-tags", "origin", new],
                    capture_output=True, env=env, timeout=FETCH_TIMEOUT, check=True)
        kept = [run_process(["git", "-C", clone, "ls-tree", sha, "--", *KEPT], capture_output=True, env=env, timeout=30,
                            check=True).stdout for sha in (head, new)]
    except (CalledProcessError, TimeoutExpired, OSError):
        return "the card repo moved (not fetched)"
    return "" if kept[0] == kept[1] else "the card repo moved (its fonts or the preview's workflow)"


def _rss() -> int:
    return getrusage(RUSAGE_SELF).ru_maxrss * (1 if platform == "darwin" else 1024)


def _log_to(path: str) -> None:
    """This process's stdout and stderr to path from now on (the job's step read the first press's)."""
    stdout.flush()
    stderr.flush()
    fd = os_open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
    dup2(fd, 1)
    dup2(fd, 2)
    close(fd)


def take(api: _Api, press: Press, run: dict, *, bake: Callable, lane: str, clone: str, repository: str, head: str,
         kept: dict, bot_repo: str, bot: str, limit: int) -> str:
    """
    One press: its out dir made (the hand-over's artifact created then), its blob (read_press), its reads
    (card_preview_reads, as the Reads step) beside the bot's check, then bake() into it; <out>.code after. Returns ""
    when baked, else why the lane ends.
    """
    out, seen = join(lane, press.request), monotonic()
    makedirs(out, exist_ok=True)
    records = mkdtemp(prefix="card-preview-lane-")
    why, code = "", "skipped"
    try:
        card, text, why = read_press(api, repository, press)
        if why:
            return why
        blob = monotonic()
        try:
            edit = loads(text)
        except ValueError:
            edit = {}
        fields = {k: v for k, v in edit.items() if not str(k).startswith("_")} if isinstance(edit, dict) else {}
        with ThreadPoolExecutor(1) as pool:
            done = pool.submit(reads, card, fields, clone, repository, head, records, environ.get(BLOBS_ENV, ""))
            why = bot_moved(api, bot_repo, bot)
            done.result()
        for name in RECORDS if not why else ():
            fresh = _read(join(records, name))
            why = why or ("the data's records not read" if fresh is None else "the data's records changed" if fresh != kept[name] else "")
        if why:
            return why
        read = monotonic()
        url = f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{run['id']}"
        result = bake(card, text.encode("utf-8")[:limit + 1], out, repository, clone=clone,
                      request=press.request, head=head, run=url)
        code = "0" if result.get("state") != "error" else "2" if str(result.get("reason", "")).startswith("input:") else "1"
        made, now = _epoch(run.get("created_at")), monotonic()
        _say(f"{press.request} ({clean(card, 80)}): {result.get('state')} in {now - seen:.2f} s (blob {blob - seen:.2f} s, "
             f"reads {read - blob:.2f} s, "
             f"bake {now - read:.2f} s)" + (f"; seen {time() - (now - seen) - made:.1f} s after its run was made (to the "
                                            "second)" + (", by its check suite" if run.get("suite") else "") if made else ""))
        return ""
    finally:
        with open(out + ".code.tmp", "w", encoding="utf-8") as fh:     # the hand-over: done (none when skipped)
            fh.write(code + "\n")
        replace(out + ".code.tmp", out + ".code")
        rmtree(records, ignore_errors=True)


def listen(api: _Api, me: int, actor: str, *, until: float, **press) -> str:
    """The lane's loop: main's check suites (Suites) and its editor's runs listed every POLL s (conditional GETs; the
    runs list when the suites brought none new), each newer run taken (take; at its commit: follow) or the lane's end
    (fits). Returns why it ended (press: take's keywords)."""
    runs = f"/repos/{press['repository']}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&actor={quote(actor)}&per_page={LISTED}"
    last, failing, seen, listed, found = monotonic(), None, set(), [], {}
    suites = Suites(api, press["repository"], me, actor)

    def moved(new: str) -> str:                 # a press at new: the lane's commit from now on
        started, why = monotonic(), follow(press.get("clone", ""), press["head"], new)
        if not why and new != press["head"]:
            _say(f"follows the card repo {press['head'][:8]} -> {new[:8]} ({monotonic() - started:.2f} s)")
            press["head"] = new
        return why
    while True:
        if monotonic() - last > LANE_IDLE:
            return f"idle {LANE_IDLE} s"
        if time() > until:
            return "the job's end is near"
        found.update((r["id"], r) for r in suites.runs() if r["id"] not in found)
        if all(r in seen for r in found):       # none new from main's check suites: the runs list
            try:
                status, _, body = api.get(runs, conditional=True)
            except (OSError, HTTPException, ValueError):
                status = 0
            if status in (200, 304):
                failing = None
            else:
                failing = failing or monotonic()
                if monotonic() - failing > FAILING:
                    return f"the runs list failed for {FAILING} s ({status})"
            if status == 200:
                listed = loads(body).get("workflow_runs", [])
        newer = sorted((r for r in [*found.values(), *listed] if int(r.get("id") or 0) > me and r["id"] not in seen),
                       key=lambda r: int(r["id"]))
        if newer:
            run = newer[0]
            seen.add(run["id"])
            pressed = press_of(run.get("display_title", ""))
            why = fits(run, pressed, me, actor) or moved(run.get("head_sha") or "") or take(api, pressed, run, **press)
            if why:
                return why
            last = monotonic()
            if _rss() > MAX_RSS:
                return f"peak memory {_rss() >> 20} MB"
            continue                            # the list again at once: the others' states as of now
        sleep(POLL_SLOW if api.left is not None and api.left < LOW else POLL)


def serve(lane: str, *, bake: Callable, out: str, clone: str, repository: str, head: str, request: str, named: str,
          until: float, code: int, limit: int) -> str:
    """
    The lane after the first press (card_preview.main --lane): listen() once the first press went well, was handed over
    and is the bot's main's. Never raises; <lane>.end says why it ended.

    Args:
        lane: Its folder (each press's out dir in it; card_preview_branch.hand_lane watches it)
        bake: card_preview.preview
        out: The first press's out dir (hand.done beside it)
        head: This run's commit of the card repo (the lane's first: it follows newer ones, listen)
        request, named: The first press's request and inputs.lane
        until: Its end at the latest (epoch s: the job's cap less a margin)
        code: The first press's exit code (a lane only after 0)
        limit: card_preview.MAX_EDIT
    """
    lane, api, why, logged = abspath(lane), None, "", False
    makedirs(lane, exist_ok=True)
    try:
        me, actor, token = environ.get("GITHUB_RUN_ID", ""), environ.get("GITHUB_TRIGGERING_ACTOR", ""), environ.get("GITHUB_TOKEN", "")
        folder = environ.get(RECORDS_ENV, "")
        kept = {name: _read(join(folder, name)) if folder else None for name in RECORDS}
        why = ("the first press failed" if code != 0 else "not on a runner" if not (me.isdigit() and actor and token and head)
               else "the job's time is short" if until - time() < MIN_LEFT
               else "the data's records not read" if None in kept.values() else _handed(out))
        if why:
            return why
        _log_to(lane + ".log")
        logged = True
        bot_repo = environ.get("BOT_REPO", "bobberlington/AutoCroissant")
        bot = run_process(["git", "-C", dirname(dirname(abspath(__file__))), "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=30).stdout.strip()
        api = _Api(environ.get("GITHUB_API_URL", "https://api.github.com"), token)
        why = served_elsewhere(api, repository, named, request) or bot_moved(api, bot_repo, bot)
        if why:
            return why
        _say(f"up for {clean(actor, 40)}'s next presses ({LANE_IDLE} s idle at most)")
        why = listen(api, int(me), actor, bake=bake, lane=lane, clone=clone, repository=repository, head=head, kept=kept,
                     bot_repo=bot_repo, bot=bot, until=until, limit=limit)
        return why
    except Exception as error:                  # noqa: BLE001 - the lane only ends
        why = f"error: {type(error).__name__}: {clean(error, 300)}"
        return why
    finally:
        if api is not None:
            api.close()
        if logged:
            _say(f"ended: {why}")
        with open(lane + LANE_END, "w", encoding="utf-8") as fh:
            fh.write(why + "\n")


# ========================
# Module Exports
# ========================
__all__ = [
    'Press',
    'Suites',
    'bot_moved',
    'fits',
    'follow',
    'listen',
    'press_blob',
    'press_of',
    'read_press',
    'serve',
    'served_elsewhere',
    'suite_run',
    'take',
]
