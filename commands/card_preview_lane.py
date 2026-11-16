"""
The warm lane (owner D22, 2026-11-15): card-preview.yml's preview process, once its own press is handed over, stays up
for its editor's next presses and bakes each in process (card_preview.preview: the same bake, fonts, venv and trees, with
no setup and no imports), handed over as the artifact card-preview-<request> of this run by the --hand process
(card_preview_branch.hand_lane: the runtime token never reaches this process). Every press is still dispatched and baked
cold, and that run publishes it on the previews branch; the website shows whichever hand-over comes first (the same
pixels). The lane ends LANE_IDLE s after its last press, near its job's cap, past MAX_RSS, or at a newer press of its
editor it doesn't take.

A press names the lane in its run's title (card-preview.yml's run-name, from the website's inputs.lane):

    Card preview <request> lane <run id>-<check> <card> <edit>

check = the first CHECK hex digits of sha256 of the request, card and edit, one line each (the website's laneCheck): a
title GitHub cut or changed doesn't check. The lane takes a run only when its title names this run and checks, its editor
is this run's, it runs on this run's commit, and the bot's main and the data's records are still the ones this process
read; any other newer run of its editor ends the lane (that run, not served, starts the next one). A run whose press its
named lane served starts none.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from http.client import HTTPException
from json import loads
from os import O_APPEND, O_CREAT, O_WRONLY, close, dup2, environ, makedirs, open as os_open, replace
from os.path import abspath, dirname, join
from re import DOTALL, compile as re_compile
from resource import RUSAGE_SELF, getrusage
from shutil import rmtree
from subprocess import run as run_process
from sys import platform, stderr, stdout
from tempfile import mkdtemp
from time import monotonic, sleep, time
from typing import Callable, NamedTuple, Optional
from urllib.parse import quote

from commands.card_preview_branch import HAND_DONE, LANE_ARTIFACT, LANE_END, _Api, clean
from commands.card_preview_reads import RECORDS, reads

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
CHECK = 16                                      # hex digits of the title's check
WORKFLOW = "card-preview.yml"
RECORDS_ENV = "CARD_RECORDS_DIR"                # psd_edit.RECORDS_DIR: the data's records the first press read
TITLE = re_compile(r"Card preview ([A-Za-z0-9-]{8,64}) lane (\d{1,20})-([0-9a-f]{16}) (.+)", DOTALL)
NAMED = re_compile(r"(\d{1,20})-[0-9a-f]{16}")  # inputs.lane


class Press(NamedTuple):
    """A run's press as its title gives it (card and edit None: it names a lane but doesn't check)."""
    request: str
    lane: int
    card: Optional[str]
    edit: Optional[str]


# ========================
# The title
# ========================
def check_of(request: str, card: str, edit: str) -> str:
    """The title's check: sha256 of the request, card and edit (UTF-8, one line each), its first CHECK hex digits."""
    return sha256(f"{request}\n{card}\n{edit}".encode("utf-8")).hexdigest()[:CHECK]


def press_of(title: str) -> Optional[Press]:
    """A run title's press, None when it names no lane; the card is whatever precedes the edit's " {" that checks."""
    match = TITLE.fullmatch(title or "")
    if not match:
        return None
    request, lane, check, rest = match.groups()
    at = rest.find(" {")
    while at > 0:
        if check_of(request, rest[:at], rest[at + 1:]) == check:
            return Press(request, int(lane), rest[:at], rest[at + 1:])
        at = rest.find(" {", at + 1)
    return Press(request, int(lane), None, None)


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


def fits(run: dict, press: Optional[Press], me: int, actor: str, head: str) -> str:
    """Why the lane doesn't take a newer run of its editor ("" when it does)."""
    if press is None:
        return "a newer press named no lane"
    if press.lane != me:
        return f"a newer press named lane {press.lane}"
    if press.card is None:
        return "a newer press's title doesn't check"
    if run.get("status") == "completed":
        return "a newer press was done before the lane saw it"
    if (run.get("triggering_actor") or run.get("actor") or {}).get("login") != actor:
        return "a newer press by another editor"
    if run.get("head_sha") != head:
        return "the card repo moved"
    return ""


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
    One press: its out dir made (the hand-over's artifact created then), its reads (card_preview_reads, as the Reads step)
    beside the bot's check, then bake() into it; <out>.code after. Returns "" when baked, else why the lane ends.
    """
    out, seen = join(lane, press.request), monotonic()
    makedirs(out, exist_ok=True)
    records = mkdtemp(prefix="card-preview-lane-")
    why, code = "", "skipped"
    try:
        try:
            edit = loads(press.edit)
        except ValueError:
            edit = {}
        fields = {k: v for k, v in edit.items() if not str(k).startswith("_")} if isinstance(edit, dict) else {}
        with ThreadPoolExecutor(1) as pool:
            done = pool.submit(reads, press.card, fields, clone, repository, head, records)
            why = bot_moved(api, bot_repo, bot)
            done.result()
        for name in RECORDS if not why else ():
            fresh = _read(join(records, name))
            why = why or ("the data's records not read" if fresh is None else "the data's records changed" if fresh != kept[name] else "")
        if why:
            return why
        read = monotonic()
        url = f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{run['id']}"
        result = bake(press.card, press.edit.encode("utf-8")[:limit + 1], out, repository, clone=clone,
                      request=press.request, head=head, run=url)
        code = "0" if result.get("state") != "error" else "2" if str(result.get("reason", "")).startswith("input:") else "1"
        made, now = _epoch(run.get("created_at")), monotonic()
        _say(f"{press.request} ({clean(press.card, 80)}): {result.get('state')} in {now - seen:.2f} s (reads {read - seen:.2f} s, "
             f"bake {now - read:.2f} s)" + (f"; seen {time() - (now - seen) - made:.1f} s after its run was made (to the "
                                            "second)" if made else ""))
        return ""
    finally:
        with open(out + ".code.tmp", "w", encoding="utf-8") as fh:     # the hand-over: done (none when skipped)
            fh.write(code + "\n")
        replace(out + ".code.tmp", out + ".code")
        rmtree(records, ignore_errors=True)


def listen(api: _Api, me: int, actor: str, *, until: float, **press) -> str:
    """The lane's loop: its editor's runs listed every POLL s (conditional GETs), each newer one taken (take) or the
    lane's end (fits). Returns why it ended (press: take's keywords)."""
    runs = f"/repos/{press['repository']}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&actor={quote(actor)}&per_page={LISTED}"
    last, failing, seen, listed = monotonic(), None, set(), []
    while True:
        if monotonic() - last > LANE_IDLE:
            return f"idle {LANE_IDLE} s"
        if time() > until:
            return "the job's end is near"
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
        newer = sorted((r for r in listed if int(r.get("id") or 0) > me and r["id"] not in seen), key=lambda r: int(r["id"]))
        if newer:
            run = newer[0]
            seen.add(run["id"])
            pressed = press_of(run.get("display_title", ""))
            why = fits(run, pressed, me, actor, press["head"]) or take(api, pressed, run, **press)
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
        head: This run's commit of the card repo
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
    'bot_moved',
    'check_of',
    'fits',
    'listen',
    'press_of',
    'serve',
    'served_elsewhere',
    'take',
]
