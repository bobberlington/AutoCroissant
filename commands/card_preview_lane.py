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

blob = the git blob the website posts beside the dispatch (press_blob: the request, card and edit, one line each):
GitHub cuts run titles at 512 characters, so the press itself never rides in one. Once that blob is posted, the
website also rings the lane's doorbell: a commit status at the lane's commit (this run's GITHUB_SHA; no push, no
workflow, no Discord post), context CONTEXT, its description:

    <run id> <request> <blob>

listed in about a second where a new run takes 2-6 s to show (rung). The lane lists its commit's statuses (one
conditional GET; a 304 isn't counted) and takes a press rung by its editor (the status's creator) at once (its blob,
the card repo's main read in place of the run's commit), and a run only when its title names this run and its editor
is this run's; each only when its blob hashes to its sha and names its request, and the bot's main and the data's
records are still the ones this process read. A press on a newer commit of the card repo is taken there (follow: its
trees fetched into the clone) when KEPT is the same there; from then on it is the lane's commit (its doorbell stays at
its first). A rung press's run, listed later, is served; any other newer run of its editor ends the lane (that run, not
served, starts the next one). A run whose press its named lane served starts none. Statuses can't be deleted: none
needs a clean-up (the lane's run id in each, so a later lane on that commit reads none of them).

The website also rings a card its editor starts editing (its PSD blob's first 8-40 hex digits, a status there too),
so the lane reads that PSD, its PNG and the status chart into its clone before the press comes (warm: one background
thread, one card at a time, never while a press is being taken; a press joins a read still going):

    <run id> next <PSD blob>
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
from threading import Event, Lock, Thread
from time import monotonic, sleep, time
from typing import Callable, NamedTuple, Optional
from urllib.parse import quote

from commands.card_preview_branch import HAND_DONE, LANE_ARTIFACT, LANE_END, _Api, clean
from commands.card_preview_reads import BLOBS_ENV, RECORDS, missing, paths_of, read_blob, read_card, read_records, tree_of

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
CONTEXT = "preview lane"                        # the doorbell's commit statuses (the website's)
RUNG = re_compile(r"(\d{1,20}) ([A-Za-z0-9-]{8,64}) ([0-9a-f]{40})")         # <run id> <request> <blob>
NAMED = re_compile(r"(\d{1,20})-[0-9a-f]{40}")  # inputs.lane
NEXT = re_compile(r"(\d{1,20}) next ([0-9a-f]{8,40})")                       # a card its editor started editing
MAX_NEXT = 20                                   # such cards a lane reads at most
KEPT = ("MDW/Fonts", ".github/workflows/card-preview.yml", ".github/actions")    # the same at a commit the lane follows
SHA = re_compile(r"[0-9a-f]{40}")
FETCH_TIMEOUT = 20                              # s for a newer commit's trees


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


def _rang(body: bytes, me: int, actor: str, pattern) -> list:
    """pattern's matches among the statuses (a commit's statuses list) actor rang at lane me's doorbell."""
    try:
        found = [pattern.fullmatch(s.get("description") or "") for s in loads(body)
                 if s.get("context") == CONTEXT and (s.get("creator") or {}).get("login") == actor]
    except (ValueError, TypeError, AttributeError):
        return []
    return [m for m in found if m and int(m.group(1)) == me]


def rung(body: bytes, me: int, actor: str) -> list:
    """The presses actor rang at lane me's doorbell, by request; a request rung with two blobs is none."""
    found = {(m.group(2), m.group(3)) for m in _rang(body, me, actor, RUNG)}
    twice = {r for r, _ in found if sum(r == q for q, _ in found) > 1}
    return sorted(Press(r, me, b) for r, b in found if r not in twice)


def hints_of(body: bytes, me: int, actor: str) -> list:
    """The PSD blobs actor rang as next at lane me's doorbell, oldest first."""
    return list(dict.fromkeys(m.group(2) for m in _rang(body, me, actor, NEXT)))[::-1]


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


def bot_moved(api: _Api, bot_repo: str, bot: str, what: str = "the bot's main") -> str:
    """Why bot_repo's main is not the commit bot (this process's) any more; "" while it is (a 304 isn't counted)."""
    status, _, body = api.get(f"/repos/{bot_repo}/git/ref/heads/main", conditional=True)
    if status == 304:
        return ""
    if status == 200:
        return "" if loads(body).get("object", {}).get("sha") == bot else f"{what} moved"
    return f"{what} not read ({status})"


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


def warm(blob: str, tree: dict, clone: str, repository: str, pending: dict) -> str:
    """A hinted card's PSD, its PNG and the status chart into the clone (card_preview_reads.read_blob: verified), those
    it lacks; pending[sha] (an Event) set once each is read or failed. Returns a line for the lane's log."""
    started = monotonic()
    psds = [path for path, sha in tree.items() if sha.startswith(blob) and path.lower().endswith(".psd")]
    if len(psds) != 1:
        return f"next {blob[:8]}: {len(psds)} PSDs"
    paths = paths_of(tree, psds[0], {"ability": ""})   # the chart too (an ability edit's)
    shas = sorted(missing(clone, [tree[p] for p in paths])) if paths else []
    events = {sha: Event() for sha in shas}
    pending.update(events)

    def one(sha: str) -> int:
        try:
            return len(read_blob(repository, sha, clone))
        except Exception:                       # noqa: BLE001 - the press reads it
            return -1
        finally:
            events[sha].set()
    with ThreadPoolExecutor(max(len(shas), 1)) as pool:
        sizes = list(pool.map(one, shas))
    return (f"next {blob[:8]} ({clean(psds[0], 80)}): {sum(s > 0 for s in sizes)} of {len(shas)} read, "
            f"{sum(max(s, 0) for s in sizes) / 1e6:.1f} MB in {monotonic() - started:.2f} s")


class Warm:
    """The lane's hinted cards (hints_of), each read by warm() on a daemon thread, one at a time (lock), while no press
    is being taken (idle); pending: {sha: Event} for a press's reads to join."""

    def __init__(self, clone: str, repository: str, head: str):
        self.clone, self.repository, self.head = clone, repository, head
        self.pending, self.asked, self.idle, self.lock, self.tree = {}, set(), Event(), Lock(), None
        self.idle.set()

    def ask(self, blobs: list) -> None:
        for blob in blobs:
            if blob not in self.asked and len(self.asked) < MAX_NEXT:
                self.asked.add(blob)
                Thread(target=self._one, args=(blob,), name=f"lane-next-{blob[:8]}", daemon=True).start()

    def _one(self, blob: str) -> None:
        with self.lock:
            self.idle.wait()
            try:
                self.tree = self.tree or tree_of(self.clone, self.head)
                _say(warm(blob, self.tree, self.clone, self.repository, self.pending))
            except Exception as error:          # noqa: BLE001 - a press reads as ever
                _say(f"next {blob[:8]}: {type(error).__name__}: {clean(error, 200)}")


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


def main_of(api: _Api, repository: str) -> tuple:
    """(the sha of repository's main, "") or ("", why not)."""
    status, _, body = api.get(f"/repos/{repository}/git/ref/heads/main")
    sha = (loads(body).get("object") or {}).get("sha", "") if status == 200 else ""
    return (sha, "") if SHA.fullmatch(sha or "") else ("", f"the card repo's main not read ({status})")


def _fields(text: str) -> dict:
    """The edit's fields (a JSON object's keys but "_" ones; {} for anything else)."""
    try:
        edit = loads(text)
    except ValueError:
        edit = {}
    return {k: v for k, v in edit.items() if not str(k).startswith("_")} if isinstance(edit, dict) else {}


def take(api: _Api, press: Press, run: Optional[dict], *, bake: Callable, lane: str, clone: str, repository: str, head: str,
         kept: dict, bot_repo: str, bot: str, limit: int, sides: tuple, hints: Optional[Warm] = None,
         moved: Optional[Callable] = None) -> str:
    """
    One press (run: its run, None when rung): its out dir made (the hand-over's artifact created then), its blob
    (read_press) beside the data's records, the bot's check and (a rung one's) the card repo's main (sides: two more
    _Api, their own kept connections), then the card's reads (card_preview_reads.read_card; joining hints' reads; on a
    newer main, moved(main) and its reads there), then bake() into it; <out>.code after. Returns "" when baked, else why
    the lane ends.
    """
    out, seen = join(lane, press.request), monotonic()
    makedirs(out, exist_ok=True)
    records = mkdtemp(prefix="card-preview-lane-")
    why, code = "", "skipped"
    blobs, pending = environ.get(BLOBS_ENV, ""), hints and hints.pending
    if hints is not None:
        hints.idle.clear()                      # no new hint read while the press is taken
    try:
        with ThreadPoolExecutor(3) as pool:
            got = pool.submit(read_records, repository, records)
            bots = pool.submit(bot_moved, sides[0], bot_repo, bot)
            mains = None if run else pool.submit(main_of, sides[1], repository)
            card, text, why = read_press(api, repository, press)
            blob, fields = monotonic(), {} if why else _fields(text)
            if not why:                         # the card's own files (most often in the clone: a hint read them)
                read_card(card, fields, clone, repository, head, blobs, pending)
            got.result()
            why = why or bots.result()
            main, w = mains.result() if mains else (head, "")
            why = why or w
        if why:
            return why
        if main != head:                        # a rung press on a newer main
            why = moved(main) if moved else "the card repo's main moved"
            if not why:
                head = main
                read_card(card, fields, clone, repository, head, blobs, pending)
        for name in RECORDS if not why else ():
            fresh = _read(join(records, name))
            why = why or ("the data's records not read" if fresh is None else "the data's records changed" if fresh != kept[name] else "")
        if why:
            return why
        read = monotonic()
        url = f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{run['id']}" if run else ""
        result = bake(card, text.encode("utf-8")[:limit + 1], out, repository, clone=clone,
                      request=press.request, head=head, run=url)
        code = "0" if result.get("state") != "error" else "2" if str(result.get("reason", "")).startswith("input:") else "1"
        made, now = _epoch(run.get("created_at")) if run else None, monotonic()
        _say(f"{press.request} ({clean(card, 80)}): {result.get('state')} in {now - seen:.2f} s (blob {blob - seen:.2f} s, "
             f"reads {read - blob:.2f} s, "
             f"bake {now - read:.2f} s)" + (f"; seen {time() - (now - seen) - made:.1f} s after its run was made (to the "
                                            "second)" if made else "; rung" if run is None else ""))
        return ""
    finally:
        with open(out + ".code.tmp", "w", encoding="utf-8") as fh:     # the hand-over: done (none when skipped)
            fh.write(code + "\n")
        replace(out + ".code.tmp", out + ".code")
        rmtree(records, ignore_errors=True)
        if hints is not None:
            hints.idle.set()


def listen(api: _Api, me: int, actor: str, *, until: float, **press) -> str:
    """The lane's loop: its doorbell (its first commit's statuses) and its editor's runs listed every POLL s
    (conditional GETs), each card rung as next read in the background (Warm), each press rung taken (take), then each
    newer run taken (at its commit: follow), served (rung before) or the lane's end (fits). Returns why it ended
    (press: take's keywords)."""
    runs = f"/repos/{press['repository']}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&actor={quote(actor)}&per_page={LISTED}"
    bells = f"/repos/{press['repository']}/commits/{press['head']}/statuses?per_page=100"     # the website's laneHead
    last, failing, seen, listed, rang, taken = monotonic(), None, set(), [], [], set()
    hints = Warm(press.get("clone", ""), press["repository"], press["head"])

    def moved(new: str) -> str:                 # a press at new: the lane's commit from now on
        started, why = monotonic(), follow(press.get("clone", ""), press["head"], new)
        if not why and new != press["head"]:
            _say(f"follows the card repo {press['head'][:8]} -> {new[:8]} ({monotonic() - started:.2f} s)")
            press["head"] = hints.head = new
            hints.tree = None
        return why
    while True:
        if monotonic() - last > LANE_IDLE:
            return f"idle {LANE_IDLE} s"
        if time() > until:
            return "the job's end is near"
        try:
            status, _, body = api.get(bells, conditional=True)
            rang = rung(body, me, actor) if status == 200 else rang
            hints.ask(hints_of(body, me, actor) if status == 200 else [])
        except (OSError, HTTPException, ValueError):
            pass                                # the runs list still brings it
        pressed, run = next((p for p in rang if p.request not in taken), None), None
        if pressed is None:
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
            if not newer:
                sleep(POLL_SLOW if api.left is not None and api.left < LOW else POLL)
                continue
            run = newer[0]
            seen.add(run["id"])
            pressed = press_of(run.get("display_title", ""))
            if pressed is not None and pressed.request in taken:
                continue                        # rung: served
            why = fits(run, pressed, me, actor) or moved(run.get("head_sha") or "")
            if why:
                return why
        taken.add(pressed.request)
        why = take(api, pressed, run, hints=hints, moved=moved, **press)
        if why:
            return why
        last = monotonic()
        if _rss() > MAX_RSS:
            return f"peak memory {_rss() >> 20} MB"


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
    lane, api, sides, why, logged = abspath(lane), None, (), "", False
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
        sides = tuple(_Api(environ.get("GITHUB_API_URL", "https://api.github.com"), token) for _ in range(2))
        why = served_elsewhere(api, repository, named, request) or bot_moved(api, bot_repo, bot)
        if why:
            return why
        _say(f"up for {clean(actor, 40)}'s next presses ({LANE_IDLE} s idle at most)")
        why = listen(api, int(me), actor, bake=bake, lane=lane, clone=clone, repository=repository, head=head, kept=kept,
                     bot_repo=bot_repo, bot=bot, until=until, limit=limit, sides=sides)
        return why
    except Exception as error:                  # noqa: BLE001 - the lane only ends
        why = f"error: {type(error).__name__}: {clean(error, 300)}"
        return why
    finally:
        for each in (api, *sides):
            if each is not None:
                each.close()
        if logged:
            _say(f"ended: {why}")
        with open(lane + LANE_END, "w", encoding="utf-8") as fh:
            fh.write(why + "\n")


# ========================
# Module Exports
# ========================
__all__ = [
    'Press',
    'Warm',
    'bot_moved',
    'fits',
    'follow',
    'hints_of',
    'listen',
    'main_of',
    'press_blob',
    'press_of',
    'read_press',
    'rung',
    'serve',
    'served_elsewhere',
    'take',
    'warm',
]
