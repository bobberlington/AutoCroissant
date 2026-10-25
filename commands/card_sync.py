"""
The bot's card data (docs/Campaigns/DATA_BUILD_ACTION.md sections 11 and 14).
The Card sync Action in the card repo builds the card data and commits it to the data branch; the bot only reads it:
- sync: card_data/ brought to the data branch (card_db.sync: one conditional request, free when nothing moved), the
  database reloaded from it, then the card list, query tables, type emojis and image links refreshed;
- on use: a card command or autocomplete queues that sync (quiet) when none ran for REFRESH_SECONDS; its reply uses
  the data loaded now, the next one the new data;
- /update_stats: that sync and one reply; Card sync starts itself (a card push: the data at once, the bake and the
  TTS save after a 30-minute timer in card-sync.yml); with force_update or cards (admins), a run dispatched
  (ACTIONS_TOKEN) and its Card data job followed until it ends (not the bake), then the sync;
- /fonts: from the data's psd_details.json, whose fonts the build marked supplied or not.
Uncompiled (not in setup.py), like card_data.
"""
from datetime import datetime, timezone
from os.path import isfile, join
from threading import Lock
from time import monotonic, sleep, time
from traceback import format_exc
from typing import Optional

from requests.exceptions import RequestException

try:
    import config
except ImportError:                                  # tests and scripts without config.py
    config = None
from global_config import (AUTHORS_PATHS, CARD_BRANCH, CARD_DATA_DIR, CARD_REPO, CREATED_PATHS, DATA_BRANCH, EDITS_PATHS,
                           ICON_MANIFEST_PATH, RULES_PATH, TYPES_DIR)
from commands import card_db
from commands.github_api import HEADERS, api_url, get_with_retry, session
from commands.psd_analyzer import card_name, stats_db                # the facade's (architecture invariant: one stats_db)
from commands.utils import queue_command, queue_edit, say

# ========================
# Configuration
# ========================
ACTIONS_TOKEN: str = getattr(config, "ACTIONS_TOKEN", "")   # fine-grained, the card repo only, Actions read and write
WORKFLOW = "card-sync.yml"                           # the card repo's Card sync Action
RUN_NAME = "Card sync"                               # its live runs' name (a dry run is "Card sync (dry run)")
RETRY_NAME = "Card sync (retry)"                     # its own retry's name (main moved during a bake: main's lane only)
DATA_JOB = "Card data"                               # a run's data lane (after the bake: "Card data after the bake")
MAIN_JOB = "Bake and save"                           # a run's main lane
POLL_SECONDS = 15                                    # between two looks at a dispatched run
WAIT_MINUTES = 30                                    # a dispatched run is followed this long
FIND_SECONDS = 60                                    # a dispatched run shows up in the run list within this
LANE_RUNS = 5                                        # notes: runs whose jobs are read for each lane's last outcome
EDIT_MINUTES = 14                                    # progress edits stop here: Discord's interaction lasts 15 minutes
WAITING = ("queued", "pending", "waiting", "requested")   # a run not started yet
FAILED = ("failure", "timed_out", "startup_failure")
BOT_LOGIN = "github-actions[bot]"                    # GITHUB_TOKEN dispatches (Card sync's retry, card-guards)
FONTS_PREFIX = "MDW/Fonts/"                           # its top-level files only (the gate's MDW/Fonts/*)
CARD_FILES = {ICON_MANIFEST_PATH, RULES_PATH, *AUTHORS_PATHS, *EDITS_PATHS, *CREATED_PATHS}
REFRESH_SECONDS = 120                                # on use: the data branch looked at most this often
READERS = frozenset({                                # the slash commands that read card data (on_use); autocomplete too
    "export_cards", "export_rulebook", "fonts", "frankenstein", "list_orphans", "query", "query_ability",
    "query_rulebook", "view_old_metadata"})

# ========================
# Global State
# ========================
_loaded: dict = {"commit": None}                     # the data commit stats_db holds (None: nothing, or card_data/ unsynced)
_sync_lock = Lock()                                  # one sync at a time (a reminder beside a command)
_run_lock = Lock()                                   # one dispatched run followed at a time
_checked: dict = {"at": monotonic()}                 # the last look at the data branch (the start counts: startup's)
_unreported: dict = {}                               # checks' reloads for the next sync's reply: before, commit, problems
_use_lock = Lock()                                   # on_use's claim of a look (never waited on)


class TokenRefused(Exception):
    """GitHub refused ACTIONS_TOKEN (401, 403, 404)."""


class RateLimited(ValueError):
    """GitHub's rate limit (a 403 or 429 saying so)."""


class NoAnswer(Exception):
    """The dispatch got no reply: the run may have started."""


# ========================
# Helper Functions
# ========================
def actions_link(repository: str = CARD_REPO) -> str:
    """The Card sync Action's page."""
    return f"https://github.com/{repository}/actions/workflows/{WORKFLOW}"


def gh_command(full: bool, names: list, repository: str = CARD_REPO) -> str:
    """The same run started by hand."""
    what = "-f full=true" if full else f'-f cards="{", ".join(names)}"'
    return f"gh workflow run {WORKFLOW} -R {repository} {what} -f dry_run=false"


def _age(stamp: Optional[str], now: float) -> str:
    """'12 min ago' from the manifest's built_at."""
    try:
        at = datetime.strptime(stamp or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return "at an unknown time"
    minutes = int(max(0.0, now - at) // 60)
    if minutes < 1:
        return "just now"
    if minutes < 120:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // (24 * 60)} days ago"


def built_line(manifest: dict, now: Optional[float] = None) -> str:
    """'built 12 min ago from TTSCardMaker@1a2b3c4d (run: <url>)'."""
    repo = (manifest.get("card_repo") or CARD_REPO).rsplit("/", 1)[-1]
    run = manifest.get("run")
    return (f"built {_age(manifest.get('built_at'), time() if now is None else now)} from "
            f"{repo}@{(manifest.get('card_commit') or '?')[:8]}" + (f" (run: <{run}>)" if run else ""))


def changes(before: dict, after: dict) -> dict:
    """{"changed", "new", "archived"} cards between two databases, duplicates left out (they follow their original)."""
    def originals(cards: dict) -> dict:
        return {name: c for name, c in cards.items() if not getattr(c, "duplicate_of", None)}
    a, b = originals(before), originals(after)
    return {"changed": sum(1 for name in a.keys() & b.keys() if card_db.record(a[name]) != card_db.record(b[name])),
            "new": len(b.keys() - a.keys()), "archived": len(a.keys() - b.keys())}


def report(manifest: dict, counts: Optional[dict], reloaded: bool, now: Optional[float] = None) -> str:
    """/update_stats' first line: what the data is and what this sync did."""
    built = built_line(manifest, now)
    if not reloaded:
        return f"Card data is already current: {built}."
    if counts is None:                               # no synced data held before (startup)
        return f"Card data loaded: {built}."
    what = ", ".join(f"{counts[k]} {k}" for k in ("changed", "new", "archived") if counts[k]) or "no card changed"
    return f"Card data {built}: {what}; reloaded."


def card_input(path: str) -> bool:
    """A card repo path Card sync's gate goes for (the hand-edited save aside): card-sync.yml's push paths."""
    return (path.lower().endswith((".psd", ".png")) or path.startswith(f"{TYPES_DIR}/") or path in CARD_FILES
            or (path.startswith(FONTS_PREFIX) and "/" not in path[len(FONTS_PREFIX):]))


def _json(url: str, headers: dict):
    """GET url's JSON (get_with_retry); raises on an error status."""
    reply = get_with_retry(url, headers=headers)
    reply.raise_for_status()
    return reply.json()


def _runs(repository: str, headers: Optional[dict] = None) -> list:
    """Card sync's newest runs, newest first."""
    data = _json(api_url(repository, f"actions/workflows/{WORKFLOW}/runs?per_page=10"), HEADERS if headers is None else headers)
    return list(data.get("workflow_runs") or [])


def _main_ahead(repository: str, base: Optional[str]) -> bool:
    """Whether the card repo's branch has card input commits after the data's card_commit (one compare request)."""
    if not base:
        return False
    data = _json(api_url(repository, f"compare/{base}...{CARD_BRANCH}"), HEADERS)
    return bool(data.get("ahead_by")) and any(card_input(f.get("filename", "")) for f in data.get("files") or [])


def _lane(job: dict) -> Optional[str]:
    """A job's lane: "data" (Card data, after the bake too), "main" (Bake and save), or None."""
    name = job.get("name") or ""
    return "main" if name == MAIN_JOB else "data" if name.startswith(DATA_JOB) else None


def _lanes(repository: str, runs: list) -> dict:
    """
    Each lane's last outcome: {"data" or "main": (failed, run)} from the newest run where that lane's job ended (one
    jobs request per run, up to LANE_RUNS; runs that skipped a lane pass over it). A run without the lanes' jobs (before
    the split) counts for both by its conclusion.
    """
    found: dict = {}
    for run in runs[:LANE_RUNS]:
        jobs = _json(api_url(repository, f"actions/runs/{run.get('id')}/jobs"), HEADERS).get("jobs") or []
        if not any(_lane(j) for j in jobs):
            jobs = [{**run, "name": name} for name in (DATA_JOB, MAIN_JOB)]
        ended = [j for j in jobs if j.get("status") == "completed" and j.get("conclusion") not in ("skipped", "cancelled", None)]
        for lane in ("data", "main"):
            done = [j for j in ended if _lane(j) == lane]
            if done and lane not in found:
                found[lane] = (any(j.get("conclusion") in FAILED for j in done), run)
        if len(found) == 2:
            break
    return found


def notes(manifest: dict, repository: str = CARD_REPO) -> list:
    """The lines after the first: a re-parse going on, unreadable PSDs, card changes not built yet, a lane that failed."""
    lines = []
    pending = manifest.get("pending") or {}
    stale = pending.get("stale") or []
    if stale:
        lines.append(f"Re-parse in progress: {len(stale)} card{'' if len(stale) == 1 else 's'} still to re-parse; "
                     "the next card syncs continue it.")
    failed = pending.get("failed") or []
    if failed:
        lines.append(f"Could not read {len(failed)} PSD{'' if len(failed) == 1 else 's'} ("
                     + ", ".join(card_name(f.get("path", "")) for f in failed[:10]) + (", ..." if len(failed) > 10 else "")
                     + "); " + ("its card keeps its" if len(failed) == 1 else "their cards keep their")
                     + " last version until the PSD is fixed.")
    try:
        runs = [r for r in _runs(repository) if r.get("display_title") in (RUN_NAME, RETRY_NAME)]
        ahead = _main_ahead(repository, manifest.get("card_commit"))
    except (RequestException, ValueError) as e:      # advice only: the sync itself worked
        print(f"Card sync status not read: {e}")
        return lines
    if ahead:
        going = next((r for r in runs if r.get("status") != "completed"), None)
        lines.append("The card repo has card changes not in the data yet: "
                     + (f"the card sync under way picks them up (<{going.get('html_url')}>)." if going else
                        "the next card sync picks them up within minutes of the push (website edits are baked into the PSDs "
                        "30 minutes after the first card push of a burst)."))
    try:
        lanes = _lanes(repository, runs)
    except (RequestException, ValueError) as e:
        print(f"Card sync jobs not read: {e}")
        return lines
    failed, run = lanes.get("data", (False, None))
    if failed:
        lines.append(f"The last card data build failed: <{run.get('html_url')}>; the bot keeps the last good card data.")
    failed, run = lanes.get("main", (False, None))
    if failed:
        lines.append(f"The last bake and save failed: <{run.get('html_url')}>; website edits wait for the next one.")
    return lines


def _problems(manifest: dict) -> list:
    """The build's validator notes as /update_stats always showed them (path, then the notes in a code block)."""
    return [f"{p.get('path')}\n```\n" + "\n".join(p.get("notes") or []) + "\n```" for p in manifest.get("problems") or []]


def _by_reminder(interaction) -> bool:
    """A reminder's run (its interaction acts as the reminder's creator): a plain sync stays quiet then."""
    from commands.analytics import _ReminderUser     # lazy: analytics is the reminders' module
    return isinstance(getattr(interaction, "user", None), _ReminderUser)


# ========================
# Loading and syncing
# ========================
def _swap(stats: dict, old_stats, commit: Optional[str]) -> None:
    """The in-memory database replaced by synced data (readers see the old or the new dict, never half)."""
    stats_db.stats, stats_db.old_stats = stats, old_stats
    _loaded["commit"] = commit


def load_local(folder: str = CARD_DATA_DIR) -> bool:
    """
    stats_db from the local copy (no network). False when it holds no readable card data. A copy that doesn't match
    its manifest (a sync cut off midway) is loaded with no commit, so the next sync reloads it whole.
    """
    if not isfile(join(folder, card_db.STATS_JSON)):
        return False
    try:
        stats, old, _, _ = card_db.read_db(folder, details=False)
    except (OSError, ValueError, TypeError) as e:
        print(f"{folder}: not loaded ({e}); the sync fetches it again")
        return False
    problems = card_db.check_db(folder, parse=False)    # parsed just above; the details file by its hash
    if problems:
        print(f"{folder}: a sync was cut off ({'; '.join(problems)}); loaded until the next sync completes it")
    _swap(stats, old, None if problems else card_db.read_json(join(folder, card_db.SYNC_JSON)).get("commit"))
    print(f"Card data loaded from {folder}: {len(stats)} cards")
    return True


def _pull() -> tuple:
    """
    One look at the data branch (card_db.sync), the database swapped when it moved. Hold _sync_lock.

    Returns:
        (the manifest, reloaded)
    """
    _checked["at"] = monotonic()
    got = card_db.sync(CARD_DATA_DIR, CARD_REPO, DATA_BRANCH)
    reloaded = got["commit"] != _loaded["commit"]
    if reloaded:
        stats, old, _, _ = card_db.read_db(CARD_DATA_DIR, details=False)
        _swap(stats, old, got["commit"])
    return (got["manifest"], reloaded)


def _follow_ups(interaction) -> None:
    """After a sync: image links re-pinned, then the card list, query tables and type emojis (_refresh_cards)."""
    from commands.repo_history import refresh as refresh_history
    queue_command(refresh_history)                   # re-pin image links; a git fetch when nothing changed
    queue_command(_refresh_cards, interaction)


def sync(interaction=None, output_problematic: bool = True, quiet: bool = False, follow_ups: bool = True) -> list:
    """
    card_data/ brought to the data branch, the database reloaded when the data moved, and one reply (quiet: the
    console only, unless the sync failed or there are problem cards), counting checks' reloads since the last; then the card list, query tables, type emojis and image links refreshed.

    Args:
        interaction: Where the reply goes (None: the console)
        output_problematic: Show the new build's problem cards
        quiet: No reply when it worked (a reminder's sync)
        follow_ups: Queue the refreshes (startup runs its own)

    Returns:
        The problem card messages of newly loaded data
    """
    with _sync_lock:
        before, before_commit = stats_db.stats, _loaded["commit"]
        try:
            manifest, reloaded = _pull()
            pending = dict(_unreported)              # checks' reloads since the last reply: this sync reports them
            _unreported.clear()
        except LookupError as e:
            say(interaction, f"No card data yet: {e}. The Card sync Action creates it: <{actions_link()}>")
            return []
        except (RequestException, OSError, ValueError, TypeError) as e:
            print(f"Card data sync failed:\n{format_exc()}")
            say(interaction, f"Card data sync failed ({type(e).__name__}: {e}); the bot keeps the card data it has.")
            return []
    if pending:
        before, before_commit = pending["before"], pending["commit"]
    text = report(manifest, changes(before, stats_db.stats) if (reloaded or pending) and before_commit is not None
                  else None, reloaded or bool(pending))
    if quiet:
        print(text)
    else:
        say(interaction, "\n".join([text] + notes(manifest)))
    problems = (pending.get("problems", []) + (_problems(manifest) if reloaded else [])) if output_problematic else []
    if problems:
        say(interaction, "Problem cards in this build:\n" + "".join(problems))
    if follow_ups:
        _follow_ups(None if quiet else interaction)
    return problems


def on_use(command: Optional[str] = None) -> bool:
    """
    A card command (or an autocomplete: command None) is running: a look at the data branch is queued when none ran
    for REFRESH_SECONDS. On the event loop: a clock read and a queue append, never a wait. The reply uses the data
    loaded now; new data serves the next one (/query and autocomplete answer within Discord's 3 seconds, and a reload's
    card list and query tables are rebuilt after it).

    Args:
        command: The slash command's name; one not in READERS is ignored

    Returns:
        True when a look was queued
    """
    if command is not None and command not in READERS:
        return False
    if not _use_lock.acquire(blocking=False):        # a reminder's thread claiming at this instant
        return False
    try:
        now, last = monotonic(), _checked["at"]
        if last is not None and now - last < REFRESH_SECONDS:
            return False
        _checked["at"] = now
    finally:
        _use_lock.release()
    queue_command(check)
    return True


def check() -> bool:
    """
    on_use's look, in the command queue's thread: the sync, quiet (a reload's report to the console), then the
    refreshes when the data moved, else the website's card ids alone (a conditional request: a new card's page link
    once the site has it). A reload's counts and problem cards wait for the next sync's reply (_unreported). Skipped
    while a sync runs (it is this look) or a dispatched run is followed (its sync loads it); a failure is logged and
    the data kept.

    Returns:
        True when new data was loaded
    """
    if _run_lock.locked() or not _sync_lock.acquire(blocking=False):
        return False
    try:
        before, before_commit = stats_db.stats, _loaded["commit"]
        manifest, reloaded = _pull()
        if reloaded:
            if not _unreported:
                _unreported.update(before=before, commit=before_commit, problems=[])
            _unreported["problems"] += _problems(manifest)
    except Exception as e:                           # noqa: BLE001 - a failed look never fails a command
        print(f"Card data check failed ({type(e).__name__}: {e}); keeping the loaded data")
        return False
    finally:
        _sync_lock.release()
    if not reloaded:
        from commands.query_card import card_repo
        card_repo._populate_site_ids()
        return False
    print(report(manifest, changes(before, stats_db.stats) if before_commit is not None else None, True))
    _follow_ups(None)
    return True


def _refresh_cards(interaction) -> None:
    """After a sync: the card list (a 304 when the tree is unchanged), the query tables, then the type emojis over it."""
    from commands.query_card import DEFAULT_REPOSITORY, card_repo
    from commands.type_emojis import sync_changed_emojis
    card_repo.populate_files()
    card_repo.prep_dataframes()
    tree = card_repo._tree                           # emojis follow the card repo, never a /set_repo one
    if tree is not None and not tree.get("truncated") and card_repo.repository == DEFAULT_REPOSITORY:
        shas = {item.get("path", ""): item.get("sha") for item in tree.get("tree", [])}
        queue_command(sync_changed_emojis, interaction, card_repo.repository, shas)


def startup() -> None:
    """
    on_ready's database: the local copy first (no network), then a sync (on_ready queues the refreshes itself).
    With neither (GitHub down at the first boot), no cards until a sync works.
    """
    load_local()
    sync(None, output_problematic=False, quiet=True, follow_ups=False)


# ========================
# Card sync runs (force_update, cards)
# ========================
def _headers() -> dict:
    return {"Authorization": f"Bearer {ACTIONS_TOKEN}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _refusal(reply) -> None:
    """Raise RateLimited or TokenRefused for an Actions reply that is one."""
    headers = {k.lower(): v for k, v in (reply.headers or {}).items()}
    if reply.status_code in (403, 429) and (headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
                                            or "rate limit" in reply.text.lower()):
        raise RateLimited(f"GitHub's rate limit, HTTP {reply.status_code}")
    if reply.status_code in (401, 403, 404):
        raise TokenRefused(f"HTTP {reply.status_code}")


def _post_dispatch(repository: str, body: dict):
    return session().post(api_url(repository, f"actions/workflows/{WORKFLOW}/dispatches"), json=body,
                          headers=_headers(), timeout=30)


def dispatch(repository: str, inputs: dict) -> Optional[dict]:
    """
    Start a Card sync run (never retried: a retry could start two). The run comes from the reply when GitHub returns
    it (return_run_details), else it is the newest dispatched run not listed before, looked for FIND_SECONDS.

    Returns:
        The run, or None when it did not show up

    Raises:
        TokenRefused: 401, 403 or 404; RateLimited, ValueError: another refusal; NoAnswer: the dispatch got no reply;
        requests' exceptions (the run list before it)
    """
    listed = get_with_retry(api_url(repository, f"actions/workflows/{WORKFLOW}/runs?per_page=10"), headers=_headers())
    _refusal(listed)
    listed.raise_for_status()
    seen = {r.get("id") for r in listed.json().get("workflow_runs") or []}
    body = {"ref": CARD_BRANCH, "inputs": inputs, "return_run_details": True}
    try:
        reply = _post_dispatch(repository, body)
        if reply.status_code == 422 and "return_run_details" in reply.text:   # GitHub without that option: not started
            reply = _post_dispatch(repository, {"ref": CARD_BRANCH, "inputs": inputs})
    except RequestException as e:
        raise NoAnswer(f"{type(e).__name__}: {e}") from e
    _refusal(reply)
    if reply.status_code >= 400:
        raise ValueError(f"HTTP {reply.status_code}: {reply.text[:200]}")
    run_id = None
    if reply.status_code == 200:
        try:
            run_id = reply.json().get("workflow_run_id")
        except ValueError:
            run_id = None
    if run_id:
        return _json(api_url(repository, f"actions/runs/{run_id}"), _headers())
    deadline = monotonic() + FIND_SECONDS
    while monotonic() < deadline:
        sleep(3)
        for run in _runs(repository, _headers()):
            if (run.get("id") not in seen and run.get("event") == "workflow_dispatch"
                    and (run.get("actor") or {}).get("login") != BOT_LOGIN):
                return run
    return None


def _data_job(run: dict) -> Optional[dict]:
    """The run's Card data job (its jobs list), or None: not listed yet, or a run without one."""
    jobs = _json(api_url(CARD_REPO, f"actions/runs/{run.get('id')}/jobs"), _headers()).get("jobs") or []
    return next((j for j in jobs if j.get("name") == DATA_JOB), None)


def _state(followed: dict) -> str:
    """The followed job's (or run's) state for the progress edit: waiting, or the step running now."""
    if followed.get("status") in WAITING:
        return "waiting for its turn"
    step = next((s for s in followed.get("steps") or [] if s.get("status") == "in_progress"), None)
    return f"running: {step.get('name')}" if step else "running"


def follow(interaction, run: dict, started: float) -> Optional[dict]:
    """
    Look at the run's Card data job every POLL_SECONDS until it ends (the data lands then; the run's Bake and save is
    not waited for); without that job, the run. Its state edits the command's reply while Discord still takes edits
    (EDIT_MINUTES), never a reminder's (it has no reply to edit: each would be a new message).

    Returns:
        The run with the followed job's status and conclusion ("ran": seen started), or None after WAIT_MINUTES
    """
    shown, ran = None, False
    if interaction is not None and _by_reminder(interaction):
        interaction = None
    while True:
        job = None
        try:
            job = _data_job(run)
            if job is None:
                run = _json(api_url(CARD_REPO, f"actions/runs/{run.get('id')}"), _headers())
        except (RequestException, ValueError) as e:  # looked at again next time
            print(f"Card sync run not read: {e}")
        followed = job or run
        ran = ran or bool((job or {}).get("steps"))  # a job cancelled before it started has no steps
        if followed.get("status") == "completed":
            return {**run, "status": "completed", "conclusion": followed.get("conclusion"), "ran": ran}
        ran = ran or followed.get("status") not in WAITING
        if monotonic() - started >= WAIT_MINUTES * 60:
            return None
        state = _state(followed)
        if state != shown and interaction is not None and monotonic() - started < EDIT_MINUTES * 60:
            queue_edit(interaction, content=f"Card sync run <{run.get('html_url')}>: {state}")
            shown = state
        sleep(POLL_SECONDS)


def _again(run: dict) -> tuple:
    """
    A cancelled Card data job: its concurrency group keeps one pending, so a newer run's (a push's, the schedule's)
    replaced it before it started. Run it again unless a newer dispatch is pending (running again would cancel that
    one's), then say so. One seen running, or with nothing newer, was cancelled by hand.

    Returns:
        (dispatch again, why not)
    """
    if run.get("ran"):
        return (False, f"Card sync run <{run.get('html_url')}> was cancelled.")
    newer = [r for r in _runs(CARD_REPO, _headers()) if (r.get("id") or 0) > (run.get("id") or 0)]
    if not newer:
        return (False, f"Card sync run <{run.get('html_url')}> was cancelled.")
    pending = [r for r in newer if r.get("status") != "completed" and r.get("event") == "workflow_dispatch"
               and r.get("display_title") != RETRY_NAME]
    if pending:
        return (False, f"Card sync run <{run.get('html_url')}> was replaced by a newer request before it started "
                       f"(<{pending[0].get('html_url')}>), which does the usual sync but not this re-parse; ask again "
                       "once it is done.")
    return (True, "")


def request_run(interaction, full: bool, names: list) -> Optional[dict]:
    """
    /update_stats force_update or cards: a Card sync run dispatched (full, or these cards) and its Card data job
    followed until it ends (WAIT_MINUTES at most; the run's bake is not waited for). A data job cancelled before it
    started (replaced by a newer pending one) is dispatched once more, or said (_again). Card sync's own retry (main
    moved during the bake) redoes only the bake and the save: not followed. Without ACTIONS_TOKEN, the link and the gh
    command.

    Returns:
        The last run, or None
    """
    what = "re-parsing every card" if full else f"re-parsing {', '.join(names)}"
    if not ACTIONS_TOKEN:
        say(interaction, f"Card data is built by the Card sync Action, and this bot has no ACTIONS_TOKEN in config.py to "
                         f"start a run {what}. Start it on <{actions_link()}> (Run workflow, dry_run off) or with "
                         f"`{gh_command(full, names)}`.")
        return None
    if not _run_lock.acquire(blocking=False):
        say(interaction, "A Card sync run started by /update_stats is already being followed; wait for its reply.")
        return None
    try:
        inputs = {"dry_run": "false", **({"full": "true"} if full else {"cards": ", ".join(names)})}
        started = monotonic()
        for attempt in range(2):
            try:
                run = dispatch(CARD_REPO, inputs)
            except TokenRefused as e:
                say(interaction, f"GitHub refused ACTIONS_TOKEN ({e}): it lacks Actions write on the card repo or "
                                 f"has expired. Start the run on <{actions_link()}> or with `{gh_command(full, names)}`.")
                return None
            except NoAnswer as e:
                say(interaction, f"GitHub did not answer the Card sync dispatch ({e}), so the run may have started: "
                                 f"look on <{actions_link()}> before asking again.")
                return None
            except (RequestException, ValueError) as e:
                say(interaction, f"Could not start a Card sync run ({type(e).__name__}: {e}); try again, or start "
                                 f"it on <{actions_link()}>.")
                return None
            if run is None:
                say(interaction, f"Card sync was started, but its run did not show up within a minute: <{actions_link()}>")
                return None
            say(interaction, f"Card sync run started ({what}): <{run.get('html_url')}>")
            run = follow(interaction, run, started)
            if run is None:
                say(interaction, f"The Card sync run is still going after {WAIT_MINUTES} minutes; /update_stats "
                                 "loads its data once it lands.")
                return None
            if run.get("conclusion") != "cancelled":
                break
            again, why = _again(run) if attempt == 0 else (False, f"Card sync run <{run.get('html_url')}> was "
                                                                  "cancelled again; ask again later.")
            if not again:
                say(interaction, why)
                return run
            say(interaction, "The run's card data was replaced by a newer card sync's before it started; starting it again.")
        if run.get("conclusion") == "success":
            say(interaction, f"Card sync run's card data finished: <{run.get('html_url')}>")
        else:
            say(interaction, f"Card sync run's card data ended ({run.get('conclusion')}): <{run.get('html_url')}>; the "
                             "bot keeps the last good card data.")
        return run
    finally:
        _run_lock.release()


# ========================
# Public API Functions
# ========================
def update(interaction=None, output_problematic: bool = True, force_update: bool = False, verbose: bool = True,
           cards: str = "") -> list:
    """
    /update_stats: with force_update or cards, a Card sync run first (request_run); then the sync
    (Card sync starts its other runs itself). A reminder's plain sync is quiet (the console), unless it fails.

    Returns:
        The problem card messages of newly loaded data
    """
    names = [c.strip() for c in (cards or "").split(",") if c.strip()]
    quiet = not verbose or _by_reminder(interaction)
    if force_update or names:
        request_run(interaction, force_update, names)
        quiet = not verbose
    return sync(interaction, output_problematic, quiet)


def fonts_text(folder: str = CARD_DATA_DIR) -> str:
    """/fonts: the card data's psd_details.json (no PSD or font downloaded)."""
    from commands import psd_details                 # lazy: fontTools
    from commands.font_check import details_report, discord_text
    if not isfile(join(folder, card_db.DETAILS_JSON)):
        return "No card data yet: run /update_stats."
    report_ = details_report(psd_details.load(join(folder, card_db.DETAILS_JSON)), stats_db.stats)
    manifest = card_db.read_json(join(folder, card_db.MANIFEST_JSON))
    return discord_text(report_, head=f"From the card data ({built_line(manifest)}): {report_.checked} card PSDs.")


# ========================
# Module Exports
# ========================
__all__ = [
    'ACTIONS_TOKEN',
    'READERS',
    'REFRESH_SECONDS',
    'WORKFLOW',
    'actions_link',
    'built_line',
    'changes',
    'check',
    'dispatch',
    'fonts_text',
    'follow',
    'gh_command',
    'load_local',
    'notes',
    'on_use',
    'report',
    'request_run',
    'startup',
    'sync',
    'update',
]
