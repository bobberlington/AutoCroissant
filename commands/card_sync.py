"""
The bot after the cutover (global_config.CARD_SYNC; docs/Campaigns/DATA_BUILD_ACTION.md sections 11 and 14, step 8).
The Card sync Action in the card repo builds the card data and commits it to the data branch; the bot only reads it:
- sync: card_data/ brought to the data branch (card_db.sync: one conditional request, free when nothing moved), the
  database reloaded from it, then the card list, query tables, type emojis and image links refreshed;
- /update_stats: that sync and one reply; with force_update or cards (admins), a Card sync run dispatched with
  ACTIONS_TOKEN and followed until it ends, then the sync;
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
                           ICON_MANIFEST_PATH, TYPES_DIR)
from commands import card_db
from commands.github_api import HEADERS, api_url, get_with_retry, session
from commands.psd_analyzer import card_name, stats_db                # the facade's (architecture invariant: one stats_db)
from commands.utils import queue_command, queue_edit, say

# ========================
# Configuration
# ========================
ACTIONS_TOKEN: str = getattr(config, "ACTIONS_TOKEN", "")   # fine-grained, the card repo only, Actions read and write
WORKFLOW = "card-sync.yml"                           # the card repo's Card sync Action
SITE_WORKFLOW = "site.yml"                           # the card repo's website build (it reads the bot's pickles)
RUN_NAME = "Card sync"                               # its live runs' name (a dry run is "Card sync (dry run)")
SCHEDULE = "every 30 minutes, at :04 and :34"        # its cron (card-sync.yml)
POLL_SECONDS = 15                                    # between two looks at a dispatched run
WAIT_MINUTES = 30                                    # a dispatched run is followed this long
FIND_SECONDS = 60                                    # a dispatched run shows up in the run list within this
EDIT_MINUTES = 14                                    # progress edits stop here: Discord's interaction lasts 15 minutes
WAITING = ("queued", "pending", "waiting", "requested")   # a run not started yet
FAILED = ("failure", "timed_out", "startup_failure")
BOT_LOGIN = "github-actions[bot]"                    # Card sync's own retry dispatches
CARD_PREFIXES = (f"{TYPES_DIR}/", "MDW/Fonts/")
CARD_FILES = {ICON_MANIFEST_PATH, *AUTHORS_PATHS, *EDITS_PATHS, *CREATED_PATHS}

# ========================
# Global State
# ========================
_loaded: dict = {"commit": None}                     # the data commit stats_db holds (None: the root pickles or nothing)
_sync_lock = Lock()                                  # one sync at a time (a reminder beside a command)
_run_lock = Lock()                                   # one dispatched run followed at a time


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
    if counts is None:                               # no synced data held before (startup on the pickles)
        return f"Card data loaded: {built}."
    what = ", ".join(f"{counts[k]} {k}" for k in ("changed", "new", "archived") if counts[k]) or "no card changed"
    return f"Card data {built}: {what}; reloaded."


def card_input(path: str) -> bool:
    """A card repo path the card data is built from (Card sync's gate, without the TTS save's own files)."""
    return path.lower().endswith((".psd", ".png")) or path.startswith(CARD_PREFIXES) or path in CARD_FILES


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


def notes(manifest: dict, repository: str = CARD_REPO) -> list:
    """The lines after the first: a re-parse going on, unreadable PSDs, card changes not built yet, a failed last run."""
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
        runs = [r for r in _runs(repository) if r.get("display_title") == RUN_NAME]
        ahead = _main_ahead(repository, manifest.get("card_commit"))
    except (RequestException, ValueError) as e:      # advice only: the sync itself worked
        print(f"Card sync status not read: {e}")
        return lines
    if ahead:
        going = next((r for r in runs if r.get("status") != "completed"), None)
        lines.append("The card repo has card changes since: " + (f"the card sync running now picks them up (<{going.get('html_url')}>)."
                                                                 if going else f"the next card sync ({SCHEDULE}) picks them up."))
    last = next((r for r in runs if r.get("status") == "completed" and r.get("conclusion") not in ("cancelled", "skipped")), None)
    if last is not None and last.get("conclusion") in FAILED:
        lines.append(f"The last card sync failed: <{last.get('html_url')}>; the bot keeps the last good card data.")
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


def sync(interaction=None, output_problematic: bool = True, quiet: bool = False, follow_ups: bool = True) -> list:
    """
    card_data/ brought to the data branch, the database reloaded when the data moved, and one reply (quiet: the
    console only, unless the sync failed); then the card list, query tables, type emojis and image links refreshed.

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
            got = card_db.sync(CARD_DATA_DIR, CARD_REPO, DATA_BRANCH)
            reloaded = got["commit"] != before_commit
            if reloaded:
                stats, old, _, _ = card_db.read_db(CARD_DATA_DIR, details=False)
                _swap(stats, old, got["commit"])
        except LookupError as e:
            say(interaction, f"No card data yet: {e}. The Card sync Action creates it: <{actions_link()}>")
            return []
        except (RequestException, OSError, ValueError, TypeError) as e:
            print(f"Card data sync failed:\n{format_exc()}")
            say(interaction, f"Card data sync failed ({type(e).__name__}: {e}); the bot keeps the card data it has.")
            return []
    manifest = got["manifest"]
    text = report(manifest, changes(before, stats_db.stats) if reloaded and before_commit is not None else None, reloaded)
    if quiet:
        print(text)
    else:
        say(interaction, "\n".join([text] + notes(manifest)))
    problems = _problems(manifest) if reloaded and output_problematic else []
    if problems:
        say(interaction, "Problem cards in this build:\n" + "".join(problems))
    if follow_ups:
        from commands.repo_history import refresh as refresh_history
        queue_command(refresh_history)               # re-pin image links; a git fetch when nothing changed
        queue_command(_refresh_cards, None if quiet else interaction)
    return problems


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
    on_ready's database after the cutover: the local copy first (no network), then a sync (on_ready queues the
    refreshes itself). With neither, the root pickles, frozen at the cutover (step 10 deletes them).
    """
    had = load_local()
    sync(None, output_problematic=False, quiet=True, follow_ups=False)
    if not had and _loaded["commit"] is None:
        print("No card data synced; loading the root pickles (frozen at the cutover)")
        stats_db.load()


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


def dispatch_site(repository: str = CARD_REPO) -> str:
    """
    A website build (site.yml) after a /push of the pickles it reads: GitHub runs its schedule hours late. Never retried
    or raised; without ACTIONS_TOKEN the schedule stays.

    Returns:
        One console line
    """
    if not ACTIONS_TOKEN:
        return "Website build: no ACTIONS_TOKEN, so the site picks the push up on its schedule."
    try:
        reply = session().post(api_url(repository, f"actions/workflows/{SITE_WORKFLOW}/dispatches"), json={"ref": CARD_BRANCH},
                               headers=_headers(), timeout=30)
    except Exception as e:                           # noqa: BLE001 - never fails the push
        return f"Website build not started ({type(e).__name__}: {e}); the site picks the push up on its schedule."
    if reply.status_code >= 400:
        return f"Website build not started (HTTP {reply.status_code}); the site picks the push up on its schedule."
    return "Website build started."


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


def _state(repository: str, run: dict) -> str:
    """A run's state for the progress edit: waiting, or the step running now."""
    if run.get("status") in WAITING:
        return "waiting for its turn"
    try:
        jobs = _json(api_url(repository, f"actions/runs/{run.get('id')}/jobs"), _headers()).get("jobs") or []
    except (RequestException, ValueError):
        return "running"
    steps = [s for job in jobs for s in job.get("steps") or []]
    step = next((s for s in steps if s.get("status") == "in_progress"), None)
    return f"running: {step.get('name')}" if step else "running"


def follow(interaction, run: dict, started: float) -> Optional[dict]:
    """
    Look at the run every POLL_SECONDS until it ends; its state edits the command's reply while Discord still takes
    edits (EDIT_MINUTES), never a reminder's (it has no reply to edit: each would be a new message).

    Returns:
        The ended run ("ran": seen started), or None after WAIT_MINUTES
    """
    shown, ran = None, False
    if interaction is not None and _by_reminder(interaction):
        interaction = None
    while run.get("status") != "completed":
        ran = ran or run.get("status") not in WAITING
        if monotonic() - started >= WAIT_MINUTES * 60:
            return None
        state = _state(CARD_REPO, run)
        if state != shown and interaction is not None and monotonic() - started < EDIT_MINUTES * 60:
            queue_edit(interaction, content=f"Card sync run <{run.get('html_url')}>: {state}")
            shown = state
        sleep(POLL_SECONDS)
        try:
            run = _json(api_url(CARD_REPO, f"actions/runs/{run.get('id')}"), _headers())
        except (RequestException, ValueError) as e:  # looked at again next time
            print(f"Card sync run not read: {e}")
    return {**run, "ran": ran}


def _again(run: dict) -> tuple:
    """
    A cancelled run: the concurrency group keeps one run pending, so a newer one replaced it before it started. Run it
    again when that cancels nothing but a scheduled run (ours does its work too), else say who replaced it. One seen
    running, or with nothing newer, was cancelled by hand.

    Returns:
        (dispatch again, why not)
    """
    if run.get("ran"):
        return (False, f"Card sync run <{run.get('html_url')}> was cancelled.")
    newer = [r for r in _runs(CARD_REPO, _headers()) if (r.get("id") or 0) > (run.get("id") or 0)]
    if not newer:
        return (False, f"Card sync run <{run.get('html_url')}> was cancelled.")
    pending = [r for r in newer if r.get("status") in WAITING and r.get("event") != "schedule"]
    if pending:
        return (False, f"Card sync run <{run.get('html_url')}> was replaced by a newer request before it started "
                       f"(<{pending[0].get('html_url')}>), which does the usual sync but not this re-parse; ask again "
                       "once it is done.")
    return (True, "")


def _retry_of(run: dict) -> Optional[dict]:
    """Card sync's own retry of a run (main moved during it, so its data was dropped), or None."""
    try:
        runs = _runs(CARD_REPO, _headers())
    except (RequestException, ValueError):
        return None
    return next((r for r in reversed(runs) if (r.get("id") or 0) > (run.get("id") or 0)
                 and r.get("event") == "workflow_dispatch" and (r.get("actor") or {}).get("login") == BOT_LOGIN), None)


def request_run(interaction, full: bool, names: list) -> Optional[dict]:
    """
    /update_stats force_update or cards: a Card sync run dispatched (full, or these cards) and followed until it ends
    (WAIT_MINUTES at most). A run cancelled before it started (replaced by a newer pending one) is dispatched once
    more, or said (_again); Card sync's own retry (main moved during the run) is followed too. Without ACTIONS_TOKEN,
    the link and the gh command.

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
            say(interaction, "The run was replaced by a scheduled card sync before it started; starting it again.")
        retry = _retry_of(run) if run.get("conclusion") == "success" else None
        if retry is not None:
            say(interaction, f"The card repo changed during run <{run.get('html_url')}>, so Card sync dropped its data and "
                             f"started again: <{retry.get('html_url')}>"
                             + (" (without the full re-parse: ask again once it is done)." if full else "."))
            run = follow(interaction, retry, started)
            if run is None:
                say(interaction, f"The Card sync run is still going after {WAIT_MINUTES} minutes; /update_stats "
                                 "loads its data once it lands.")
                return None
        if run.get("conclusion") == "success":
            say(interaction, f"Card sync run finished: <{run.get('html_url')}>")
        else:
            say(interaction, f"Card sync run ended ({run.get('conclusion')}): <{run.get('html_url')}>; the bot keeps "
                             "the last good card data.")
        return run
    finally:
        _run_lock.release()


# ========================
# Public API Functions
# ========================
def update(interaction=None, output_problematic: bool = True, force_update: bool = False, verbose: bool = True,
           cards: str = "") -> list:
    """
    /update_stats after the cutover: with force_update or cards, a Card sync run first (request_run); then the sync.
    A reminder's plain sync is quiet (the console), unless it fails.

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
    """/fonts after the cutover: the card data's psd_details.json (no PSD or font downloaded)."""
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
    'WORKFLOW',
    'actions_link',
    'built_line',
    'changes',
    'dispatch',
    'fonts_text',
    'follow',
    'gh_command',
    'load_local',
    'notes',
    'report',
    'request_run',
    'startup',
    'sync',
    'update',
]
