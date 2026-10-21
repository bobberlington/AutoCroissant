---
name: autocroissant-run-and-operate
description: Day-to-day operations runbook for the AutoCroissant Discord bot. Load this when the task sounds like "run the bot", "start the bot", "stop", "restart", "deploy", "update the bot", "push/pull the pickles", "switch machines" / "handoff to the other machine", "sync commands" / "commands not showing up after deploy", "run update_stats", "local mode vs remote mode", "set a reminder" / "schedule a command" / "reminder didn't run", "sync emojis" / "/sync_emojis" / "type emojis not showing", "TTS save" / "Tabletop Simulator save" / "/update_tts_save" / "TTS/decks.json", or "where do files go" (exports, music downloads, temp files). Contains the start command and healthy startup output, on_ready init order, stop/restart (jetson systemd), the command-sync model, the ANATOMY of /push, /pull, /update (PICKLE commits, pickles-ours/code-theirs merge, force_reset, every failure message; live triage is autocroissant-debugging-playbook), the machine handoff runbook, the /update_stats operating rule (just run it), artifact locations, reminder, type-emoji and TTS-save operations, and a guild-admin table.
---

# AutoCroissant: Run and Operate

Operational runbook, verified against the code on 2026-07-11; sections 1, 5, 7, 8, 9 and 11 re-verified
2026-10-19 against the uncommitted working tree (HEAD a7a4bb3); sections 1, 3, 5, 7 and 8 re-verified
2026-10-20 (HEAD 9fb3093; sections 1, 3, 7, 8 again after that day's optimization pass). Audience: an operator (human or model)
with zero prior context. All commands assume cwd = repo root (`/Users/michaelsrouji/Desktop/AutoCroissant`
on this Mac). "Admin" means your Discord user ID is in the `ADMINS` list in `config.py`
(`lacks_admin`, commands/utils.py: True when you are NOT an admin).
On the dev Mac (2026-10-19) `config.TOKEN` logs in as the "Waxwell" bot identity; the AutoCroissant
token line is commented out. Which identity is logged in matters for type emojis (section 11).

Production topology (2026-10-20): the bot runs on the jetson as systemd `autocroissant.service`
(section 3); the Mac is for development. The three committed pickles
(`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are the database, and the bot's own git commands
(`/push`, `/update`) are how state moves between machines. Everything else is per-machine.
Section 6 is only for a (rare) host move.

## 1. Starting the bot

Prerequisites (see autocroissant-build-and-env for building the environment):

- [ ] cwd is the repo root. Every path in the bot is relative (`stats.pkl`, `music/`,
      `./startup.sh`). Starting from anywhere else makes the bot silently rebuild EMPTY databases —
      the tell is `stats.pkl is empty or doesn't exist, rebuilding...` in the output. Stop immediately
      if you see that on a machine that should have data.
- [ ] `config.py` exists in the repo root with a valid `TOKEN`. It is SECRET (never print or commit it).
      main.py:12 does `from config import TOKEN` — a missing/invalid config.py kills the process at import.
- [ ] No stale compiled artifacts: if `*.so` files exist from an old Cython build they shadow the `.py`
      files you just pulled. `python3 setup.py clean` first (none on the Mac as of 2026-10-19; the jetson
      rebuilds them on every deploy, section 3; details in autocroissant-build-and-env).

Start it:

```bash
python3 main.py
```

### What healthy startup output looks like (as of 2026-07-11; 2026-10-19 additions marked)

```
Git token found, API limited to 5000 requests/hour.    <- import time: main.py:89 print_token_status() (github_api.py); "No git token in config..." if GIT_TOKEN unset
We have logged in as AutoCroissant#....                <- on_ready (main.py:107-108)
Registered 50 slash commands for reminder system.      <- slash_registry populated (main.py:110-112); 50 since 2026-10-20 (/list_guild_members removed)
Bot initialization complete                            <- loops started (main.py:123-126)
Synced commands to guild: <name> (<id>)                <- one line per guild, async (main.py:102)
Trying to open reminder.pkl.                           <- init_reminder
reminder.pkl is empty or missing.                      <- or "Existing dict found in reminder.pkl, updating entries..." plus "Rescheduled reminder <id> from <t1> to <t2>" and (2026-10-19, first start after the rewrite) "Reminder <id>: migrated command `...` -> `...`" lines
Trying to open stats.pkl                               <- init_psd
Loaded existing stats from stats.pkl
Trying to open old_stats.pkl
Loaded existing old stats from old_stats.pkl
Loaded 765 cards and 48 rulebook entries               <- prep_dataframes (query_card.py:452); counts as of 2026-07-11
Stats database initialized.
Loading aliases from aliases.pkl                       <- init_query (1 GitHub tree API call follows)
Type emojis: 126 icons and crossed-out variants in MichaelJSr/TTSCardMaker, N uploaded, 126 ready.   <- init_type_emojis (2026-10-19); N = 0 once synced
```

Lines after "Bot initialization complete" come from queued async work and can interleave.
The token-presence line prints ONCE, from main.py (`print_token_status`; since the 2026-10-20 split the card modules print nothing at import, so scripts show no token line).

### on_ready init order (main.py:106-126)

`on_ready` populates `slash_registry` (command name -> callback; this is what lets reminders run
bot commands), spawns a per-guild command sync task, then QUEUES five init functions and finally
starts the three queue-drain loops (main.py:123-125). Queued work only executes once the loops
start draining — so the loops effectively start first, and nothing is sent to Discord before they run.

| Init (queued in this order) | What it does | Failure/no-op behavior |
|---|---|---|
| `init_reminder` (analytics.py:940) | Loads `reminder.pkl`, migrates old entries in place, rolls past-due REPEATING reminders forward to their next future slot in one step, saves atomically | Missing/empty file: "reminder.pkl is empty or missing." — fine. Unreadable file: "ERROR: reminder.pkl could not be read ... Reminders are OFF and the file was left untouched" — every reminder command then replies "✗ Reminders are off"; move the file aside and restart |
| `init_psd` (psd_analyzer.py:476) | Loads `stats.pkl` + `old_stats.pkl` into `stats_db`, builds the query DataFrames | Missing pickles: rebuilds empty (dangerous if unexpected — see cwd warning) |
| `init_query` (query_card.py:772) | Loads `aliases.pkl`, fetches the TTSCardMaker GitHub file tree (1 API call; 3 tries 1 s and 2 s apart on a connection error, 429 or 5xx, `github_api.get_with_retry`) | A failed or non-tree reply keeps the last good list: "Card list: GitHub tree fetch for <repo> failed (...); keeping the last good list (N entries)" (0 entries right after a start, so /query finds nothing until the next /update_stats or restart) and "Warning: Failed to populate files from repository (status N)" |
| `refresh_history` (repo_history.py:130) | Fetches the bare history clone (`~/Documents/TTSCardMaker-history.git`) and maps each card PNG to the commit that last changed it, for sha-pinned image links | Never raises; on failure the old map stays and links fall back to `main` |
| `init_type_emojis(client)` (type_emojis.py:413; added 2026-10-19) | Uploads any MISSING type emojis for the logged-in bot identity and fills the `[token]` → emoji map (section 11). Shares the card list's tree (since 2026-10-20 the second startup tree fetch is a 304, 0 bytes) + 1 Discord list call; uploads only on first start per identity | A failed GitHub listing falls back to the card list's last tree ("using the last good card tree"); any other failure prints "Emoji sync failed: <err> (using the N existing emojis)" and the existing emojis still render. Nothing else depends on it |

AI generation (and its `init_pipeline`) lives only on branch `archive/ai-generation`; main has none.

The three loops (main.py:839-876) each tick once per second: `process_command_queue` (also enqueues
`check_reminder` every tick — that is the reminder heartbeat), `process_dispatch_queue` (message/file
sends), `process_edit_queue` (edits). All bot output rides these queues, so replies have up to ~1 s of
latency by design. Depth on the queue model: autocroissant-architecture-contract.

## 2. Command anatomy (30-second version)

Every slash command in main.py follows one pattern: `func = to_thread(func)` then a
`@tree.command(...)` wrapper that calls it. Worker code never awaits Discord directly; it calls
`queue_message` / `queue_file` / `queue_edit` / `queue_command` (commands/utils.py:36-60) and the
loops deliver. If an interaction token has expired (Discord kills them after 15 minutes), delivery
falls back to a plain `channel.send` — so output from long jobs arrives as a normal channel message
instead of a reply. That is expected, not a bug. Admin gating is a `lacks_admin(interaction)` call at
the top of the wrapper. Full contract: autocroissant-architecture-contract.

## 3. Stopping and restarting

| Way | Gate | What happens |
|---|---|---|
| `/stop_bot` | admin | Stops the three loops, replies "Stopping bot!", `client.close()`, `exit(0)` (main.py:159-169) |
| `/restart_bot` | NONE (verified main.py:150-156 on 2026-10-20 — no lacks_admin; anyone in the guild can restart) | Stops the loops, replies "Restarting bot!", then `restart_bot()` |
| Ctrl-C in the terminal | local | discord.py shuts down; main.py:881-882 prints "Bot shutting down via keyboard interrupt". Any other fatal error prints "Fatal error: ..." and exits 1 |

`restart_bot()` (commands/update_bot.py:11-15) does `execv('./startup.sh', argv)`; on
`FileNotFoundError` it falls back to `execl(python, python, main.py ...)` — i.e. re-exec the same
interpreter on main.py. The process is REPLACED, so silence after "Restarting bot!" is normal.

**Production (the jetson, 2026-10-20):** systemd `autocroissant.service`; no startup.sh, so /restart_bot
re-execs main.py. Deploy: `git pull --ff-only`, `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` (compiled .so
load before .py; ~6 min), `sudo systemctl restart autocroissant`, check `journalctl -u autocroissant`.
The rebuild is NOT optional since the 2026-10-20 split: stale `.so` next to new `.py` fails at import
(`cannot import name perms_check` from analytics.so) and systemd restarts into the same error.
Hand-run scripts need `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`; push from the box with
`SSH_AUTH_SOCK=/run/autocroissant/ssh-agent.sock git push`. Memory: jetson-server.

**startup.sh is a deploy-machine artifact, NOT in the repo** (verified absent 2026-07-11; the repo has
no startup.sh and .gitignore would not exclude it — it simply was never committed). Operators may
create one per machine, e.g. a build-then-run script (`python3 setup.py build_ext --inplace &&
exec python3 main.py`), but its contents are per-machine and UNVERIFIED (the jetson has none). Two traps:

- Only `FileNotFoundError` triggers the fallback. A startup.sh that exists but is not executable
  raises `PermissionError`, which is NOT caught: the restart fails, the process stays alive with all
  three queue loops stopped (they were stopped before the exec), and `/restart_bot` will keep failing
  the same way. Fix: `chmod +x startup.sh` or delete it, then restart manually. `/stop_bot` still works.
- Because argv is passed through, startup.sh receives the original arguments; keep it argument-agnostic.

## 4. Command sync model

- On EVERY start, `sync_guild_commands` (main.py:98-102) runs per guild: `tree.copy_global_to(guild)`
  then `tree.sync(guild=guild)`. Guild-scoped syncs take effect essentially immediately, so after a
  restart the commands are current in every guild the bot is in (a stale Discord client may need a
  restart/reload to show them — client-side cache, not the bot).
- `/sync_global` (admin; main.py:670-675 -> management.py:75-97) clears the tree's global command set,
  syncs that (now-empty) global set, then re-copies and re-syncs each guild, finishing with
  "Commands synced globally and to N guilds." Use it to clean up duplicated/stale command entries
  (e.g. commands appearing twice because a global copy lingers). Discord platform note (not
  repo-verified): GLOBAL command changes can take up to ~1 hour to propagate; guild ones do not —
  which is exactly why this bot syncs per guild.
- If commands are missing/duplicated after a deploy: restart first (re-registers and re-syncs
  everything), then `/sync_global` if duplicates persist. Deeper triage: autocroissant-debugging-playbook.

## 5. The self-update flow (/push, /pull, /update)

All three are admin-gated. This is how the bot updates ITSELF from Discord. Remote: fetch over
https, push over `ssh://git@github.com/bobberlington/AutoCroissant.git` (verified `git remote -v`);
the gitignored `.ssh/` directory in the repo root holds the deploy key (never commit or print it).

### /push — commit, rebase, push the pickles (update_bot.py:18-44)

`git add aliases.pkl stats.pkl old_stats.pkl` -> `git commit -m "PICKLE"` (no-op if nothing changed) ->
`git pull --rebase --autostash origin main` (picks up code pushed since the last pull) -> `git push`.
Pickle snapshots are commits named PICKLE.
Success: "Successfully pushed!" (also pushes a PICKLE commit left by an earlier failed push).
Nothing ahead of origin/main after the rebase: "Pickles are already up to date.".
Rebase fails (conflict, network, auth): `git rebase --abort`, then "Push failed: <err>" and a re-raise; the
local PICKLE commit stays. If no rebase had started (fetch failed or timed out), the pull's error is shown.

GATE (owned by autocroissant-change-control, summarized here): pickle corruption is the #1
costliest failure class in this project's history — run the diff before every push:

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
```

Since 2026-10-19 a pushed PICKLE is also PUBLIC: the TTSCardMaker card website builds from the
pickles on AutoCroissant's `main` (every TTSCardMaker push, every 6 h, or on demand), so a bad push
shows up on the site within hours. Site mechanics: ttscardmaker-companion.

### /pull — fetch and merge (update_bot.py:47-115)

**/pull does NOT hard-reset anything** (its description and help text said so until 2026-10-20, C1;
both now say "fetches and merges"). Actual behavior:

1. "Fetching from remote..." — `git fetch`.
2. If local HEAD != origin/main and the histories diverged: "Branches have diverged. Attempting to
   resolve..." then `git pull origin main --no-edit`. Clean merge: "Merge successful!".
3. On CONFLICT: "Merge conflict detected. Resolving automatically..." — per conflicted file:
   pickles (`aliases.pkl`/`stats.pkl`/`old_stats.pkl`) get "Keeping local version of X"
   (`checkout --ours`), everything else gets "Accepting remote version of X" (`checkout --theirs`).
   Committed as "AUTO-RESOLVE: Merge conflict resolved (kept local pickles, accepted remote code)",
   then "Merge conflict resolved automatically!". **Rule: LOCAL PICKLES WIN, REMOTE CODE WINS.**
4. Final `git pull origin main` -> "Pull complete:" with the git output in a code block.
5. On `GitCommandError`: "Git pull failed: <err>"; if the error mentions divergent branches it tries
   `git pull --rebase` ("Attempting to reconcile divergent branches..." -> "Rebase successful!" or
   "Rebase also failed: <err>" + "Manual intervention may be required." — at that point ssh to the
   machine and resolve by hand).

Consequence of "local pickles win": if machine B pulls while holding OLDER pickles than origin
(because machine A pushed newer ones and B never pulled before diverging), the auto-merge would keep
B's stale pickles. The handoff runbook below exists to prevent ever being in that state.

### /update — push, pull, restart (main.py:193-206 -> update_bot.py:136-161)

Sequence: stop all three queue loops -> "Doing a complete update of the bot!" ->
"Pushing aliases.pkl, stats.pkl, and old_stats.pkl." -> git_push -> "Pulling latest changes from
remote!" -> git_pull -> "Restarting bot!" -> `restart_bot()` (process replaced; per-guild sync runs
on the way back up).

`force_reset:True` replaces push+pull with `git fetch` + `git reset --hard origin/main`
("Force reset mode: Local changes will be discarded!" / "Hard reset complete" / "Warning: All local
changes have been discarded!" — update_bot.py:118-133). **DANGEROUS: this discards local pickles,
i.e. every card parse and alias created since the last successful /push is gone.** When is it ever
right? Only when the clone is wedged (half-finished merge/rebase the bot cannot resolve) AND you have
verified the local pickles are expendable — either already on origin (diff_stats against
`git show origin/main:stats.pkl` reports 0/0/0) or known-corrupt and origin is the good copy.
Treat it as a change-control decision, not an ops reflex: autocroissant-change-control owns the gates.

### Failure messages you may see, and what they mean

| Message | Meaning / action |
|---|---|
| "Pickles are already up to date." | Nothing local is ahead of origin/main after the rebase. Normal. |
| "Push failed: <err>" | Rebase conflict (aborted), fetch/auth failure, or push error (ssh key, a race). A remote that is merely ahead is handled by the rebase. If inside /update, the update aborts. |
| "Git pull failed: <err>" | Merge machinery failed; read the git error. Rebase fallback may follow. |
| "Rebase also failed: ..." + "Manual intervention may be required." | Bot cannot self-heal the clone. SSH in, `git status`, resolve, restart. |
| "Update failed: <err>" + "Bot was NOT restarted due to errors." | /update aborted mid-way. CRITICAL QUIRK: the queue loops were stopped BEFORE the update (main.py:201-203) and nothing restarts them on failure — the bot looks alive but every queue-delivered reply (most commands) goes silent. Recover with `/restart_bot` (sends directly, still works) or a manual restart. `/pull` and `/push` alone do NOT stop the loops. |
| "Restarting bot!" then nothing | Process replaced by execv/execl. Watch the terminal/host for the fresh startup lines. |

## 6. Multi-machine handoff runbook

The pickles are the ONLY state that follows the bot. Per-machine and NOT synced (verified in
.gitignore): `config.py`, `.ssh/`, `reminder.pkl`, `music/`, all `*.csv`/`*.txt` exports, compiled
`*.so`, venvs. Both machines need their own config.py and deploy key already in place.

Moving hosting from machine A (currently running) to machine B:

1. **On A, land the pickles on origin.** In Discord: `/push` and wait for "Successfully pushed!"
   (run the diff_stats gate first — section 5). `/update` also works if you want A to keep running
   until you stop it; plain `/push` is the minimal move.
2. **Stop A**: `/stop_bot` in Discord (or Ctrl-C on A's terminal). Do not run two hosts at once —
   both would answer every command and their pickles would immediately diverge.
3. **On B, refresh the clone BEFORE starting**: from B's repo root, `git pull` in the terminal
   (equivalently: start the bot and immediately `/pull`, but pulling first means you boot on fresh
   code AND fresh pickles).
4. **Verify pickle freshness on B**:
   ```bash
   python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py
   ```
   The summary ends with a `newest timestamp: <card> @ <datetime>` line
   (as of 2026-07-11: `Anubisath Guardian @ 2026-06-18 ...`). It should match what A had — a much
   older date means step 1 or 3 did not land.
5. **Start on B**: `python3 main.py` from B's repo root; check the startup output against section 1
   (especially that no pickle prints "rebuilding..."/"starting fresh...").
6. **Recreate reminders.** `reminder.pkl` is gitignored — reminders DO NOT follow the bot across
   machines (direct implication of the topology). `/list_reminders all:True` on B and re-`/set_reminder`
   anything the group relies on, e.g. a nightly stats update (section 9).
7. **Type emojis follow the bot identity, not the machine.** If B logs in with a different token than A,
   B's first startup uploads its own 126 emojis (section 11). Same token on both = nothing to do.

## 7. Operating /update_stats

**Cost (since 2026-10-20):** a card is re-parsed only when its PSD's git blob sha (from the tree) differs from
`card.blob_sha` (`_psd_changed`), or it is new or moved. Nothing changed: 1 request (the tree, sent with
If-None-Match; GitHub answers 304 and the run ends with "Nothing changed in the card repo since the last
update."). The ETag cache (`_tree_etags`) is in memory only, so the first run after a restart gets the full tree
and compares shas instead. Per re-parsed card: 1 commit page (newest and first commit in one request, 2 lookups run ahead of parsing) + the PSD
by blob sha (4 downloads run ahead of parsing). Small files (authors, edits, icon manifest) download only when their blob changes. `cards:Sonic, Pix`
also re-parses those (after a parser fix); `force_update:True` re-downloads every PSD. The run saves whenever
anything changed, PSD or not. Cards saved before blob shas existed cost one `get_commits(since=...)` pass the
first time. Measured 2026-10-20 (live, 6 PSDs): 16-18 s before the commit lookups moved into a 2-thread pool
beside the downloads, 6.5-8.7 s after; commit requests 25 → 6 (network-speed-results.md on
`docs/Finished/optimization-pass-2026-10-20/network-speed-results.md`).

Re-parses changed card PSDs from the TTSCardMaker repo on GitHub into `stats.pkl`. Entry point
repo_traverser.py:469; slash wrapper main.py:359-371.

> **OPERATING RULE (as of 2026-10-20): just run `/update_stats`.** It always reads GitHub (remote mode).
> Local mode (walking the clone on disk, the old `use_local_repo` / `use_local_timestamp` options) was
> ARCHIVED on 2026-10-20: the options are gone from the slash command and from `update_stats()`, and the
> code lives only on the local branch `archive/local-update-stats` (file `archive/local_update_stats.py` there;
> main has no `archive/` folder), so the bot can't import it.
>
> - **Why it was archived:** local mode with its default `use_local_timestamp:True` used file mtimes as the
>   change signal while every stored timestamp is a GitHub commit date. On 2026-10-19 all 813 local PSDs had
>   an mtime newer than their stored timestamp, so a default local run would re-parse all 813, append every
>   current card to old_stats.pkl (an archive balloon) and replace commit dates with mtimes
>   (docs/BUGS_AND_IMPROVEMENTS.md B6). Its path bug (removeprefix, live 2025-12-02 → 2026-10-19, story:
>   autocroissant-failure-archaeology Entry 1) had been fixed the day before, but production only ever ran
>   remote mode, so a second traversal was a second set of bugs to keep.
> - **Old reminders keep working:** a stored `/update_stats use_local_repo:False` (the form reminders were
>   saved in on 2026-10-19/20) has the retired options dropped when the bot loads it (section 9).
> - **Sandbox sweeps from the clone** (no PSD downloads) still exist for testing only:
>   `update_stats_local()` from branch `archive/local-update-stats`, copied into a sandbox
>   (autocroissant-validation-and-qa method B). It
>   writes the pickles in the current directory, so never run it in the live checkout.
> - **Gate:** diff_stats before any `/push` (section 5). Red flags for a bad run: hundreds of modified cards
>   with `timestamp` as the changed field, or old_stats.pkl growing by more than about 10%
>   (`git cat-file -s HEAD:old_stats.pkl; wc -c old_stats.pkl`). Post-update checks are owned by
>   autocroissant-validation-and-qa.

The three options (slash defaults in parentheses, main.py:367-369). (`use_local_repo` and
`use_local_timestamp` existed until 2026-10-20; see the rule.)

| Option (default) | What it actually does |
|---|---|
| `output_problematic_cards` (True) | Run `CardValidator` on each re-parsed card and post the problems to the channel |
| `force_update` (False) | Re-parse EVERY card and skip the 304 shortcut. The old version goes to old_stats only if the PSD's blob changed or the card moved, so a forced sweep of unchanged cards adds no history. (The Python-level default was True until 2026-10-19) |
| `cards` ("") | Comma-separated card names to re-parse anyway (e.g. after a parser fix); also skips the 304 shortcut. Same archiving rule |

What a run looks like: "Updating card statistics database... This may take a while." -> the bot's
deferred reply gets EDITED to "25 cards updated.", "50 cards updated.", ... (every `UPDATE_RATE = 25`
RE-PARSED cards; repo_traverser.py:32,390) -> the summary (repo_traverser.py:445):

```
N cards changed or new, M unchanged, K moved.
```

- N counts new cards and every re-parse at an unchanged path (forced ones included); M the skipped cards.
- Saving (repo_traverser.py:503-506): `N + K > 0` runs `prune_clean_cards()` (the deletion mechanism: cards
  not seen this traversal move to `old_stats`) and saves; a run that only changed authors, edits, duplicates or
  stored blob shas (`traverser.changed`) saves too. Moves-only runs are saved. Open: a deletion alone does
  not prune (docs/BUGS_AND_IMPROVEMENTS.md B8), and a GitHub 5xx mid-run either raises out of the run (tree) or
  leaves that card stale with the ETag recorded (blob) (B9).
- Then "Card statistics update complete!" and, if enabled, the problem cards: full detail first
  (path + fenced problem list, split into <=1950-char chunks), then a names-only line per card —
  all posted to the invoking channel. There are TWO problem baselines (surface distinction owned
  by autocroissant-validation-and-qa §2), as of 2026-07-11: the STORED/pickled parse-time surface
  is exactly 1 card ("20 Creature Types", MISSPELT TYPE: tornado), but the posted report re-runs
  `CardValidator.validate` per reparsed card, so a full sweep (e.g. `force_update:True`)
  legitimately posts 5 (adds Computer Virus, Anubisath Guardian, Qiraji Soldier, Silithid).
  The corruption alarm is NEW names in the report — not a count of 5 on a full sweep.

API budget: PSDs under `EXCLUDE_FOLDERS` (`Markers`, `MDW`) are skipped before any request, and the commit
lookup (`_get_remote_timestamp`, one per_page=100 page) runs only for re-parsed cards (a new or author-less card
with 100+ commits costs one more request for its first commit). A daily run spends a few requests; `force_update:True` on ~813 cards
spends a large share of the 5000/hr `GIT_TOKEN` budget, so don't schedule it. Without a token (60/hr) a forced
or first-ever run cannot finish; check for the "Git token found" startup line.

PSDs come through the git/blobs API (`_download`/`_blob`, repo_traverser.py:113-140) into a temp file deleted
right after parsing; the icon-font manifest `MDW/Fonts/stoogedings-src/stoogedings.json` and the authors/edits
files (`site/Card Authors.txt`, `site/Card Edits.json`; in `MDW/` before 2026-10-20) through `_repo_text` (cached in memory by blob sha). The offline diagnostics scripts read the manifest
from the clone. If it is missing, icon tokens still parse by plain names (impossibility-cards-reference §4.1).

### TTS save (/update_tts_save, added 2026-10-20; the automatic path is a TTSCardMaker Action since the same day)

The Tabletop Simulator save lives in the card repo at `TTS/Impossibility Simulator.json` (`TTS_SAVE_PATH`).
commands/tts_save.py updates it: image links pinned to commit shas, renames followed, missing cards added to
complete decks, Nicknames set to the website's names, role tags cleared, the Lua resizeTargets table regenerated.
Two ways to run it:

- **Automatic: TTSCardMaker's GitHub Action** `.github/workflows/tts-save.yml` (details: ttscardmaker-companion
  §2.6). On a push to main that changes a PNG, `TTS/decks.json` or `site/Card Edits.json`, daily at 11:41 UTC, or
  by hand (`workflow_dispatch`), it sparse-checks out the updater and its imports from `bobberlington/AutoCroissant`
  main (no config.py: github_api reads it in a try/except), runs `python -m commands.tts_save` on the checkout's save
  with the two env overrides `AUTOCROISSANT_LOCAL_DIR=$GITHUB_WORKSPACE` (where it reads `TTS/decks.json` and
  `site/Card Edits.json`) and `AUTOCROISSANT_HISTORY_CLONE=~/TTSCardMaker-history.git` (its own bare blobless clone;
  global_config.py reads both with `environ.get`), and commits the save as AutoCroissant when it changed (message
  from the report's first section titles). The report is the job log. A save commit matches none of the triggers,
  so it does not run again.
- **By hand: `/update_tts_save`** (admin only). The bot downloads main's copy by blob sha into a temp dir, updates
  it and, when anything changed, commits it to main through the GitHub git data API with `GIT_TOKEN` (message like
  "TTS save: 812 links pinned, 3 cards added"; when the token can't write — 401/403/404 on the blob upload, or no
  token — git objects in the history clone pushed over SSH). If main moved meanwhile, it redoes the update on the
  new save (3 tries). It never writes a checkout; the temp dir is deleted. One run at a time.
- **`/update_stats` does not run it.** Until 2026-10-20 a changed-tree /update_stats queued a run
  (`save_affected` / `update_after_stats`); the Action replaced that and the code was deleted.
- **Inputs (bot run):** the rules, the save and `site/Card Edits.json` (`EDITS_PATHS`) come from main; names from the website (when it is
  unreachable, Nicknames are left as they are); never from a local checkout. The history clone (and its links map) must
  be at main's head, else the run is retried 10 s later, then fails after 3 tries (also when nothing would
  change).
- **Where the report goes (bot run):** the invoking channel (also when a reminder ran it): a short summary
  (header lines and section titles with counts), the full markdown report as `tts_save_report.md`, then the
  download link; on failure "TTS save update failed: ...".
- **Players' download link:** `https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/<commit>/TTS/Impossibility%20Simulator.json`
  (the commit just made; with no change, the `main` link). The 5.5 MB save is never attached in Discord.
- **Rules file:** `TTS/decks.json` in the card repo (deck GUID -> folder globs, include/exclude, copies per card,
  `partial` for hand-picked decks). The bot reads main's copy. Edit it in TTSCardMaker and push; the Action
  picks it up. A rule that matches no PNG, or a malformed deck entry, leaves its deck alone and is listed
  under Errors. Rules only cover PNGs: a .jpg/.jpeg card in a complete deck is never removed.
- **Standalone:** `python -m commands.tts_save <save.json>` (from the bot's root; what the Action runs) updates a
  local file in place and writes `<stem>.<time>.bak.json` and `<stem>.report.md` beside it. `TTS/decks.json` and
  `site/Card Edits.json` come from the checkout at `LOCAL_DIR_LOC` (else raw main / nothing).
- Save commits do not rebuild the card website (`paths-ignore: TTS/**` in TTSCardMaker's site.yml).

## 8. Artifact conventions — where files land

All paths relative to the CWD the bot was started from (= repo root, always).

| Artifact | Produced by | Location | Committed? |
|---|---|---|---|
| `stats.pkl`, `old_stats.pkl`, `aliases.pkl` | /update_stats, /alias, metadata commands | repo root | YES — the database, synced via /push |
| `reminder.pkl` | /set_reminder | repo root | NO (gitignored) — per-machine |
| `reminder.pkl.tmp` | every reminder save (written, fsynced, then renamed over reminder.pkl) | repo root, normally gone within milliseconds | NO (gitignored since 2026-10-19, `.gitignore:41`). If a hard kill leaves one behind, delete it |
| `repo_history.json` | startup and every /update_stats (commands/repo_history.py; a failed refresh retries on the next one) | repo root; its bare clone at `~/Documents/TTSCardMaker-history.git` | NO (gitignored) — per-machine, rebuilt from the clone if deleted |
| `stats.csv` / `stats.txt` | /export_cards (`as_csv` picks which) | repo root | NO (gitignored via `*.csv`/`*.txt`) |
| `rules.txt` | /export_rulebook | repo root | NO (gitignored) |
| Music downloads | /play | `music/`, named `%(title)s-%(id)s.%(ext)s` (restrictfilenames on); playlists get a `music/<playlist title>/` subfolder (music_player.py:52-53, 250-251) | NO (`music/` gitignored) |
| /update_stats PSD temp files | /update_stats | OS temp dir via `mkstemp` (`_download`, repo_traverser.py:127-140); deleted after each parse (repo_traverser.py:363; before 2026-10-19 they piled up, archaeology Entry 10.2) | n/a |
| TTS save working copy | /update_tts_save | OS temp dir (`TemporaryDirectory`), deleted after each run; the only output is the commit to the card repo (the automatic path runs in TTSCardMaker's Action, not here) | n/a |
| Type emojis | /sync_emojis, startup sync, /update_stats on a changed tree | Discord-side application emojis owned by the logged-in bot identity; `type_emojis.json` (repo root) records each one's icon blob, per bot identity | NO (gitignored) — per-machine; if deleted, startup adopts the existing emojis as current |
| `.ssh/` deploy key | operator-provisioned | repo root | NO (gitignored) — per-machine, SECRET |
| Offline test suite | developers (`python -m pytest -q tests`, before a push) | `tests/` (goldens in `tests/goldens/`, re-pinned only by `tests/make_goldens.py`) | YES — code; never writes the pickles |
| 2026-10-20 optimization-pass audits, bench and regression scripts | the optimization pass (read-only audits; sandboxes in /tmp) | reports and `bench/` on (`docs/Finished/optimization-pass-2026-10-20/`: profile.md, network-robustness.md, network-speed-results.md, code-health.md, docs.md, refs_table.md); scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` | YES — scripts on main (same commit as this row), reports on the archive branch only |

## 9. Reminders operations

Rewritten 2026-10-19 (commands/analytics.py, 957 lines; uncommitted as of that date). Times are
Pacific — `TIMEZONE = ZoneInfo("America/Los_Angeles")` (global_config.py:15), so they follow daylight
saving. `check_reminder` is queued once per second by the command loop (main.py:841). History of
what the rewrite fixed: autocroissant-failure-archaeology Entries 10.1 and 10.6-10.8. Flow and
invariants: autocroissant-architecture-contract.

### /set_reminder (admin only, main.py:213)

| Option | Accepts | Notes |
|---|---|---|
| `msg` | text | Posted in this channel when it fires (split into chunks if long) |
| `when` | `13:00`, `1PM`, `1:30PM` (case and spaces tolerated), optionally after `today`, `tomorrow`/`tmrw`, a weekday (3+ letters: `sun 3AM` = the next Sunday 3AM, never in the past) or a date (`2026-12-25 9AM`) | A bare time that already passed today fires right away, and the reply suggests `when:tomorrow ...` or `offset:1d`. A date in the past is refused. Blank `when` is allowed only together with `offset` (counts from now) |
| `offset` | number + `s/m/h/d/w`, combinable (`30m`, `1d12h`) | Added to the first run. Must be over 0 and at most 366 days (`MAX_INTERVAL`) |
| `frequency` | same format as `offset` | Repeat interval; blank = one-shot |
| `command` | `/name option:value ...` | Checked against the command's real slash signature when you create it: unknown options, wrong types and missing required options are refused with "✗ Reminder NOT set: <reason>". Quote multi-word values (`command:/query query:"The Freezer"`). Booleans accept true/false/yes/no/on/off/1/0 in any case. Positional values still work if they come before any `name:value`, but named options are clearer. Stored in canonical `/name option:value` form |

At least one of `msg` / `command` is required. The reply looks like: "Reminder `a1b2c3d4` set for
**2026-10-25 03:00 PDT (in 5 days)**, repeats every 1w, runs `/sync_emojis` with your permissions.
Output goes to this channel."

**Retired options are dropped, not refused.** `RETIRED_OPTIONS` (analytics.py:41) lists options a command
no longer has: since 2026-10-20, /update_stats's `use_local_repo` and `use_local_timestamp` (local mode was
archived, section 7). A new reminder that names one gets it removed, and the reply says so ("Dropped
`use_local_repo:True`: /update_stats no longer has that option."). Stored reminders are cleaned on load (see
Persistence below). From 2026-10-19 until then, `SCHEDULED_DEFAULTS` added `use_local_repo:False` to every
scheduled /update_stats; that mechanism was removed along with the option.

Copy-paste examples:

```
/set_reminder when:22:00 frequency:1d command:/update_stats
/set_reminder when:sun 3AM frequency:1w command:/sync_emojis
/set_reminder when:tomorrow 9AM msg:Game night tonight!
```

The first is stored as `/update_stats`. The second is a weekly full emoji sync, only an example: the live one (f590a57e) was removed on 2026-10-20 because /update_stats keeps emojis current
(section 11).

### Whose permissions a reminder uses

- A reminder runs **as its creator**. It stores `creator_id`/`creator_name`, and the fake interaction it fires
  with carries a stand-in user with that id (`_ReminderUser`, analytics.py:69). The slash wrapper's own
  `lacks_admin` therefore runs again at fire time: if the creator has left `ADMINS`, an admin command replies
  "You do not have permission." in the channel instead of running.
- **Legacy reminders** (saved before 2026-10-19, so no creator) still run as the bot: `LEGACY_RUNS_AS_BOT = True`
  (analytics.py:37). The bot's own id is in `ADMINS` on the dev Mac, so those pass every admin check.
  `/list_reminders` labels them "by unknown (older reminder, runs as the bot)". To bring one under the new
  rule, re-create it and then remove the old one.

### When things go wrong

| Situation | What happens |
|---|---|
| The command crashes | "⚠ Reminder `<id>`: `/<name>` failed (<Error>: <text>)" in the channel; the reminder still repeats |
| The stored command no longer resolves (command renamed, option removed) | The message (if any) still posts, then "⚠ Reminder `<id>` skipped its command: <reason>. Re-create it, then `/remove_reminder reminder_id:<id>`." The reminder still repeats or completes normally |
| Channel deleted or no longer visible | Skipped, with a terminal log line. A repeating reminder moves to its next slot; a one-shot is retried every hour (`CHANNEL_RETRY`) until the channel is back or you remove it |
| Bot was offline, or stalled, past one or more runs | A one-shot fires right after startup. A repeating reminder fires once, then jumps straight to its next future slot. Missed runs are not replayed |
| A reminder crashes the scheduler itself | Only that reminder is disabled ("crashed the scheduler"); `/list_reminders` shows "⚠ DISABLED: <reason>". Remove it |
| reminder.pkl unreadable at startup | Reminders are OFF for the session and the file is left untouched. Every reminder command replies "✗ Reminders are off: ...". Move the file aside and restart for a fresh schedule |
| Two scheduler passes overlap | The second is skipped (single-flight lock), so nothing double-fires |

### /list_reminders, /remove_reminder, persistence

- **/list_reminders** `[all] [hidden]` (not gated): one line per reminder with ID, message, next run (Pacific
  time plus a Discord relative timestamp each reader sees in their own zone), repeat (`1d12h` style), command,
  creator, channel (with `all:True`) and any DISABLED reason. Mentions do not ping. Long lists are split.
- **/remove_reminder** `reminder_id`: admins, or the reminder's creator (anyone else gets "Only an admin or the
  reminder's creator can remove it."). The ID is the 8-character code from /list_reminders; case and
  backticks are ignored.
- **Persistence**: `reminder.pkl` is rewritten atomically on every change (temp file, fsync, rename; section 8
  for the `.tmp` file). The layout is unchanged (guild id → list of dicts), so HEAD-era code can still read a
  migrated file — but that older code would again read `True` as False in commands. On the first start after
  the rewrite, each old reminder is migrated in place: its command is rewritten to canonical form keeping what
  it actually did (old booleans: only lowercase `true` was True). Since 2026-10-20 (`REMINDER_VERSION = 3`)
  the same load also strips retired options (`RETIRED_OPTIONS`), so `/update_stats use_local_repo:False`
  becomes `/update_stats`; old positional values are matched against the option order of their time first.
  The fire path drops them too, so a reminder never fails on them. Watch for "Reminder <id>: migrated command
  ... (dropped retired ...)" lines. Per-machine file: reminders DO NOT
  follow a machine handoff (section 6 step 6).
- **Not yet observed live** as of 2026-10-19 (docs/BUGS_AND_IMPROVEMENTS.md V2): watch the first startup's migration
  lines and the first firing.

## 10. Guild admin quick table

| Command | Gate | Notes |
|---|---|---|
| `/purge [user] [num] [bulk]` | none | Deletes messages in the current channel; defaults to the BOT's own messages, num 100 (`-1` = up to 1,000,000); `bulk:True` is faster but needs the bot to have Manage Messages (main.py:631-642) |
| `/list_guilds` | admin | Name, ID, member count per guild |
| `/leave_guild guild_id` | admin | Numeric ID required |
| `/list_guild_channels guild_id` | admin | Works with default intents |
| `/get_channel_messages guild_id channel_id [limit]` | admin | Default 50, `-1` = all (1M cap); text channels only; oldest first; permission failure -> "I don't have permission to read messages in that channel." |
| `/react reactions [message_id]` | none | Space-separated emojis (`👍 👎` or `<:custom:1234567890>`); no id = most recent message; replies "Added X/Y reactions..." |
| `/sync_global` | admin | Section 4 |

## 11. Type emojis (/sync_emojis)

Added 2026-10-19 (`commands/type_emojis.py`, uncompiled). The bot turns `[fire]`-style tokens in its
replies into emoji icons. As of 2026-10-19 only `/query_rulebook` uses this; it now sends plain escaped
text instead of a code block, because code blocks don't render emojis.

- **What gets uploaded**: Discord *application* emojis (owned by the bot application, usable wherever the
  bot posts, not guild emojis) made from the PNGs in TTSCardMaker `Types/` (35) and `Types/Stars/` (56) on the
  repo set by `/set_repo` (default `MichaelJSr/TTSCardMaker`, branch main), plus a crossed-out `<name>_no`
  copy of each `Types/` icon (35): **126 emojis** as of 2026-10-19. Names and images: see the rules below.
- **They belong to the logged-in bot identity.** On the dev Mac that is "Waxwell" (2026-10-19); the
  production AutoCroissant identity has its own set, uploaded on its own first startup. Same token on two
  machines = same emojis.
- **Startup**: `init_type_emojis` uploads only the MISSING ones (none deleted, none re-uploaded), records existing
  emojis that `type_emojis.json` lacks as current, and loads the token map. Terminal line: "Type emojis: 126 icons and crossed-out variants in <repo>, N uploaded, 126
  ready." Until it finishes, or if it fails ("Emoji sync failed: ..."), tokens stay as plain text.
- **/update_stats keeps them current** (2026-10-20): after a changed, complete tree of the default card repo
  (never on "Nothing changed", never after `/set_repo` to another repo, never from a tree without `Types/` icons),
  `sync_changed_emojis` compares each icon's blob sha with `type_emojis.json` (bot application id -> {emoji
  name -> blob it was made from}, gitignored) and replaces only emojis whose icon is new or edited (with its
  `_no` copy), deletes the ones whose icon is gone, and says one line: "Type emojis: 2 updated, 1 removed"
  (silent when nothing changed; no Discord call then). Icons download from raw links (commit-pinned when
  repo_history knows the blob) and are checked against the blob sha; only a lagging raw main falls back to
  the git/blobs API, so syncs cost no API quota. An emoji with no recorded blob (first start after deploy,
  JSON deleted, a new bot identity) counts as current, so nothing is re-uploaded; an icon edited meanwhile
  needs `/sync_emojis refresh:True`. On any failure the tree ETag is dropped and the line ends "(next
  /update_stats retries)".
- **/sync_emojis `[refresh]` `[prune]`** (admin only): the manual full sync, reported in the channel. Uploads
  missing icons and icons edited since their emoji was made; `refresh:True` re-uploads every icon.
  `prune:True` deletes bot emojis whose icon is no longer in the repo; without it they are only listed ("No
  longer in Types/ (kept, use prune:True to delete)"). It retries failed uploads (not deletes) right away.
- **Rendering rules** (`render_type_tokens`): a known token becomes its emoji; `[fire:no]` becomes the
  `fire_no` emoji, or 🚫 followed by the fire emoji if `fire_no` doesn't exist; unknown brackets such as `[X]`
  or `[Status]` stay as text.
- **Naming and image rules** (needed when a Types/ file is renamed): lowercase; dots and apostrophes
  dropped; any other run of non-alphanumerics becomes `_`; at most 32 characters, cut at a word boundary
  (`thunderfury_blessed_blade_of_the`). Images are 128x128 transparent PNGs, the icon cropped to its visible
  pixels and centered; small pixel art is enlarged nearest-neighbor so it stays crisp; `_no` adds a red ring
  and slash in the same red as the font's `:no`. The font side of these names: ttscardmaker-companion.
- **Requires discord.py 2.5+** (application-emoji API; `requirements.txt` says `discord.py>=2.5`).
- **Not yet run against live Discord** (offline fake-client test only, docs/BUGS_AND_IMPROVEMENTS.md V1). On the
  first live start, check the terminal line above, then run `/query_rulebook` on a page with icons.

## When NOT to use this skill

- Building or fixing the Python environment, ffmpeg/opus, Cython builds, requirements split ->
  **autocroissant-build-and-env**.
- Something is BROKEN (won't start, commands missing, silent bot, stats look wrong, music
  failures) -> **autocroissant-debugging-playbook** (this skill only tells you what healthy looks like).
- Changing code, pickles-as-data decisions, what may be committed, force_reset approval ->
  **autocroissant-change-control**.
- The story behind the local-mode bug and other incidents -> **autocroissant-failure-archaeology**.
- The icon font, the card website the pickles feed, and Types/ icon naming on the TTSCardMaker side ->
  **ttscardmaker-companion**.
- How reminders and type emojis fit the queue architecture (flow, invariants) -> **autocroissant-architecture-contract**.
- Judging whether an update's output is GOOD (golden cards, diff review, problem-count gates) ->
  **autocroissant-validation-and-qa**.
- Script usage details for inspect_pickle/diff_stats/parse_one -> **autocroissant-diagnostics-and-tooling**.
- What a config field or module constant means -> **autocroissant-config-and-flags**.

## Provenance and maintenance

Written 2026-07-11 against the working tree at commit 284d13c ("PICKLE"); sections 1, 5, 7, 8, 9, 11
and the line numbers in 3, 4 and 10 re-verified 2026-10-19 against HEAD a7a4bb3 plus the uncommitted
working tree (adds commands/type_emojis.py); sections 1, 3, 5, 7, 8 again 2026-10-20 (HEAD
9fb3093), and sections 1, 3, 7, 8 and the line numbers after that day's optimization pass. Every claim was verified by
reading main.py, commands/update_bot.py, commands/analytics.py, commands/management.py,
commands/help.py, commands/psd_analyzer.py (and since the split repo_traverser.py, github_api.py),
commands/query_card.py, commands/music_player.py,
commands/utils.py, global_config.py, and .gitignore. Line numbers drift —
re-find volatile facts with these before trusting them:

| Volatile fact | Re-verify with |
|---|---|
| on_ready order / loop starts / startup prints | `grep -n "queue_command(init\|process_.*_queue.start\|Bot initialization complete" main.py` |
| Slash-command count (50 on 2026-10-20: /update_tts_save added, /list_guild_members removed) | `grep -c "@tree.command" main.py` |
| Members intent off, so no /list_guild_members (removed 2026-10-20, C6) | `grep -n "Intents\|list_guild_members" main.py` (Intents lines only) |
| /restart_bot still ungated | `grep -n -A4 'name="restart_bot"' main.py` |
| /update stops loops first | `grep -n -B2 -A6 'name="update"' main.py` |
| startup.sh still absent from repo | `ls startup.sh; git ls-files startup.sh` |
| execv/execl fallback | `grep -n -A4 "def restart_bot" commands/update_bot.py` |
| PICKLE commit convention / push rebase / merge strategy | `grep -n "PICKLE\|--rebase\|rev_list\|--ours\|--theirs\|AUTO-RESOLVE" commands/update_bot.py` |
| Push remote is ssh | `git remote -v` |
| update_stats options are output_problematic_cards, force_update, cards (local mode archived 2026-10-20) | `grep -n -A4 "def slash_update_stats" main.py`; `grep -n -A4 "^def update_stats" commands/repo_traverser.py`; `grep -rn "use_local" main.py commands/` (only analytics.py `RETIRED_OPTIONS`) |
| Local traversal archived on its own branch, not in main | `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived`; `ls archive 2>/dev/null` (nothing: main has no archive folder) |
| Retired reminder options dropped | `grep -n -A6 "^RETIRED_OPTIONS" commands/analytics.py`; `grep -n "^REMINDER_VERSION" commands/analytics.py` (3) |
| Blob-sha change check, blob downloads, 304, summary, save gate | `grep -n "_psd_changed\|def _download\|def _blob\|fetch_tree\|changed or new\|num_changed > 0 or traverser.changed" commands/repo_traverser.py` |
| old_stats archived only on blob change or move | `grep -n -B1 "old_stats\[name\].append(old)" commands/repo_traverser.py` |
| Per-card cost (exclusions skipped, commit lookup only for re-parsed cards) | `grep -n "EXCLUDE_FOLDERS\|_get_remote_timestamp\|COMMITS_PAGE =" commands/repo_traverser.py` |
| Reminder formats, limits, defaults | `grep -n "^TIME_FORMATS\|^WEEKDAYS\|^MAX_INTERVAL\|^CHANNEL_RETRY\|^LEGACY_RUNS_AS_BOT\|^INTERVAL_UNITS" commands/analytics.py` |
| Reminders run as creator; remove gated | `grep -n "_ReminderUser(\|Only an admin or the reminder" commands/analytics.py` |
| reminder.pkl.tmp ignored | `git check-ignore -v reminder.pkl.tmp` (`.gitignore:41`) |
| TTS save wiring (bot: /update_tts_save only; no tts_save import in repo_traverser since 2026-10-20) | `grep -n "update_tts_save" main.py`; `grep -c "tts_save" commands/repo_traverser.py` (0); `grep -n "^TTS_SAVE_PATH\|environ.get" global_config.py` (the two Action env overrides) |
| Type emoji sync wiring | `grep -n "init_type_emojis\|sync_emojis" main.py`; `grep -n "^EMOJI_SIZE\|^MAX_NAME_LEN\|^NO_SUFFIX" commands/type_emojis.py` |
| Pickle names / timezone | `cat global_config.py` |
| Per-machine files | `cat .gitignore` |
| Music/export paths | `grep -n "MUSIC_BASE_DIR\|outtmpl" commands/music_player.py; grep -n "EXPORTED_" commands/psd_analyzer.py` |
| Card/rulebook dataframe counts, newest pickle timestamp | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |

Maintenance: update the date stamps and counts above whenever the pickles are refreshed or commands
are added. If local mode is ever revived from branch `archive/local-update-stats`, it must first stop comparing
mtimes with commit dates (B6); then revisit section 7's operating rule and the `RETIRED_OPTIONS` note in
section 9, keeping the pointer to autocroissant-failure-archaeology for the history.

2026-10-20 (later): the optimization-pass pointers above moved from `~/Documents/ac-optimization/` to
`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` (scripts) and `docs/Finished/optimization-pass-2026-10-20/` (reports, bench).
