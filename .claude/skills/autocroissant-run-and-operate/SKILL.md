---
name: autocroissant-run-and-operate
description: Day-to-day operations runbook for the AutoCroissant Discord bot. Load this when the task sounds like "run the bot", "start the bot", "stop", "restart", "deploy", "update the bot", "push/pull the pickles", "switch machines" / "handoff to the other machine", "sync commands" / "commands not showing up after deploy", "run update_stats", "local mode vs remote mode", "set a reminder" / "schedule a command" / "reminder didn't run" / "/reminders panel" / "reminder.json", "sync emojis" / "/sync_emojis" / "type emojis not showing", "TTS save" / "Tabletop Simulator save" / "/update_tts_save" / "TTS/decks.json", "card guards" / "CARD_GUARDS" / "push guards" / "stale PNG" / "standards channel", or "where do files go" (exports, music downloads, temp files). Contains the start command and healthy startup output, on_ready init order, stop/restart (jetson systemd), the command-sync model, the ANATOMY of /push, /pull, /update (PICKLE commits, pickles-ours/code-theirs merge, force_reset, every failure message; live triage is autocroissant-debugging-playbook), the machine handoff runbook, the /update_stats operating rule (just run it), artifact locations, reminder, type-emoji and TTS-save operations, a guild-admin table, and (section 12) the bot after the data-build cutover: "cutover", "CARD_SYNC", "card sync", "card_data/", "ACTIONS_TOKEN" (/update_stats syncs the Card sync Action's data branch, force_update and cards dispatch a run, /push retired, /fonts from the data).
---

# AutoCroissant: Run and Operate

Operational runbook, verified against the code on 2026-07-11; sections 1, 5, 7, 8, 9 and 11 re-verified
2026-10-19 against the uncommitted working tree (HEAD a7a4bb3); sections 1, 3, 5, 7 and 8 re-verified
2026-10-20 (HEAD 9fb3093; sections 1, 3, 7, 8 again after that day's optimization pass); sections 1, 3, 5, 7, 8, 9, 10
and 12 re-verified 2026-10-23 (main 2ada357, deployed on the jetson at 10:36). Audience: an operator (human or model)
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
      main.py:13 does `from config import TOKEN` — a missing/invalid config.py kills the process at import.
- [ ] No stale compiled artifacts: if `*.so` files exist from an old Cython build they shadow the `.py`
      files you just pulled. `python3 setup.py clean` first (none on the Mac as of 2026-10-19; the jetson
      rebuilds them on every deploy, section 3; details in autocroissant-build-and-env).

Start it:

```bash
python3 main.py
```

### What healthy startup output looks like (the jetson journal of 2026-10-23 10:36)

```
Git token found, API limited to 5000 requests/hour.    <- import time: main.py:89 print_token_status() (github_api.py); "No git token in config..." if GIT_TOKEN unset
We have logged in as AutoCroissant#....                <- on_ready (main.py:114-118)
Registered 52 slash commands for reminder system.      <- slash_registry populated (main.py:130-132); 52 since 2026-10-21 (/fonts and /bake_cards added; 50 on 2026-10-20 after /list_guild_members was removed)
Bot initialization complete                            <- the four loops started (main.py:141-144)
Synced commands to guild: <name> (<id>)                <- one line per guild, async (main.py:110)
Removed N stale temp PSDs (X MB) from <dir>            <- sweep_temp_psds (2026-10-22), only when it deletes something
Trying to open reminder.json.                          <- init_reminder (or "No reminder.json yet: an empty schedule.")
Converted reminder.pkl to reminder.json (3 reminders; backup reminder.pkl.bak).   <- once, the first start after 2026-10-24; plus any "Rescheduled reminder <id> from <t1> to <t2>" lines
Trying to open stats.pkl                               <- init_psd
Loaded existing stats from stats.pkl
Trying to open old_stats.pkl
Loaded existing old stats from old_stats.pkl
Loaded 764 cards and 48 rulebook entries               <- prep_dataframes (query_card.py:535); 2026-10-23 (765 on 2026-07-11)
Stats database initialized.
Loading aliases from aliases.pkl                       <- init_query (1 GitHub tree API call follows)
Repo history: 2268 files at f219a126 (0.3s)            <- refresh_history (repo_history.py:230); 2026-10-23
Type emojis: 126 icons and crossed-out variants in MichaelJSr/TTSCardMaker, 0 uploaded, 126 ready.   <- init_type_emojis (2026-10-19); N uploaded = 0 once synced
```

Lines after "Bot initialization complete" come from queued async work and can interleave. After a failed RESUME
discord.py calls `on_ready` again; since 2026-10-21 that repeat runs no init, syncs only new guilds, restarts the loops
whose task failed and prints `Reconnected: inits skipped, revived <loops or "no loops">` (main.py:125-128).
The token-presence line prints ONCE, from main.py (`print_token_status`; since the 2026-10-20 split the card modules print nothing at import, so scripts show no token line).

### on_ready init order (main.py:114-144)

`on_ready` spawns a per-guild command sync task, populates `slash_registry` (command name -> callback; this is what
lets reminders run bot commands), then QUEUES six commands (main.py:134-139) and finally starts the reminder heartbeat
and the three queue-drain loops through `_start()` (main.py:141-142). It does all that once (`_started`; a reconnect
only revives failed loops, above). Queued work only executes once the loops
start draining — so the loops effectively start first, and nothing is sent to Discord before they run.

| Init (queued in this order) | What it does | Failure/no-op behavior |
|---|---|---|
| `sweep_temp_psds` (card_runner.py:103; 2026-10-22, c0fc117) | Deletes mkstemp `tmp*.psd` files (`TEMP_PSD`) older than 1 h (`TEMP_AGE`) in the temp dir, left by a cut-off run (the jetson had 56, 1.1 GB, on 2026-10-22), under `card_runner.bake_lock` | Prints `Removed N stale temp PSDs (X MB) from <dir>` only when it deleted something; never touches a file of a running update or bake |
| `init_reminder` (analytics.py:1474) | Loads `reminder.json` (once, 2026-10-24: converts `reminder.pkl` to it, checked by reading it back, and renames the pickle `reminder.pkl.bak`), migrates old entries, rolls past-due REPEATING reminders that are on forward to their next future slot in one step, saves atomically | No file: an empty schedule. Unreadable file: "ERROR: reminder.json could not be read ... Reminders are OFF" (the file is left untouched; every reminder command replies "✗ Reminders are off"). A failed conversion: "ERROR: could not convert reminder.pkl ..." and the bot runs and saves from the pickle this session; the next start tries again. Only `reminder.pkl.bak` left: reminders off until you restore `reminder.json` or rename the backup back |
| `init_psd` (psd_analyzer.py:326) | Loads `stats.pkl` + `old_stats.pkl` into `stats_db`, builds the query DataFrames | Missing pickles: rebuilds empty (dangerous if unexpected — see cwd warning) |
| `init_query` (query_card.py:991) | Loads `aliases.pkl`, fetches the TTSCardMaker GitHub file tree (1 API call; 3 tries 1 s and 2 s apart on a connection error, 429 or 5xx, `github_api.get_with_retry`) | A failed or non-tree reply keeps the last good list: "Card list: GitHub tree fetch for <repo> failed (...); keeping the last good list (N entries)" (0 entries right after a start, so /query finds nothing until the next /update_stats or restart) and "Warning: Failed to populate files from repository (status N)" |
| `refresh_history` (repo_history.py:198) | Fetches the bare history clone (`~/Documents/TTSCardMaker-history.git`) and maps each card PNG to the commit that last changed it, for sha-pinned image links | Never raises; on failure the old map stays and links fall back to `main` |
| `init_type_emojis(client)` (type_emojis.py:413; added 2026-10-19) | Uploads any MISSING type emojis for the logged-in bot identity and fills the `[token]` → emoji map (section 11). Shares the card list's tree (since 2026-10-20 the second startup tree fetch is a 304, 0 bytes) + 1 Discord list call; uploads only on first start per identity | A failed GitHub listing falls back to the card list's last tree ("using the last good card tree"); any other failure prints "Emoji sync failed: <err> (using the N existing emojis)" and the existing emojis still render. Nothing else depends on it |

AI generation (and its `init_pipeline`) lives only on branch `archive/ai-generation`; main has none.

The three drains (main.py:932-964) each tick every 0.1 s (once a second until 2026-10-21): `process_command_queue`,
`process_dispatch_queue` (message/file sends), `process_edit_queue` (edits). A fourth loop, `process_reminders`
(main.py:926-928), enqueues `check_reminder` once a second — that is the reminder heartbeat. All bot output rides
these queues, so replies have up to ~0.1 s of queue latency by design. Depth: autocroissant-architecture-contract.

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
| `/stop_bot` | admin | Stops the four loops, replies "Stopping bot!", `client.close()`, `exit(0)` (main.py:178-189) |
| `/restart_bot` | NONE (verified main.py:167-175 on 2026-10-23 — no lacks_admin; anyone in the guild can restart) | Stops the loops, replies "Restarting bot!", then `restart_bot()` |
| Ctrl-C in the terminal | local | discord.py shuts down; main.py:988-989 prints "Bot shutting down via keyboard interrupt". Any other fatal error prints "Fatal error: ..." and exits 1 |

`restart_bot()` (commands/update_bot.py:71-75) does `execv('./startup.sh', argv)`; on
`FileNotFoundError` it falls back to `execl(python, python, main.py ...)` — i.e. re-exec the same
interpreter on main.py. The process is REPLACED, so silence after "Restarting bot!" is normal.

**Production (the jetson, 2026-10-20):** systemd `autocroissant.service`; no startup.sh, so /restart_bot
re-execs main.py. Deploy (2026-10-24): run `~/Documents/AutoCroissant/scripts/deploy.sh` ON the jetson (or
`ssh jetson Documents/AutoCroissant/scripts/deploy.sh`). It refuses 05:00-05:40 Pacific (checked again before the
restart), `git pull --ff-only`, reinstalls the psd-tools fork only when `requirements-cards.txt`'s URL and tag differ from
the venv's (pip's `direct_url.json`), `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` (incremental: only
modules whose source changed, seconds when none did; a full rebuild was ~6 min 44 s), `sudo systemctl restart
autocroissant`, then waits up to 90 s for `Registered N slash commands` (52) and `Bot initialization complete` in the
new process's journal (`_PID=` its MainPID; a Traceback from it, or a changed MainPID, fails early) and prints OK, else the last 30 journal lines and FAILED (exit 1; a failed pull or build stops before the
restart). `--bot ojwikibot` does the same for ~/Documents/OJWikibot (no psd-tools step; its own setup.py still rebuilds
everything). Later lines, not checked: `Repo history: ...` and `Type emojis: ... ready`.
The rebuild is NOT optional since the 2026-10-20 split: stale `.so` next to new `.py` fails at import
(`cannot import name perms_check` from analytics.so) and systemd restarts into the same error. Standing rules
(2026-10-22/23 incidents; memory jetson-server):

- **Never deploy between 05:00 and 05:40**: the 05:00 `/update_stats` (reminder d2664752) and the 05:30 `/push`
  (054b5933) run then.
- **Push main and deploy in one sitting**: if origin/main has new code at 05:30 and the jetson does not, `/push` rebases
  the new `.py` files in next to stale `.so` files and the next restart fails at import.
- **When the psd-tools fork tag moves** (its one pin: `requirements-cards.txt`, which `requirements.txt` includes),
  reinstall it in the venv after the pull (deploy.sh does this):
  `~/venvs/autocroissant/bin/pip install --force-reinstall --no-deps "$(grep '^psd-tools' requirements-cards.txt)"`
  (gcc compiles `_distance`), then `~/venvs/autocroissant/bin/pip check`. A plain `pip install -r requirements.txt`
  keeps the old commit while the fork's version is unchanged (1.25.0 at every tag up to `cards-20261023d`): pip calls
  it satisfied (checked 2026-10-23). `pip freeze | grep psd` shows the commit. Today: tag `cards-20261023` = ecc4b773,
  installed 2026-10-23 10:36. The Actions install the same file (autocroissant-change-control, class (a)).
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

- On EVERY start, `sync_guild_commands` (main.py:102-110) runs per guild (a reconnect: only guilds not synced yet): `tree.copy_global_to(guild)`
  then `tree.sync(guild=guild)`. Guild-scoped syncs take effect essentially immediately, so after a
  restart the commands are current in every guild the bot is in (a stale Discord client may need a
  restart/reload to show them — client-side cache, not the bot).
- `/sync_global` (admin; main.py:757-762 -> management.py:75-97) clears the tree's global command set,
  syncs that (now-empty) global set, then re-copies and re-syncs each guild, finishing with
  "Commands synced globally and to N guilds." Use it to clean up duplicated/stale command entries
  (e.g. commands appearing twice because a global copy lingers). Discord platform note (not
  repo-verified): GLOBAL command changes can take up to ~1 hour to propagate; guild ones do not —
  which is exactly why this bot syncs per guild.
- If commands are missing/duplicated after a deploy: restart first (re-registers and re-syncs
  everything), then `/sync_global` if duplicates persist. Deeper triage: autocroissant-debugging-playbook.

## 5. The self-update flow (/push, /pull, /update)

This section is the bot before the cutover (`CARD_SYNC` off, global_config.py); with it on, /push only replies with the Card
sync link, /update pulls and restarts without a push, and /pull keeps no local file on a conflict: section 12.

All three are admin-gated. This is how the bot updates ITSELF from Discord. Remote: fetch over
https, push over `ssh://git@github.com/bobberlington/AutoCroissant.git` (verified `git remote -v`);
the gitignored `.ssh/` directory in the repo root holds the deploy key (never commit or print it).

### /push — commit, rebase, push the pickles (update_bot.py:78-110)

`git add aliases.pkl stats.pkl old_stats.pkl psd_details.json` (the details file travels with the pickles since 2026-10-21) -> `git commit -m "PICKLE"` (no-op if nothing changed) ->
`git pull --rebase --autostash origin main` (picks up code pushed since the last pull) -> `git push`.
Pickle snapshots are commits named PICKLE.
Success: "Successfully pushed!" (also pushes a PICKLE commit left by an earlier failed push). When the push carried
one of those four files, the bot then dispatches the website build (TTSCardMaker site.yml) with `ACTIONS_TOKEN`
(card_sync.dispatch_site; one console line, never a failed push; without the token the site's 6-hour schedule stays).
Nothing ahead of origin/main after the rebase: "Pickles are already up to date.".
Rebase fails (conflict, network, auth): `git rebase --abort`, then "Push failed: <err>" and a re-raise; the
local PICKLE commit stays. If no rebase had started (fetch failed or timed out), the pull's error is shown.

GATE (owned by autocroissant-change-control, summarized here): pickle corruption is the #1
costliest failure class in this project's history — run the diff before every push:

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl && git show HEAD:old_stats.pkl > /tmp/old_stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl \
    --history /tmp/old_stats_head.pkl old_stats.pkl
```

Since 2026-10-19 a pushed PICKLE is also PUBLIC: the TTSCardMaker card website builds from the
pickles on AutoCroissant's `main` (every TTSCardMaker push, every 6 h, or on demand), so a bad push
shows up on the site within hours. Site mechanics: ttscardmaker-companion.

### /pull — fetch and merge (update_bot.py:113-190)

**/pull does NOT hard-reset anything** (its description and help text said so until 2026-10-20, C1;
both now say "fetches and merges"). Actual behavior:

1. "Fetching from remote..." — `git fetch`.
2. If local HEAD != origin/main and the histories diverged: "Branches have diverged. Attempting to
   resolve..." then `git pull origin main --no-edit`. Clean merge: "Merge successful!".
3. On CONFLICT: "Merge conflict detected. Resolving automatically..." — per conflicted file:
   pickles (`aliases.pkl`/`stats.pkl`/`old_stats.pkl`) and `psd_details.json` get "Keeping local version of X"
   (`checkout --ours`), everything else gets "Accepting remote version of X" (`checkout --theirs`).
   Committed as "AUTO-RESOLVE: Merge conflict resolved (kept local pickles, accepted remote code)",
   then "Merge conflict resolved automatically!". **Rule: LOCAL PICKLES WIN, REMOTE CODE WINS.**
   After either merge, a card the file-by-file merge dropped from both pickles (stats.pkl from one side,
   old_stats.pkl from the other) goes back into old_stats.pkl from the side that had it, committed as PICKLE:
   "Kept in old_stats (the merge dropped them): ..." (a failed check says so and the pull goes on).
4. Final `git pull origin main` -> "Pull complete:" with the git output in a code block.
5. On `GitCommandError`: "Git pull failed: <err>"; if the error mentions divergent branches it tries
   `git pull --rebase` ("Attempting to reconcile divergent branches..." -> "Rebase successful!" or
   "Rebase also failed: <err>" + "Manual intervention may be required." — at that point ssh to the
   machine and resolve by hand).

Consequence of "local pickles win": if machine B pulls while holding OLDER pickles than origin
(because machine A pushed newer ones and B never pulled before diverging), the auto-merge would keep
B's stale pickles. The handoff runbook below exists to prevent ever being in that state.

### /update — push, pull, restart (main.py:212-230 -> update_bot.py:212-240)

Sequence: stop all four loops -> "Doing a complete update of the bot!" ->
"Pushing aliases.pkl, stats.pkl, old_stats.pkl and psd_details.json." -> git_push -> "Pulling latest changes from
remote!" -> git_pull -> "Restarting bot!" -> `restart_bot()` (process replaced; per-guild sync runs
on the way back up).

`force_reset:True` replaces push+pull with `git fetch` + `git reset --hard origin/main`
("Force reset mode: Local changes will be discarded!" / "Hard reset complete" / "Warning: All local
changes have been discarded!" — `git_reset_hard`, update_bot.py:193-208). **DANGEROUS: this discards local pickles,
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
| "Update failed: <err>" + "Bot was NOT restarted due to errors." | /update aborted mid-way. CRITICAL QUIRK: the queue loops were stopped BEFORE the update (main.py:225-228) and nothing restarts them on failure — the bot looks alive but every queue-delivered reply (most commands) goes silent. Recover with `/restart_bot` (sends directly, still works) or a manual restart. `/pull` and `/push` alone do NOT stop the loops. |
| "Restarting bot!" then nothing | Process replaced by execv/execl. Watch the terminal/host for the fresh startup lines. |

## 6. Multi-machine handoff runbook

The pickles are the ONLY state that follows the bot. Per-machine and NOT synced (verified in
.gitignore): `config.py`, `.ssh/`, `reminder.json`, `music/`, all `*.csv`/`*.txt` exports, compiled
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
   (at PICKLE a7c310e, 2026-10-23: `Clot White @ 2026-10-22 20:56:26`). It should match what A had — a much
   older date means step 1 or 3 did not land.
5. **Start on B**: `python3 main.py` from B's repo root; check the startup output against section 1
   (especially that no pickle prints "rebuilding..."/"starting fresh...").
6. **Recreate reminders.** `reminder.json` is gitignored — reminders DO NOT follow the bot across
   machines (direct implication of the topology). Copy it from A with both bots stopped, or browse
   `/reminders` on A and re-`/set_reminder` on B what the group relies on (section 9).
7. **Type emojis follow the bot identity, not the machine.** If B logs in with a different token than A,
   B's first startup uploads its own 126 emojis (section 11). Same token on both = nothing to do.

## 7. Operating /update_stats

**Cost (since 2026-10-20):** a card is re-parsed only when its PSD's git blob sha (from the tree) differs from
`card.blob_sha` (`_psd_changed`), or it is new or moved. Nothing changed: 1 request (the tree, sent with
If-None-Match; GitHub answers 304 and the run ends with "Nothing changed in the card repo since the last
update."). The ETag cache (`_tree_etags`) is in memory only, so the first run after a restart gets the full tree
and compares shas instead. Per re-parsed card: 1 commit page (newest and first commit in one request, 2 lookups run ahead of parsing) + the PSD
(4 fetched ahead of parsing). Since 2026-10-23 (`RANGE_READS`, github_api.py:27) a psd-tools with `range_reader` (the fork on
the jetson) reads a changed PSD by HTTP range requests to raw.githubusercontent at the tree's commit, which are not REST API
requests (a run with changed cards adds one `branches/main` API request to find that commit); 3 refusals turn ranges off for
the run and the rest download whole by blob sha. Small files (authors, edits, icon manifest) download only when their blob changes. `cards:Sonic, Pix`
also re-parses those (after a parser fix); `force_update:True` re-downloads every PSD. The run saves whenever
anything changed, PSD or not. Cards saved before blob shas existed cost one `get_commits(since=...)` pass the
first time. Measured 2026-10-20 (live, 6 PSDs): 16-18 s before the commit lookups moved into a 2-thread pool
beside the downloads, 6.5-8.7 s after; commit requests 25 → 6 (network-speed-results.md on
`docs/Finished/optimization-pass-2026-10-20/network-speed-results.md`).

**One-time re-parse after a parser bump (2026-10-21):** every parse is stamped `PARSER_VERSION` (psd_parser.py:23, now 2:
the printed title read into `display_name`), and `_psd_changed` re-parses a card whose stamp differs (card_data.py:240;
the tree ETag is skipped while any card is stale, :267). So the first `/update_stats` after deploying a bump downloads every
PSD once (~813 blobs plus their commit pages, minutes; "813 cards changed or new"; no history grows: the blobs are unchanged)
and later runs are cheap again. Expect it after the 2026-10-21 deploy; no `force_update` needed. Once re-parsed, `/query` titles its
embed with the printed title (`query_card.printed_title`; an embed instead of a bare image URL): 157 cards, including the 94 that differ
from the file name only by letter case or spacing (POORLY DRAWN CABBAGE, A P E X), shown as printed (the user's call, 2026-10-21); the
website shows the same name, and `/query` finds a card by its printed title (`_populate_titles`, refreshed after every `/update_stats`).

Re-parses changed card PSDs from the TTSCardMaker repo on GitHub into `stats.pkl`. Entry point
repo_traverser.py:61 (the bake lock, the Discord messages, the follow-ups); since 2026-10-22 the traversal itself is
`card_data.update` in legacy mode (commands/card_data.py:1117, uncompiled, shared with the Card sync Action's build);
slash wrapper main.py:405-424.

After the cutover (`CARD_SYNC` on) `/update_stats` re-parses nothing: it loads the data the Card sync Action built (section 12).
This section describes it before.

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

The three options (slash defaults in parentheses, main.py:415-418). (`use_local_repo` and
`use_local_timestamp` existed until 2026-10-20; see the rule.)

| Option (default) | What it actually does |
|---|---|
| `output_problematic_cards` (True) | Run `CardValidator` on each re-parsed card and post the problems to the channel |
| `force_update` (False) | Re-parse EVERY card and skip the 304 shortcut. The old version goes to old_stats only if the PSD's blob changed or the card moved and its data changed, so a forced sweep of unchanged cards adds no history. (The Python-level default was True until 2026-10-19) |
| `cards` ("") | Comma-separated card names to re-parse anyway (e.g. after a parser fix); also skips the 304 shortcut. Same archiving rule |

What a run looks like: "Updating card statistics database... This may take a while." -> the bot's
deferred reply gets EDITED to "25 cards updated.", "50 cards updated.", ... (every `UPDATE_RATE = 25`
RE-PARSED cards; card_data.py:43, :675) -> the summary (card_data.py:691):

```
N cards changed or new, M unchanged, K moved.
```

- N counts new cards and every re-parse at an unchanged path (forced ones included); M the skipped cards.
- Saving (`card_data.update`, called by `_update_stats`): every full run (`can_prune`: no 304, tree not truncated, no
  failed new-name download or unreadable new-name PSD, some cards) runs `prune_clean_cards(at)`, which archives the cards whose PSD left the
  tree (moved to the end of their old_stats list, tagged `archived`; a deletion alone is enough since 2026-10-22, was
  B8) and tags names that left stats.pkl before archiving existed. Messages: "Archived (PSD deleted from the card
  repo): A, B (now C)" (C: the rename's new name), "Archived N cards deleted before this update: ..." (backfill, once),
  "Restored (PSD back in the card repo): ..." (a PSD back under an archived name: the record returns, no download for
  the same blob), "Card repo tree truncated or without cards; no card archived." Since 2026-10-23 also: "Archived alt-art
  cards (their PNG or original left the card repo): X (same card as Y)", "Lines still keyed by an archived name (move them
  to its new name, or drop them): ..." (Card Authors.txt / Card Creation Dates.json lines), "Back under a new PSD (the
  archived record stays as history): ..." and, on the console, `<file>: N names match no card: ...` for side-file lines
  no card owns. Archive tags are dated from git (the deleting commit; `renamed_to` for an exact rename, `moved_to` for a
  move into MDW/, e.g. White Palace to MDW/The Morgue). The run saves when anything changed
  (cards, archives, authors, edits, duplicates, blob shas, a stale archived copy). Archived cards: read them with
  `StatsDatabase.archived()`; no query or export sees them. A PSD that cannot be read, or whose commit lookup fails (a
  5xx, a rate limit), fails only its card: "Could not read N PSDs ..." / "Could not look up the commits of N PSDs ...";
  its card keeps its last version and the next run tries again (no ETag kept). Open: a GitHub 5xx on the tree (B9).
- Then "Card statistics update complete!" and, if enabled, the problem cards: full detail first
  (path + fenced problem list, split into <=1950-char chunks), then a names-only line per card —
  all posted to the invoking channel. There are TWO problem baselines (surface distinction owned
  by autocroissant-validation-and-qa §2), as of 2026-10-23: the STORED/pickled parse-time surface
  is 0 cards (20 Creature Types' `tornado` layer became `wind`, TTSCardMaker 100511d3), but the posted report re-runs
  `CardValidator.validate` per reparsed card, so a full sweep (e.g. `force_update:True`) legitimately posts 1 (Computer
  Virus, HP/DEF/ATK/SPD NOT FOUND; the owner leaves it). On 2026-07-11 it was 1 and 5: Anubisath Guardian, Qiraji
  Soldier and Silithid joined ABILITY_EXCLUSIONS in e4e0a58 (2026-10-21).
  The corruption alarm is NEW names in the report — not the count on a full sweep.
- **Authors from git (.mailmap, 2bf00e1, 2026-10-23):** git author names go through the card repo's `.mailmap`
  (`global_config.MAILMAP_PATH`): in clones `card_history.mailmap_args` passes head's blob as `-c mailmap.blob=` for `%aN`;
  on the API path `card_edits.Mailmap` is read from the tree once a run. TTSCardMaker's `.mailmap` (74bb56e7, pushed in
  a7d1fe56) maps "Michael <...>" to MichaelJSr, and the website's `[people] Michael` special case is gone; the site build and
  card-sync.yml check out `/.mailmap`. One person's several git names show as one; to merge names, edit
  TTSCardMaker/.mailmap, not the bot.

### History cleanup and recovery (every legacy run, 2026-10-23)

Before the traversal, `card_data.update` runs `card_data.cleaned` (card_data.py:1077, `clean_history` :1012): with the
history clone's index (`HISTORY_CLONE`, a full clone: `history_index`) and the bot's own PICKLE commits (`BOT_CLONE`, else
this checkout; the jetson's `~/Documents/AutoCroissant` is full) it recovers the history lost in the 2026-02-08 cleanup and
the 2025 dict-era snapshots, archives the cards that left before archiving existed, merges type misreads (the PNG stayed),
sets impossible legacy values to unknown, drops duplicate versions, brings stored versions to today's series and type
names, dates archive tags from git and gives carried starts their commit. It works on a copy: one `History: ...` line
when it changed something; an error prints `History cleanup skipped (<Error>: ...)` and changes nothing. After the
parse it runs once more over what the run read (a second `History: ...` line when it finds something;
`History cleanup after the update skipped (...)` on an error), so the next run finds nothing left.

- **The recovery is one time only** (`card_history.recovered_before`: once a restored version is stored, no PICKLE commit
  is read again). Without the clones it skips quietly and the next run does it: console `History clone <dir> not read
  (...)`, or `History recovery: <clone> is no full clone (git fetch --unshallow); the PICKLE commits wait for one`.
- **The next run (the first since the 10:36 deploy of 2026-10-23; the 05:00 reminder or the owner), as rehearsed:**
  "History: 698 versions recovered from the bot's PICKLE commits (553 cards), 171 starts moved earlier; 64 cards gone
  before archiving archived: ...; 42 type misreads merged (the PNG stayed): ...; 12 impossible values set to unknown:
  ...; 32 brought to today's series and type names; 299 carried starts given their commit." plus about 219-236 re-parsed
  PSDs (the Photoshop batch B2, TTSCardMaker 7fd2aa30) and a second History line ("103 carried starts given their commit",
  known noise). old_stats goes from 446 versions / 354 names to about 1,155 / 670.
- **Gate and undo:** the growth is expected; read it with `diff_stats.py --history` (section 5's gate) before `/push`. To
  undo the recovery, revert that run's PICKLE commit (pickle-only; autocroissant-change-control).
- **Cutover precondition:** the data branch's seed PICKLE must be one pushed after this run: the runners' checkouts are
  shallow, so the Action's seed pass never recovers (DATA_BUILD_ACTION.md §14 step 7; section 12).

API budget: PSDs under `EXCLUDE_FOLDERS` (`Markers`, `MDW`) are skipped before any request, and the commit
lookup (`_get_remote_timestamp`, one per_page=100 page) runs only for re-parsed cards (a new or author-less card
with 100+ commits costs one more request for its first commit). A daily run spends a few requests; `force_update:True` on ~812 PSDs
spends a large share of the 5000/hr `GIT_TOKEN` budget, so don't schedule it. Without a token (60/hr) a forced
or first-ever run cannot finish; check for the "Git token found" startup line.

PSDs come by range reads (above) or through the git/blobs API (`_download`/`_blob`, card_data.py:150-187) into a temp
file deleted right after parsing; the icon-font manifest `MDW/Fonts/stoogedings-src/stoogedings.json` and the authors/edits
files (`site/Card Authors.txt`, `site/Card Edits.json`; in `MDW/` before 2026-10-20) through `_repo_text` (cached in memory by blob sha). The offline diagnostics scripts read the manifest
from the clone. If it is missing, icon tokens still parse by plain names (impossibility-cards-reference §4.1).
`site/Card Ratings.json`: card ratings (`python -m commands.tts_ratings`); never read by the bot.

### TTS save (/update_tts_save, added 2026-10-20; the automatic path is a TTSCardMaker Action since the same day)

The Tabletop Simulator save lives in the card repo at `TTS/Impossibility Simulator.json` (`TTS_SAVE_PATH`).
commands/tts_save.py updates it: image links pinned to commit shas, renames followed, missing cards added to
complete decks (an empty one filled from a template card at its own Transform), new cards added to hand-picked decks
with `add_new_since`, Nicknames set to the website's names, the Lua resizeTargets table regenerated. An existing card's
Description, GMNotes, Memo, Tags and scripts are the team's: never changed (role tags were cleared once, until
2026-10-22); new cards start with them empty.
Two ways to run it:

- **Automatic: TTSCardMaker's GitHub Action** `.github/workflows/tts-save.yml` (details: ttscardmaker-companion
  §2.6). Daily at 11:41 UTC when its inputs changed since the last successful run (a PNG, `TTS/decks.json`,
  `site/Card Edits.json`, the save, or the card browser's names: a hash of cards.json's png -> name map, cached per set
  the updater ran with; TTSCardMaker 075e6c43, 2026-10-22), or
  by hand (`workflow_dispatch`), it sparse-checks out the updater and its imports from `bobberlington/AutoCroissant`
  main (no config.py: github_api reads it in a try/except), runs `python -m commands.tts_save` on the checkout's save
  with the two env overrides `AUTOCROISSANT_LOCAL_DIR=$GITHUB_WORKSPACE` (where it reads `TTS/decks.json` and
  `site/Card Edits.json`) and `AUTOCROISSANT_HISTORY_CLONE=~/TTSCardMaker-history.git` (its own bare blobless clone;
  global_config.py reads both with `environ.get`), and commits the save as AutoCroissant when it changed (message:
  the updater's `_commit_message`, which it writes to `<stem>.message.txt`; since 2026-10-22). The report is the job log.
  The Action's own save commits don't count as changed inputs.
- **By hand: `/update_tts_save`** (admin only). The bot downloads main's copy by blob sha into a temp dir, updates
  it and, when anything changed, commits it to main through the GitHub git data API with `GIT_TOKEN` (message like
  "TTS save: 812 links pinned, 3 cards added"; when the token can't write — 401/403/404 on the blob upload, or no
  token — git objects in the history clone pushed over SSH; a rate-limited API also goes over SSH for that run). The commit
  path is commands/git_commit.py since 2026-10-21, shared with the edit pipeline's card_bake. If main moved meanwhile, it
  redoes the update on the new save (3 tries). It never writes a checkout; the temp dir is deleted. One run at a time.
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
  `partial` for hand-picked decks; `add_new_since: <sha>` on a hand-picked rule adds every rule card the save at
  that commit lacked, renames followed (nothing is recorded: a card taken out by hand comes back unless the rule
  excludes it; a card moved to other stars moves to that stars' `add_new_since` deck, a fresh CustomDeck id if its key
  shows another image there; other hand-picked cards stay): the four "Creatures Developer Version" decks since 318001ea, 2026-10-22; a rule
  without `count` leaves its deck alone). The bot reads main's copy. Edit it in TTSCardMaker and push; the Action
  picks it up. A rule that matches no PNG, or a malformed deck entry, leaves its deck alone and is listed
  under Errors. Rules only cover PNGs: a .jpg/.jpeg card in a complete deck is never removed.
- **Standalone:** `python -m commands.tts_save <save.json>` (from the bot's root; what the Action runs) updates a
  local file in place and, when it changed, writes `<stem>.<time>.bak.json`, `<stem>.report.md` and
  `<stem>.message.txt` (the commit subject) beside it. `TTS/decks.json` and
  `site/Card Edits.json` come from the checkout at `LOCAL_DIR_LOC` (else raw main / nothing).
- Save commits do not rebuild the card website (`paths-ignore: TTS/**` in TTSCardMaker's site.yml).

### Font check (/fonts, added 2026-10-21; the automatic path is TTSCardMaker's font-check.yml Action, written the same day against the uncommitted tree)

commands/font_check.py (uncompiled; stock psd-tools + fontTools, no Discord or config imports; fontTools is in requirements.txt since
2026-10-21 and main.py imports the module lazily inside /fonts, so the bot starts without it) lists the PostScript names
a card's visible type layers ask for and which of them `MDW/Fonts` in the card repo does not supply. Hidden layers are
ignored; a font counts as present only when a face has exactly its PostScript name (no aliases or loose matches since
2026-10-23, owner). It reports and never rejects: a missing font is a message to a person, not a revert.

- **Automatic: TTSCardMaker's `.github/workflows/font-check.yml`** (ttscardmaker-companion §3). On a push to main that
  changes a PSD or `MDW/Fonts/`, it sparse-checks out the fonts folder (cached by its tree sha) and `commands/font_check.py`,
  `commands/github_api.py`, `global_config.py`, `requirements-cards.txt` from `bobberlington/AutoCroissant` (since 2026-10-24,
  branch work/py314: on Python 3.14 it installs requirements-cards.txt, the psd-tools fork's one pin, from the card workflows'
  wheel cache instead of stock psd-tools), reads the push's PSDs by blob sha
  (`--repo --before --after`, each blob's sha1 verified), writes the markdown to the job summary and, when something is
  missing, posts to the standards channel's webhook in the repo secret `DISCORD_STANDARDS_WEBHOOK` (owner, 2026-10-22;
  `DISCORD_FONT_WEBHOOK` is deleted; the URL is in no file) and then comments on the commit. The post pings role
  546427268751294475 once (the only role ping of any check's post, owner 2026-10-23), then one line per author and font: "<name> has pushed a card without
  adding the <font> font to MDW/Fonts, bite this user." with the cards and layers below. A dispatch with `all` scans every
  card PSD by blob (~813 downloads).
- **After the cutover** (`CARD_SYNC` on) `/fonts` reads the card data's `psd_details.json` instead (section 12): no PSD or font is
  downloaded, and `font_names.json` is no longer used.
- **By hand: `/fonts`** (anyone). Runs in a thread against the card repo's current tree through the API with the per-blob
  cache `font_names.json` (blob sha -> the fonts a PSD asks for / the faces a font file supplies; gitignored). The first run
  downloads the fonts (270 files, 215 MB) and at most `BATCH` = 200 PSDs (a 15 MB blob took 1.4 s on the dev Mac; the whole
  repo is ~813 PSDs, 15.6 GB, about 25 min), says "Warming the cache", reports partial results and how many PSDs remain;
  later runs download only new blobs, and blobs that left the tree leave the cache. Reply: `**font**: card (layers)` per
  missing font, the fonts folder count, errors. One run at a time ("A font check is already running").
- **Standalone:** `python -m commands.font_check --fonts <dir> --all <checkout>` (9.8 s wall over the 813 PSDs on 2026-10-21 in a
  process pool of half the cores from 8 PSDs up; 98 s sequential before, same JSON byte for byte: the six faces of
  ttscardmaker-companion §3 on 15 cards, 29 visible layers; on 2026-10-23, TTSCardMaker 92b5ae6b with Kung-Fu-Panda's rename:
812 PSDs, 274 files / 282 faces, nothing missing, 7 s), `--psd <files>`, `--json <out>`,
  `--discord <payload out>`; `--help` lists the --repo options the Action uses.

### Card check (added 2026-10-21; CLI only, no slash command; the automatic path is TTSCardMaker's card-check.yml Action, on TTSCardMaker main)

commands/card_check.py (uncompiled; imports psd_parser, card_models, card_edits, font_check, github_api and global_config, nothing with
Discord or config; `SPARSE_FILES` is that closure and the Action's sparse list) tells what the bot's parser and validator note about a card
PSD: the lines /update_stats prints for it (`NO STARS FOUND`, `MISSPELT TYPE`, `SEVERAL CREATURES`, `HP/DEF/ATK/SPD NOT FOUND`, `STATS TOO
HIGH`, `ABILITY TEXT NOT FOUND`, `UNKNOWN TYPE`; the parser's own, the first three and `NO ABILITY LAYER`, are also the website's Parser
notes, the validator's are printed only), after the card's `site/Card Edits.json` entry (`problems_ok`, `allow_high_stats`, `creatures` ...)
and `ABILITY_EXCLUSIONS`. The second push-guard rule (docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md section 5). It reports and never
rejects: exit 0 always; an unreadable PSD, a PSD under `MDW/` or `Markers/` (no card) or a failed API call (a 5xx after the retries, no
network, a truncated tree) is one error line.

- **Automatic: TTSCardMaker's `.github/workflows/card-check.yml`.** On a push to main that changes a PSD, `Types/` or `site/Card Edits.json`
  it sparse-checks out `SPARSE_FILES` from `bobberlington/AutoCroissant` (nothing of the card repo), installs AutoCroissant
  `requirements-cards.txt` (the psd-tools fork's one pin, fonttools, requests, pyoxipng) on Python 3.14 from the card workflows'
  wheel cache (since 2026-10-24, branch work/py314; stock psd-tools, fonttools and requests before), and reads the push's PSDs by blob sha (`--repo --before --after`, `font_check.pushed_files`); `Types/`, the icon manifest and
  the edits file come from the tree at the pushed commit (`font_check.tree_blobs`). The markdown goes to the job summary; when a card has
  notes it posts to the standards channel's webhook in `DISCORD_STANDARDS_WEBHOOK` (owner, 2026-10-23; unset: a warning) and comments
  on the commit. The message pings nobody (`allowed_mentions` `{"parse": []}`): one line per card, "<login> has pushed <card name> with: <note>; <note>.", and last
  "The bot's next /update_stats prints these too; the parser's own notes (no stars found, misspelt type, several creatures) also show on the
  website as Parser notes." (under 2000 characters; "... and N more cards" when cut). A push that changes only `Types/` or the edits file
  lists no PSD, so it reports nothing.
- **Dispatch:** Actions > Card check > Run workflow with `all` reads every card PSD of the tree (~813 blob downloads, 16 GB, 25-30 min) into
  the job summary only (no Discord message or comment on a dispatch); without `all`, the head commit's PSDs.
- **By hand:** `python -m commands.card_check --psd <files>` (each file in the checkout of its nearest ancestor holding `Types/`: its path
  from there classifies it and that checkout's `Types/` and edits file apply; a file outside any checkout is `UNKNOWN TYPE`),
  `--all <checkout>` (every card PSD, a process pool of half the cores from 8 PSDs up, like the font check; 0.1-0.4 s a PSD sequential,
  the 813 PSDs of the clone in 14.6 s wall on the Mac on 2026-10-21, 0 errors, exactly the seven cards below),
  `--repo OWNER/NAME --after <sha> [--before <sha>] [--whole-tree]`, `--json <out>`, `--discord <out>`. On the clone at e4e0a58 the notes
  are Bubble and God Tamer `NO STARS FOUND`, 20 Creature Types `MISSPELT TYPE: tornado`, Tainted Lazarus, Pix and Twin Emperors
  `SEVERAL CREATURES`, Computer Virus `HP/DEF/ATK/SPD NOT FOUND`; Anubisath Guardian, Qiraji Soldier, Silithid, Mini Doomer and Royal
  Eradicator Main Cannon (`allow_high_stats`) are clean (`tests/test_card_check.py` pins these). On TTSCardMaker 07589cc7
  (2026-10-22) only Computer Virus has notes (812 PSDs); the test reads the three fixed cards' old versions from the clone's history.

### PSD details (psd_details.json, added 2026-10-21; the data behind the website's card Details dropdown, plan item 23)

commands/psd_details.py (uncompiled; stock psd-tools; no Discord or config imports, like font_check) gathers one record per card PSD:
fonts of the visible type layers with their layers and a `supplied` flag (MDW/Fonts has the face; null when no index was at hand),
every type layer's text (200 chars), font, size, colour and visibility, effects per layer (`(off)` when disabled), smart objects,
canvas (size, mode, depth, ICC description), type-icon style (images / tokens / mixed / none), the stat-row encoding per stat
(template / two-row / pixel / other / none), `FX:` and overlay word layers, the title layer the parser reads (2026-10-23: a
root `Name`, a layer named after the card or the one named after its own text is no deviation) and the Ability layers with the ledger's cheap deviations, and a
compact layer tree (80 children per group, depth 8). Records are keyed by the PSD's git blob sha
in `psd_details.json` (repo root, committed; `{"version": 1, "cards": {sha: record}}`, one card per line so diffs stay per card).

- **In `/update_stats`:** after each re-parsed card's parse, `_details` reads the same open PSD and the record is stored
  under the blob sha (a failure is one console line `<path>: details not gathered (...)`, never a failed update); at the end of a
  run that read the tree the records are pruned to the tree's blobs and the file is saved like a pickle (tmp, fsync, replace)
  when anything was gathered or pruned. A 304 writes nothing. The `supplied` flags come from the /fonts cache `font_names.json`
  when it exists (the jetson after a /fonts), else stay null. Cards whose blob did not change keep their record; one that was
  never gathered gets it on its next re-parse (a `cards:` run, `force_update` or a PARSER_VERSION bump).
- **Backfill by hand:** `python -m commands.psd_details --clone ~/Desktop/TTSCardMaker [--out psd_details.json] [--fonts <dir>]`
  walks every card PSD of a clone (MDW/ and Markers/ excluded, like the traverser) in a process pool of half the cores and writes
  the file; `--fonts` defaults to the clone's MDW/Fonts. 813 cards in 8.9 s on the Mac on 2026-10-21, 0 errors, 2.7 MB, records
  913-16,699 B (median 3,738 B; Tainted Lazarus's 234 layers are the largest). One summary line, then the errors. The 48
  rulebook PSDs get records too (the traverser parses them; the website joins cards only, so those ~6 % are unused).
- **Travels with the pickles:** `/push` adds it to the PICKLE commit when the file exists (a missing file is skipped, not an
  error: until the backfill commit lands a checkout has none), `/pull` keeps the local copy on a conflict (`DETAILS_JSON` in
  global_config.py:11; `psd_details.json.tmp` gitignored). TTSCardMaker's site build fetches it from this repo's main like the
  pickles and serves one small file per card to the dropdown (ttscardmaker-companion).
- **`/fonts` reads it first:** `font_check.repo_fonts` seeds its per-blob cache with the records' fonts, so only blobs the file
  does not describe are downloaded.

### Card bake (/bake_cards, added 2026-10-21; the one live runner is TTSCardMaker's card-edits.yml Action, the jetson a manual fallback)

The edit pipeline (`commands/card_bake.py`, docs/EDIT_PIPELINE_DESIGN.md) bakes the website's PSD-kind edits in
`site/Card Edits.json` (stats, name, ability text) into the card PSDs and PNGs, one commit per run (one per card until
2026-10-22). `commands/card_runner.py`
(uncompiled; imports card_bake and psd_edit lazily inside the bake, so the bot starts without the fork's psd-tools) runs it
inside the bot on the jetson, by hand only; design section 8 "Jetson".

- **No automatic bake in the bot (decided by the user 2026-10-21): the Action is the one live runner.** The bot never queues a bake on
  its own; `/update_stats` ends with the emoji sync and the card-list refresh. The `notice_edits` hook (a dry-run `bake_cards`
  queued at the end of `/update_stats` when `site/Card Edits.json`'s blob sha changed) landed and was removed the same day.
  Since 2026-10-22 TTSCardMaker's `card-edits.yml` bakes live every 30 minutes when its gate (`card_bake.pending`: a PSD-kind
  edit without a refused verdict for exactly its values on the current PSD) finds work, bakes only those cards (`--pending`;
  a dispatch re-checks every card), at most 40 a run, and posts to Discord (`DISCORD_CHANGELOG_WEBHOOK`) when a streak of
  failed scheduled runs starts (once, not every tick); nothing in this repo triggers it.
- **`--local <clone>`** (2026-10-22, for card-sync.yml, on TTSCardMaker main and dispatch only until its cutover,
  docs/Campaigns/DATA_BUILD_ACTION.md §6.3 stage 1): the tree and blobs come from that clone (blobless is enough; a
  missing blob by REST, never a lazy fetch) and the run's commit B is made there on `--head` (default its HEAD) and pushed
  by nobody; a checkout at `--head` fast-forwards to B. Same gates.
- **By hand: `/bake_cards [card] [dry_run]`** (admin only; `dry_run` defaults to True, `card` restricts the run to one card
  matched like Card Authors.txt). Same code, same Lock. Not dry: commits through `git_commit.commit_files` with
  `GIT_TOKEN` (the API), SSH through the history clone when the token can't write (the unit's `SSH_AUTH_SOCK`).
- **One at a time: `card_runner.bake_lock`.** `update_stats` holds it for its whole run and the bake takes it around
  `card_bake.run`; queued commands otherwise run concurrently (one thread each). Whoever finds it held says "Waiting for the
  running card bake or update to finish..." and waits. Two `/update_stats` therefore no longer overlap either.
- **Where the report goes:** "Baking the website's card edits (dry run)...", then the report's head lines and sections as
  messages (the `<details>` tracebacks stripped) with the full markdown attached as `card_bake_report.md`, then the commit
  links when any. A failure outside a card's steps is one line, "Card bake failed: <Error>: <why>" (traceback on stdout);
  the traverser never crashes on it.
- **The jetson today (2026-10-23):** the fork at tag `cards-20261023` (ecc4b773) is in `~/venvs/autocroissant`, so the
  text engine is there; "Card bake refused: this Python has no text engine ..." on the jetson now means a broken fork
  install (reinstall the tag, section 3). Fonts: `card_bake._fonts_dir()` reads `PSD_TOOLS_FONT_DIRS`, else the checkout's
  `MDW/Fonts`, else `fonts_cache/` brought to the tree's MDW/Fonts (66cfe9e, 2026-10-21): the jetson has no
  `~/Desktop/TTSCardMaker` and uses the cache. The work dir is a fresh `mkdtemp(prefix="card-bake_")` per run.
- **What refuses since 2026-10-23 (owner):** a pixel miss no longer refuses. Only `psd_edit.PARITY_FLOOR` (more than 25%
  of an edited box past 24 levels), a blank render or a read-back that differs from the edit refuses; a smaller miss is
  baked and written in the bake report and the commit note. The bake also takes the layers it used to refuse
  (refused-layers): a printed-title layer named after the card (no `Name` layer, or outside the Text group) or, failing
  those, after its own text (title-own-text: Papyrus, Sans; a bake renames it after the new title), and a single
  visible Ability layer under another name. Text bakes are line-spliced (fork feat/line-splice): only the edited lines
  are re-set, so the rest keeps Photoshop's own ink. An edit takes the colour and style of what it replaces, part by part.
  A stat bar with a digit mask wholly off the canvas (Green Mage's four bars, the only ones) refuses `unsupported-card`
  as a final verdict; it was an `error` that failed the run until work/bake-verdicts (2026-10-23).
- **Ability FX layers (BUG 2, branch work/fx-rebuild, not landed):** an ability bake makes the card's `Ability FX: ...`
  layers again as Effect Layers.jsx's Update all does (the gradient cards' badge layers included): each cut in place over
  its words' new places, its own style and place kept (q6); a status layer stays on its status's own words (a numbered
  one renumbered, Standstill's `Frozen In Time#2` -> `#3`; "Frozen" made "Frozen In Time" drops the Frozen layer and
  makes a Frozen In Time one in the chart's style). The report says when a layer is deleted (its words left the text,
  or they are no longer its status: "'Frozen In Time' is no Frozen"), renamed (a phrase dropped or renumbered) or cut
  again clean. Since 2026-10-23 (owner) an FX layer whose style is no chart status's (Fire Staff's Smoldering stroke,
  Mr Krabs' `2T`) is rebuilt in its own style too, staying on its own words; a typed word joining such a layer is told
  ("... also covers the typed 'Smoldering', in its own style"; "moved to" when its old words left) and takes the
  phrase's colour (a stroke alone shows the fill). Where a layer can't follow its words (on Monkey D Goofy a whole-word
  "1" anywhere in the ability, an own-style phrase edited, deleted and typed elsewhere or glued into a longer word,
  a phrase the ability lacks) the layer is deleted, told, its words printing plain (owner 2026-10-24; they refused
  before); a "1" typed before its "1" renames the layer `1#2`.
  The commit line and the website's verdict reason carry the FX notes too.
- **Typed badges and icons (owner 2026-10-23, same branch):** a timing tag ("(1CD)", any case) or a token the icon
  font draws ("[fire]", "[fire:no]", "[Ice:Sticker]") typed anywhere in an ability is set in the stoogedings font at
  the card's own badge or token style (on a card with none, at the text's style, the font added to the PSD); on a
  gradient badge card it joins the badge layer, and so does a badge edited by deleting letters ("(1GCD)" made "(1CD)").
  A typed "★" is set in the icon font (the gold star). A typed badge or token is unstretched (`unstretch`: its
  HorizontalScale set so it prints as the font draws it under the layer's scales, as designers set theirs; Monster
  Meat's [fire] printed 2.6x wide). Told, not refused (owner 2026-10-24): a badge under a tinted Ability (any colour,
  gradient or pattern overlay, a Pin Light or other non-Normal blend) with no badge FX layer (Cat, Ice Staff, the Pin
  Light Isaac items without one), a `[type]` token on a card whose type icons are image layers in the text (Nebulous
  Precipice), letters left in the icon font it draws no badge of ("(abc)", "(GCD)"), a token the website shows as an
  icon but the font prints as letters ("[fire:]"), a typed character its font has no glyph for, a see-through icon
  under a badge layer, a timing badge typed in a title (printed as plain words, as the website shows it). An edit
  keeps the card's own icon spellings ("[dark:sticker]" where
  the website shows "[dark]"). Typed `:glow`/`:darkglow` tokens bake on the `cards-20261024b` pin (its halo as
  Photoshop draws it; on a stale install of an older fork the edit is a runner error, no verdict, retried once the
  fork is reinstalled). Also told: a title or Ability whose saved pixels are in another font than the layer names
  (`drawn_in_font`: Dull Knife, Shiny Knife, Dark Tentacle and Snow Angel's Omori titles). Refused: a text layer with
  tab stops (`unsupported-card`: Michael's Blessing; the engine lays no tabs). A
  deleted line or word is cut whole, so the text after it keeps its own colours and badge size; an Ability's new line
  printed past the old text onto the design below it bakes with a note ("... prints past the old text onto the design
  below it (N px, 'layer'); check the card": `ink_beyond`, 18 of 503 cards given a line).
- **Content flags (owner 2026-10-24 ~18:30, same branch):** "In general I do not want to refuse bakes": the editor
  previewed the edit, so a content problem bakes best-effort and is told as `<code>: <field>: <why>` in the bake report,
  the commit line and the baked verdict's reason (the preview's notes too): text that won't fit (`overflow`: off the
  canvas, onto the stars or icons, more title lines, under a layer above or over a stat bar, past the design below or
  a texture clipped to it: `ink_beyond`, `ink_past_clip`), typed badges and tokens, FX layers that can't follow
  (deleted, told), overlay word layers the chart has no style for (hidden, told), font stand-ins (`font`: a face no
  file holds is drawn in the fork's stand-in, `font_map`), new validator problems (`validator: new problems: ...`), and
  a stale Txt2 story (synced by `txt2_repair.sync_stories`, told). Still refused: the waits (`recent`, `head-moved`,
  `edit-changed`, `blob`, `capped`), impossible cases (no PSD, a numbered PNG copy, a creatures list setting the
  stats, a stat above 20), the bake's own checks (`bytes`, `region`, `size`, `shape`, `reparse` but new problems,
  `idempotence`, `parity` past the 25% floor) and what the bake can't edit yet (`unsupported-card`: split or several
  titles, non-template stat rows, tab stops, folder-set stars and the other stars/types layouts, FX layers that are no
  pixel layers, two FX layers one name; `stat>10` on one bar row).
- **card-edits.yml** was paused from 2026-10-22 ~21:5x (split titles would have baked wrongly, and B2 was rewriting the
  same cards) and re-enabled on 2026-10-23 ~10:36, pinned to `cards-20261023` (dry run 39996528699 dispatched then).
- **Previews (2026-10-23; TTSCardMaker's `card-preview.yml`, design section 16):** the website dispatches one card's edit
  (inputs `card`, `edit` JSON, `request` id) and the Action's read-only first job runs `python -m commands.card_preview`:
  `card_bake.bake_card` on that card alone, with the bake's gates, fonts, chart, vocabulary and fork pin, never committing
  to main. Its second job, the only one that writes, runs `python3 -m commands.card_preview_branch` (standard library, no
  install) to put `result.json` (state baked / redundant / refused / error, the bake's reason, notes, base blob shas, the
  edit as sent), `preview.webp` (720 px) and `before.webp` (the card before, for the Diff) under `previews/<request>/` on the orphan branch `previews`, one commit per run,
  rewritten as one fresh commit past 200 previews or 7 days. No Discord, no secret beyond GITHUB_TOKEN, nothing for the bot
  to do; previews are public like the repo. A red run is a bad request (nothing published), a bad edit (published as an
  input error) or a runner error, a failed install too (published as "try again"). By hand on the Mac, read-only on the
  clone: `python -m commands.card_preview --card "Mini Doomer" --edit edit.json --out /some/dir --clone
  ~/Desktop/TTSCardMaker` (stat 6-14 s, ability 26 s here; preview.png in the out dir is the full-size render).
- **Design and status:** EDIT_PIPELINE_DESIGN.md sections 8 (runner), 13 (S7) and 16 (previews); the PSD-edit ledger is
  docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md.

### Card guards (added 2026-10-22; TTSCardMaker's card-guards.yml Action; landed 2026-10-23 on AutoCroissant main 2ada357 and TTSCardMaker a7d1fe56, off while `CARD_GUARDS` is unset)

The push guards (`commands/card_guards.py` with `png_export.py` and `card_standards.py`, all uncompiled; design and
failure table: docs/Campaigns/PUSH_GUARDS.md sections 8 and 10). On a push changing a card PSD or PNG: a PSD pushed
without its PNG gets the PNG the bake would make (owner's ruling 2026-10-22: the fork pinned at `cards-20261023`
renders the changed layers in the card's ICC export mode and the old PNG takes the change; the PSD's stored composite is
never written, only a cross-check that can refuse; a refusal is told as "PNG not made: ...; export it in Photoshop";
nothing when the PNG already shows the PSD, pushed before it, a new card's too), the pushed PSDs are checked against the
card standards (fonts, R7, until 2026-10-24: since then font-check.yml's alone, branch work/py314) (the owner's one easy fix, R9's Txt2 sync; R11's exact-name rename is built but told
until the owner says yes, `card_standards.AUTO_FIX`; R3 and R4 keep only the real issues, several visible Ability layers, a hidden one
with words the parser reads nowhere and a hidden or rasterized Name layer (kept unattended, for the owner to confirm): their
template layout was dropped by the owner on 2026-10-23), one commit comment, one Discord post to the standards channel
(`DISCORD_STANDARDS_WEBHOOK`, no ping; card-check.yml posts there too, no ping, and font-check.yml, with its role ping) and at most one bot commit,
both after a 30-minute quiet window (since 2026-10-24, branch work/py314: the post job waits in the TTSCardMaker
environment `card-guards-quiet`'s 30-minute timer, created by the owner after both branches land; no runner sleeps; a PSD saved without
Maximize Compatibility gets its PNG anyway, told "no stored image to cross-check"; docs/Campaigns/PUSH_GUARDS.md section 20). Since the owner's ruling of 2026-10-23 the bot's drawing refuses only past 25% bad;
a smaller miss is made and told in the comment and the bot commit ("made with a pixel difference ..."). On the 34
replayed commits the PNG is made on 29 and refused on 5 (PUSH_GUARDS.md sections 17 and 18). The comment and the post
carry each card's semantic PSD diff (`psd_diff`, 0a34292): "{login} pushed {card}{changes} with these issues: ...", the
changes cut to 120 characters on Discord and 600 in the comment. A new PSD pushed beside its PNG counts as current only
within 8 levels (`png_export.BESIDE_CUT`, owner); with no stored image, up to 300 px past it the PNG is kept and told
(`png_export.KEPT`/`KEPT_PX`, default 2026-10-24, owner to confirm), past that redrawn.

- **The switch: repo variable `CARD_GUARDS`** (TTSCardMaker Settings > Variables, not a secret; any letter case). Unset or
  `off`: push runs are skipped. `report`: check, comment and post (after the window), never commit: the PNG the bot
  would make is told ("PNG not exported with the PSD: export it in Photoshop") and the comment says it "would be
  re-exported (CARD_GUARDS=on)". `on`: also land the PNGs and fixes. Rollout (owner, 2026-10-22): no variable, then dry dispatches, then `report` for about 3 days, then `on`
  (owner's call; the items to confirm first are at the top of PUSH_GUARDS.md section 15).
- **Dispatch (Actions > Card guards > Run workflow):** `dry_run` defaults to true (job summary and the `card-guards`
  artifact only: plan.json, report.md, discord.json, the planned PNGs and fixed PSDs, crops); `sha` checks one commit
  against its first parent; `cards` limits it; `all` audits every card PSD (about 40 minutes, always dry). `dry_run` off
  comments, posts and lands at once on the blob check alone (no quiet window), so it refuses ("still in its quiet
  window") while a push run of that commit, or of a later one, is unfinished: dispatch again once it has finished.
  `CARD_GUARDS` does not bind a dispatch: unset or `report`, a dispatch with `dry_run` off still comments, posts and
  lands, so dispatch dry while testing in the `report` phase. An author with no GitHub account is named by the git author
  name there, never the dispatcher.
- **Locally, read-only:** `python -m commands.card_guards check --clone <card repo clone> --after <sha> [--before <sha>]
  --out <dir>` (any clone with the history; blobs it lacks come by API), then `land --plan <dir>/plan.json --dry-run`
  or `--local <clone>` (commits there, pushes nothing). A bare `land` commits to TTSCardMaker with config.py's token, so it
  refuses a plan not checked with `--mode on` (exit 2). Needs the fork's psd-tools (`~/.venvs/ac-py314-fork`).
- **Re-runs:** "Re-run jobs" edits the same commit comment and never posts twice (the told mark on any guards comment;
  the lookup is tried 3 times, and a re-run that still cannot read the comments counts the push as told: neither its
  failure alert nor its post goes out, and its error says the push may be untold, so re-run again once GitHub answers).
  A re-run that fails in Check looks the mark up in the alert step, so a push already told gets no "was not guarded".
  The quiet window counts from the push, so a late re-run does not wait again. After a run that landed, the re-run finds
  its blobs on the branch ("already on the branch"), writes nothing and starts the site build and the TTS save again
  (the first attempt's may not have started). A re-run (or a dispatch) that cannot read the branch tells no write as
  "not made" (an earlier run may have landed it); its comment says this run committed nothing.
- **Failures:** a push run that fails without its post going out (not checked, the commit failed unposted, a step broke
  or timed out) sends one message to the standards channel with the run link: "was not guarded", or "was checked, but its
  run failed" once Check passed. Once per streak: a run that sent it, or stayed quiet inside a told streak, leaves the
  `card-guards-failure-told` artifact, and the next failing run is quiet only when the last finished push run left it
  and is not green. A red run that posted its own lines (a card error, a failed comment) leaves none, so the next
  failure is told. A card the guards could not finish is told in the post and turns the run red ("Fail on card
  errors"); re-check it with a dispatch (`sha`, `cards`). When the site build or the TTS save does not start after the
  bot's commit, the comment and the post say so and the run is red (site.yml catches up within 6 hours, the TTS save the
  next day; a re-run starts them again). Cancelling a run by hand (before the job's 150-minute timeout; on work/py314: no cancelled job ran to its timeout, check 100 or post 75 minutes) sends no message.
- **Loops:** GITHUB_TOKEN pushes start no run; the job skips the AutoCroissant and github-actions actors, and inside a range
  the files only bot commits changed.

### Discord archive and card matching (added 2026-10-22; CLI only, run by hand)

commands/discord_archive.py (the miner) and commands/discord_match.py (the matcher) are both uncompiled and not in setup.py.
Together they find the earliest Discord post of each card, plus its posters, the names people used for it and its revisions.

- **Privacy rule:** the mined data is other people's words and images. It stays under `~/Documents/Tools/discord-archive/`
  (the miner's `--out` must be that folder or one inside it, and the matcher refuses a root inside a git repo). Nothing mined goes into any repo, test
  fixtures included (this repo is public), so tests use made-up guilds, ids, names and images only.
- **Miner:** `python -m commands.discord_archive --guild <id> --all --config <path to config.py>` (`--list` for
  channels only, `--channels <id>...` to go first, `--retry-downloads` for failed images). It sends GET requests only
  through REST v10, never the gateway or a write, so it can run beside the bot on the same token. It is resumable
  (cursors/) and stores images by sha256. The token is read from config.py with ast and is never printed. Long runs:
  start it with `nohup` and log under the archive folder. Outputs: messages/, attachments/, attachments.jsonl,
  channels.json, summary.json, runs.jsonl.
- **Matcher:** `python -m commands.discord_match score` (perceptual hashes, file names, chat text and macOS Vision OCR
  of card titles; `--no-ocr` without pyobjc; card caches in match/), then `build`. `build` writes matches.json,
  REPORT.md, proposal/ and ambiguous.json/.html, whose entries A1, A2... wait for the owner's ruling. The owner rules
  with `python3 -I apply_rulings.py "A12 = <card>, A13 = none"` in the archive folder, which also rebuilds.
  `selftest` measures accuracy on degraded copies of the cards. The matcher never talks to Discord, and bot and
  webhook posts never date a card.

## 8. Artifact conventions — where files land

All paths relative to the CWD the bot was started from (= repo root, always).

| Artifact | Produced by | Location | Committed? |
|---|---|---|---|
| `stats.pkl`, `old_stats.pkl`, `aliases.pkl` | /update_stats, /alias, metadata commands | repo root | YES — the database, synced via /push |
| `reminder.json` | /set_reminder, /reminders, every firing (schema_version 2, `ReminderStore` in analytics.py) | repo root | NO (gitignored) — per-machine. Edit by hand only with the bot stopped |
| `reminder.json.tmp` | every reminder save (written, fsynced, then renamed over reminder.json) | repo root, normally gone within milliseconds | NO (gitignored). If a hard kill leaves one behind, delete it |
| `reminder.pkl.bak` | the one-time conversion (2026-10-24) | repo root | NO (gitignored). The old store; keep it a while, then delete |
| `repo_history.json` | startup and every /update_stats (commands/repo_history.py; a failed refresh retries on the next one) | repo root; its bare clone at `~/Documents/TTSCardMaker-history.git` | NO (gitignored) — per-machine, rebuilt from the clone if deleted |
| `stats.csv` / `stats.txt` | /export_cards (`as_csv` picks which) | repo root | NO (gitignored via `*.csv`/`*.txt`) |
| `rules.txt` | /export_rulebook | repo root | NO (gitignored) |
| Music downloads | /play | `music/`, named `%(title)s-%(id)s.%(ext)s` (restrictfilenames on); playlists get a `music/<playlist title>/` subfolder (music_player.py:52-53, 250-251) | NO (`music/` gitignored) |
| /update_stats PSD temp files | /update_stats (when a PSD downloads whole) | OS temp dir via `mkstemp` (`_download`, card_data.py:164-187; the build's `GitBuild._download` :1328-1345); deleted after each parse (`_release`, card_data.py:733-738; before 2026-10-19 they piled up, archaeology Entry 10.2). Since 2026-10-22 startup deletes the ones a cut-off run left (`sweep_temp_psds`: `tmp*.psd` older than 1 h, under bake_lock; c0fc117) | n/a |
| TTS save working copy | /update_tts_save | OS temp dir (`TemporaryDirectory`), deleted after each run; the only output is the commit to the card repo (the automatic path runs in TTSCardMaker's Action, not here) | n/a |
| Card bake work dir | /bake_cards (commands/card_runner.py -> card_bake.run) | OS temp dir via `mkdtemp(prefix="card-bake_")` per run (`$RUNNER_TEMP/card-bake` on the Action), kept: edited PSD, merged PNG, before/after/diff crops, `report.md`, the rewritten edits file; the report is also posted in Discord as `card_bake_report.md` | n/a (commits go to the card repo) |
| Type emojis | /sync_emojis, startup sync, /update_stats on a changed tree | Discord-side application emojis owned by the logged-in bot identity; `type_emojis.json` (repo root) records each one's icon blob, per bot identity | NO (gitignored) — per-machine; if deleted, startup adopts the existing emojis as current |
| `psd_details.json` | /update_stats (commands/psd_details.py; `python -m commands.psd_details --clone` backfills) | repo root (+ `psd_details.json.tmp` while saving): PSD blob sha -> the card's PSD facts for the website's Details dropdown; pruned to the tree on every update that read it | YES — with the pickles in every PICKLE commit (/push); kept local on a /pull conflict |
| `card_data/` | after the cutover (`CARD_SYNC` on): startup and every /update_stats (commands/card_sync.py, `card_db.sync`) | repo root: the card repo's `data` branch's `db/` (`stats.json`, `old_stats.json`, `psd_details.json`, `manifest.json`) plus `sync.json` (the data commit held and its ETag); each file replaced atomically, the manifest and `sync.json` last | NO (gitignored) — the Card sync Action's output; delete it and the next sync fetches it again (about 3.5 MB); `python -m commands.card_db export-pickles card_data <dir>` turns it into pickles |
| `fonts_cache/` | the bake, psd_details and /fonts where there is no card repo checkout (`FONTS_CACHE`; font_check `refresh_cache`) | repo root: a copy of the card repo's MDW/Fonts, refreshed by tree sha | NO (gitignored) — per-machine; delete it and the next reader fetches it again |
| `font_names.json` | /fonts (commands/font_check.py) | repo root (+ `font_names.json.tmp` while saving): blob sha -> the fonts a card PSD asks for or a font file supplies; pruned to the current tree on every run | NO (gitignored) — per-machine; if deleted, the next /fonts downloads every PSD again in batches |
| `.ssh/` deploy key | operator-provisioned | repo root | NO (gitignored) — per-machine, SECRET |
| Offline test suite | developers (`python -m pytest -q tests`, before a push) | `tests/` (goldens in `tests/goldens/`, re-pinned only by `tests/make_goldens.py`) | YES — code; never writes the pickles |
| 2026-10-20 optimization-pass audits, bench and regression scripts | the optimization pass (read-only audits; sandboxes in /tmp) | reports and `bench/` on (`docs/Finished/optimization-pass-2026-10-20/`: profile.md, network-robustness.md, network-speed-results.md, code-health.md, docs.md, refs_table.md); scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` | YES — scripts on main (same commit as this row), reports on the archive branch only |

## 9. Reminders operations

Rewritten 2026-10-19; panel, on/off and `reminder.json` 2026-10-24 (commands/analytics.py, the panel UI in
commands/reminder_panel.py). Times are Pacific — `TIMEZONE = ZoneInfo("America/Los_Angeles")` (global_config.py:20), so
they follow daylight saving. `check_reminder` is queued once per second by its own loop, `process_reminders`
(main.py:920). History of what the rewrite fixed: autocroissant-failure-archaeology Entries 10.1 and 10.6-10.8. Flow
and invariants: autocroissant-architecture-contract. OJWikibot has a near-copy of this module (same parser, panel and
store shape; its own adapter block at the top of its analytics.py; no drift check).

### /reminders (the panel, anyone)

An ephemeral panel (only the invoker sees and uses it) on the first reminder in this channel: an embed card (message,
command, next run in Pacific time, repeat, channel, creator, On/Off/⚠ problem), `◀ ↻ ▶`, a jump list (over 25
reminders: 23 per page plus earlier/later), and Reschedule (a form: when, repeat; `when` also takes a delay like
`30m`), Turn off/on, Delete (with a confirm) and New (a form: message, command, when, offset, repeat; created in this
channel as you). Reschedule, Turn off/on and Delete need an admin or the reminder's creator (no creator: admins only);
New needs an admin. An off reminder keeps its schedule and never fires; on again, a repeating one skips the runs it
missed and a one-shot whose time passed must be rescheduled first. Every click re-reads the list ("That reminder is
gone." when it fired or was deleted elsewhere). It closes after 10 idle minutes; after a restart open panels fail
("This interaction failed"): run `/reminders` again. A failed conversion shows in its footer.

### /set_reminder (admin only, main.py:238; the gate at :252; the panel's New does the same)

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

**Retired options are dropped, not refused.** `RETIRED_OPTIONS` (analytics.py:44) lists options a command
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
  with carries a stand-in user with that id (`_ReminderUser`, analytics.py:95). The slash wrapper's own
  `lacks_admin` therefore runs again at fire time: if the creator has left `ADMINS`, an admin command replies
  "You do not have permission." in the channel instead of running.
- **Legacy reminders** (saved before 2026-10-19, so no creator) still run as the bot: `LEGACY_RUNS_AS_BOT = True`
  (analytics.py:37). The bot's own id is in `ADMINS` on the dev Mac, so those pass every admin check.
  `/reminders` labels them "unknown (runs as the bot)". To bring one under the new
  rule, re-create it and then remove the old one.

### When things go wrong

| Situation | What happens |
|---|---|
| The command crashes | "⚠ Reminder `<id>`: `/<name>` failed (<Error>: <text>)" in the channel; the reminder still repeats |
| The stored command no longer resolves (command renamed, option removed) | The message (if any) still posts, then "⚠ Reminder `<id>` skipped its command: <reason>. Re-create it, then `/remove_reminder reminder_id:<id>`." The reminder still repeats or completes normally |
| Channel deleted or no longer visible | Skipped, with a terminal log line. A repeating reminder moves to its next slot; a one-shot is retried every hour (`CHANNEL_RETRY`) until the channel is back or you remove it |
| Bot was offline, or stalled, past one or more runs | A one-shot fires right after startup. A repeating reminder fires once, then jumps straight to its next future slot. Missed runs are not replayed |
| A reminder crashes the scheduler itself, or its stored data is unreadable | Only that reminder stops (its `problem`, e.g. "crashed the scheduler"); `/reminders` shows "⚠ <problem>". Reschedule it (clears the problem) or delete it |
| reminder.json unreadable at startup | Reminders are OFF for the session and the file is left untouched. Every reminder command replies "✗ Reminders are off: ...". Fix it, or move it aside and restart for a fresh schedule |
| Two scheduler passes overlap | The second is skipped (single-flight lock), so nothing double-fires |

### /remove_reminder, persistence

- **/remove_reminder** `reminder_id`: the shortcut of the panel's Delete (admins, or the reminder's creator; anyone
  else gets "✗ Only an admin or the reminder's creator can change it."). Case and backticks are ignored.
- **Persistence**: `reminder.json` (`ReminderStore`, schema_version 2) is rewritten atomically on every change
  (temp file, fsync, rename, fsync of the folder). Per guild a list of reminders in creation order: `id`,
  `channel_id`, `msg`, `when` (ISO with that instant's Pacific offset; read back into Pacific, so a daily 05:00 stays
  05:00 across daylight saving), `frequency` (seconds or null), `command`, `enabled`, `creator_id`, `creator_name`,
  `v` (`REMINDER_VERSION`), `problem`; ids are text. A save is canonical (a no-op save changes nothing). An entry
  with an older `v` is migrated on load: its command is rewritten to canonical form keeping what it actually did,
  and retired options (`RETIRED_OPTIONS`) are stripped, so `/update_stats use_local_repo:False` becomes
  `/update_stats`.
- **The conversion (once, 2026-10-24)**: the first start finds no `reminder.json`, loads `reminder.pkl` as before,
  writes `reminder.json`, reads it back and compares every field, and only then renames the pickle
  `reminder.pkl.bak`. Ids, creators, schedules and commands carry over (the cutover's `/remove_reminder` of
  d2664752 and 054b5933 still works). On any failure the half-made JSON is removed and the session keeps running
  and saving from the pickle (the panel footer and /set_reminder replies say so). The pickle path goes once every
  host has converted.
  The fire path drops them too, so a reminder never fails on them. Watch for "Reminder <id>: migrated command
  ... (dropped retired ...)" lines. Per-machine file: reminders DO NOT
  follow a machine handoff (section 6 step 6).
- **Not yet observed live** as of 2026-10-19 (docs/BUGS_AND_IMPROVEMENTS.md V2): watch the first startup's migration
  lines and the first firing.

## 10. Guild admin quick table

| Command | Gate | Notes |
|---|---|---|
| `/purge [user] [num] [bulk]` | admin (since 2026-10-22) | Deletes messages in the current channel; defaults to the BOT's own messages, num 100 (`-1` = up to 1,000,000); `bulk:True` is faster but needs the bot to have Manage Messages (main.py:715-730; the gate at :725). A reminder cannot run it (`user` has no default) |
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

## 12. After the cutover: the bot reads Card sync's data (`CARD_SYNC`)

Built 2026-10-22 (design docs/Campaigns/DATA_BUILD_ACTION.md §11 and §14 step 8); merged to main on 2026-10-23 (7b99438)
and deployed at 10:36 with `CARD_SYNC = False`. **Off until the cutover**: every section above is the bot with it off, and
with it off the bot is byte-identical to the one before this code (the suite, the rehearsal harness and a surface diff of
every slash text, /help and /website showed no difference).

**Status (2026-10-23):** TTSCardMaker main a7d1fe56 turned on card-sync.yml's schedule (cron `4,34 * * * *`, `MAIN_COMMIT` and
`POSTS` off): that is the shadow start (§14 step 7). At 11:18 no scheduled run had fired yet (GitHub's scheduler runs hours
late for this repo) and the card repo had no `data` branch. The shadow lasts 3 days from 2026-10-23 10:36 or 10 card
pushes, whichever is later; its daily check is diff_stats between the `data` tip and the jetson's last PICKLE
(autocroissant-diagnostics-and-tooling, the shadow's daily check).

**Seed precondition (history recovery):** the `data` branch's seed PICKLE must be one the jetson pushed after its first
`/update_stats` with the recovery (the "History: N versions recovered from the bot's PICKLE commits ..." run, section 7).
The runners' checkouts are shallow, so the seed pass there never recovers (DATA_BUILD_ACTION.md §14 step 7). The first
scheduled run creates `data` from AutoCroissant's latest PICKLE at that moment, which may be a7c310e (before the recovery).
**Check after the next 05:00 run and its 05:30 /push:** if `data` was seeded before that PICKLE, re-seed it with a dispatch
`seed=<the recovery PICKLE>` (`dry_run=false`).

**The switch** is one constant, `CARD_SYNC` in global_config.py, `False` until the cutover commit sets it `True`. Nothing to
remember at runtime and no flag file: not the data branch's presence either, since the shadow (step 7) creates `data` days
before the cutover. Code reads `global_config.CARD_SYNC` at call time; the slash command texts are picked at import, so the
deploy's restart applies them. Rollback: revert that commit (DATA_BUILD_ACTION §13), deploy.

**The cutover sitting, the bot's part** (after §14 step 8's reconcile: clean `git status` on the jetson, its last /push on
origin):
1. `ACTIONS_TOKEN` in the jetson's config.py (fine-grained, TTSCardMaker only, Actions read and write; never print it).
   **Done** 2026-10-22 ~12:25 (the owner).
2. Discord, as the owner: `/remove_reminder reminder_id:054b5933` (05:30 /push) and `/remove_reminder reminder_id:d2664752`
   (05:00 /update_stats); then `/set_reminder when:<a time at :12 or :42> frequency:30m command:/update_stats`, a few minutes
   after each Card sync (cron `4,34 * * * *`). Keep 134e3b24.
3. The cutover commit (`CARD_SYNC = True`), then the deploy: pull, `build_ext`, restart (memory `jetson-server`).
4. Check: the log shows "Card data loaded ..." at startup; `/update_stats` says "Card data is already current: built … from
   TTSCardMaker@…"; `/query` the three goldens; `/fonts`.

**What each command does with it on** (commands/card_sync.py, uncompiled):

| Command | With `CARD_SYNC` on |
|---|---|
| startup (`init_psd`) | `card_data/` loaded (no network; a copy a sync left half-written loads with no commit, so the sync reloads it whole), then one sync; with neither, the root pickles (frozen since the cutover; gone at step 10). The startup sync queues nothing: on_ready queues the card list, image links and emojis itself |
| `/update_stats` | `card_db.sync`: one conditional `GET repos/MichaelJSr/TTSCardMaker/commits/data` (sha media type, `If-None-Match` from `card_data/sync.json`; a 304 is free); on a new commit its `db/manifest.json` and only the data files whose sha256 differ, raw at that commit, each checked against the manifest and parsed before any is written. Then the database is reloaded from JSON, never a pickle, and the card list (a 304 when the tree is unchanged), query tables, type emojis and image links are refreshed |
| `/update_stats` reply | "Card data built 12 min ago from TTSCardMaker@1a2b3c4d (run: <…>): 3 changed, 1 new, 1 archived; reloaded." (counted against the database it held, duplicates left out), "Card data is already current: built …", or "Card data loaded: built …" (it held the pickles). Then, when they apply: "Re-parse in progress: N cards still to re-parse"; "Could not read N PSDs (…)"; "The card repo has card changes since: the next card sync (every 30 minutes, at :04 and :34) picks them up" or "the card sync running now picks them up (<run>)" (one compare request); "The last card sync failed: <run>; the bot keeps the last good card data" (dry runs and cancelled runs left out). `output_problematic_cards` shows the new build's `manifest.problems` (only when new data loaded) |
| `/update_stats` from a reminder | quiet: the reply goes to the console only (and no status requests), unless the sync failed or new data has problem cards |
| `force_update` / `cards:` | admin only (a non-admin gets "You do not have permission.", a non-admin's reminder too). Dispatches Card sync with `ACTIONS_TOKEN`: `dry_run=false` and `full=true`, or `cards=<names>`; the run comes from the dispatch reply (`return_run_details`) or else is the newest dispatched run not listed before (a minute at most). Looked at every 15 s for 30 minutes at most, the state ("waiting for its turn", "running: Build") edited into the reply for its first 14 minutes (Discord's interaction lasts 15; a reminder's run gets none). Then the sync. A full re-parse continues over the next scheduled runs (the "Re-parse in progress" line) |
| a cancelled run | the concurrency group keeps one run pending, so a newer one replaced it before it started: dispatched once more when that cancels nothing but a scheduled run; said ("replaced by a newer request (<run>) … ask again once it is done") when another dispatch is pending, and "was cancelled." when it was seen running or nothing newer exists (cancelled by hand). A second cancel is said. A green run whose data Card sync dropped (main moved meanwhile) is followed by its own retry ("The card repo changed during run … started again: <run>"; a full re-parse is not carried: ask again) |
| no `ACTIONS_TOKEN` | the reply links the Action and gives `gh workflow run card-sync.yml -R MichaelJSr/TTSCardMaker -f full=true -f dry_run=false` (or `-f cards="…"`), then syncs. A refused token (401, 403, 404, at the run list before the dispatch too) is said the same way, a rate limit as one, and a dispatch with no answer as "the run may have started: look on <Action> before asking again"; the token is never printed |
| `/push` | replies "Nothing to push: the card data is built by the Card sync Action …" with the Action's link; no git. Removed at step 10 |
| `/update` | pull, then restart; no pickle push |
| `/pull` | a conflict takes the remote's file for every path ("AUTO-RESOLVE: Merge conflict resolved (accepted remote)"); no pickle rescue. Normally a fast-forward: the bot commits nothing any more |
| `/fonts` | from `card_data/psd_details.json`: each current card PSD's fonts with `supplied: false` (the build judged them against MDW/Fonts); a PSD without a record, or with fonts not judged, is an error line. First line "From the card data (built …): N card PSDs." |
| `/update_metadata`, `/mass_replace`, `/alias` | commit to the card repo and apply in memory as before; stats.pkl is no longer saved. A sync of a run that started before the edit's commit shows the card without it until the next run (30 minutes at most); a restart likewise |
| `/website` | "… the next card sync applies it (every 30 minutes)." |

**API cost:** one conditional request per sync (free when nothing moved); raw downloads on a new commit are not API calls;
an /update_stats that speaks adds two status requests (the run list, a compare); a followed run about two requests per look.

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
commands/help.py, commands/psd_analyzer.py (and since the split repo_traverser.py, github_api.py; since 2026-10-22 card_data.py),
commands/query_card.py, commands/music_player.py,
commands/utils.py, global_config.py, and .gitignore. Line numbers drift —
re-find volatile facts with these before trusting them:

| Volatile fact | Re-verify with |
|---|---|
| on_ready order / loop starts / startup prints | `grep -n "queue_command(\|_start(loop)\|Bot initialization complete\|Reconnected" main.py` (134-139 queued, 126/142 starts, 127, 144) |
| Startup sweep of stale temp PSDs | `grep -n "def sweep_temp_psds\|^TEMP_\|stale temp PSD" commands/card_runner.py` (103, 27-28, 133) |
| History cleanup and recovery (section 7) | `grep -n "line = cleaned\|again = cleaned\|History cleanup skipped\|def bot_pickles\|def history_index" commands/card_data.py` |
| Fork tag in the jetson venv | on the jetson: `~/venvs/autocroissant/bin/pip freeze \| grep psd` (`@ecc4b773...` on 2026-10-23) |
| Shadow state | `gh run list -R MichaelJSr/TTSCardMaker --workflow card-sync.yml --limit 5`; `gh api repos/MichaelJSr/TTSCardMaker/branches/data` (404 until the first live run) |
| Slash-command count (52 on 2026-10-21: /fonts and /bake_cards added; 50 on 2026-10-20: /update_tts_save added, /list_guild_members removed) | `grep -c "@tree.command" main.py` |
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
| Blob-sha change check, blob downloads, 304, summary, save gate | `grep -n "_psd_changed\|def _download\|def _blob\|fetch_tree\|changed or new\|num_changed > 0 or gone" commands/card_data.py` |
| old_stats archived only on blob change or move (legacy mode) | `grep -n -A3 "def _history" commands/card_data.py` |
| Per-card cost (exclusions skipped, commit lookup only for re-parsed cards) | `grep -n "EXCLUDE_FOLDERS\|_get_remote_timestamp\|COMMITS_PAGE =" commands/card_data.py` |
| Reminder formats, limits, defaults | `grep -n "^TIME_FORMATS\|^WEEKDAYS\|^MAX_INTERVAL\|^CHANNEL_RETRY\|^LEGACY_RUNS_AS_BOT\|^INTERVAL_UNITS" commands/analytics.py` |
| Reminders run as creator; changes gated | `grep -n "_ReminderUser(\|def can_edit\|Only an admin or the reminder" commands/analytics.py` |
| Reminder files ignored | `git check-ignore -v reminder.json reminder.json.tmp reminder.pkl.bak` |
| TTS save wiring (bot: /update_tts_save only; no tts_save import in repo_traverser since 2026-10-20) | `grep -n "update_tts_save" main.py`; `grep -c "tts_save" commands/repo_traverser.py commands/card_data.py` (0, 0); `grep -n "^TTS_SAVE_PATH\|environ.get" global_config.py` (the two Action env overrides) |
| Type emoji sync wiring | `grep -n "init_type_emojis\|sync_emojis" main.py`; `grep -n "^EMOJI_SIZE\|^MAX_NAME_LEN\|^NO_SUFFIX" commands/type_emojis.py` |
| Font check wiring (/fonts; the Action is TTSCardMaker's font-check.yml) | `grep -n "check_repo_fonts\|name=\"fonts\"" main.py`; `grep -n "^BATCH\|^CACHE_JSON\|^ROLE_ID\|^BITE" commands/font_check.py`; `git check-ignore -v font_names.json` |
| Card check (CLI only; no main.py or help.py wiring; the Action is TTSCardMaker's card-check.yml) | `grep -c "card_check" main.py commands/help.py setup.py` (0, 0, 0); `grep -n "^SPARSE_FILES\|^LINE\|^CLOSING" commands/card_check.py` |
| Card bake wiring (/bake_cards; the Lock; no notice since 2026-10-21) | `grep -n "bake_cards\|name=\"bake_cards\"" main.py`; `grep -n "bake_lock" commands/repo_traverser.py commands/card_runner.py`; `grep -c "notice_edits" commands/repo_traverser.py commands/card_runner.py` (0 and 0) |
| Pickle names / timezone | `cat global_config.py` |
| Per-machine files | `cat .gitignore` |
| Music/export paths | `grep -n "MUSIC_BASE_DIR\|outtmpl" commands/music_player.py; grep -n "EXPORTED_" commands/psd_analyzer.py` |
| The cutover switch and its readers (off until the cutover commit) | `grep -n "^CARD_SYNC\|^DATA_BRANCH\|^CARD_DATA_DIR" global_config.py`; `grep -rn "global_config.CARD_SYNC" main.py commands/` |
| Card/rulebook dataframe counts, newest pickle timestamp | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |

Maintenance: update the date stamps and counts above whenever the pickles are refreshed or commands
are added. If local mode is ever revived from branch `archive/local-update-stats`, it must first stop comparing
mtimes with commit dates (B6); then revisit section 7's operating rule and the `RETIRED_OPTIONS` note in
section 9, keeping the pointer to autocroissant-failure-archaeology for the history.

2026-10-20 (later): the optimization-pass pointers above moved from `~/Documents/ac-optimization/` to
`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` (scripts) and `docs/Finished/optimization-pass-2026-10-20/` (reports, bench).

2026-10-21: the queue drains tick every 0.1 s and `process_reminders` is the 1 s reminder heartbeat (section 1); main.py line numbers re-pointed (review: on_ready start range, /update range). Review: the "Loaded N cards" print is query_card.py:472.

2026-10-21 (later): `/fonts` added (section 7's Font check subsection, section 8's `font_names.json` row, 51 commands). main.py grew by eight
lines at :278-285 (the /fonts block; its font_check import is lazy, so fontTools loads only when /fonts runs): every main.py citation in
this file re-pointed by grep the same day (review). The Action sends the Discord message before the commit comment (review).

2026-10-21 (audit fixes): `--all` parses in a process pool (section 7's standalone line re-measured); the TTS save commit path is
commands/git_commit.py (section 7's by-hand paragraph). main.py's /fonts block (:278-285) unchanged in size: the lazy import now binds the
module, not a shadowing name.

2026-10-21 (titles): section 7 gained the one-time re-parse paragraph (PARSER_VERSION 2 reads the printed title; ~813 downloads on the next /update_stats).

2026-10-21 (review): section 7 notes that `/query` titles its embed with the printed title (`query_card.printed_title`; 157 cards, case/spacing-only ones shown as printed per the user; printed titles join the name lookup); psd_analyzer.py `init_psd` :478; repo_traverser.py `update_stats` :472, `_send_summary` :443, the `UPDATE_RATE` check :391.

2026-10-21 (S7): `/bake_cards` and the edits-file notice (section 7's Card bake subsection, section 8's work-dir row, 52 commands: main.py grew by 17 lines at :21 and :399-414, every main.py, repo_traverser.py and help.py citation in this file re-pointed by the old-to-new line map). The automatic bake is a dry run under the PSD hold; one live runner when it lifts.

2026-10-21 (hook removed): the edits-file notice is gone, decided by the user: the Action is the one live runner, the jetson a manual fallback (/bake_cards only; nothing in the bot queues a bake). Section 7's Card bake subsection, section 8's work-dir row and the wiring row rewritten; repo_traverser.py lost its last two update_stats lines (:541-542), so no cited line in this file moved.

2026-10-21 (details): section 5 (/push adds psd_details.json; /pull keeps it; the /update message), section 7's PSD details subsection, section 8's `psd_details.json` row; repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-21 (card check): section 7's Card check subsection (commands/card_check.py, CLI only; TTSCardMaker's card-check.yml staged in /tmp/card-check until the orchestrator commits it) and its wiring row; nothing in main.py or help.py changed, so no cited line moved. Same day: the subsection no longer says the website shows the validator's notes (only the parser's own reach the site's Parser notes); the error-line sentence names MDW/Markers PSDs and failed API calls (exit 0 kept).

2026-10-22 (data build): /update_stats' traversal moved to commands/card_data.py (`card_data.update`, legacy mode);
the entry point, summary, download, temp-file and re-parse citations and the re-verification greps re-pointed; the
saving bullet names the per-PSD failure messages (an unreadable PSD, a failed commit lookup) that replaced a whole-run
abort.
2026-10-22 (one commit per run): section 7's Card bake subsection says one commit per run and that card-edits.yml bakes live every 30 minutes behind its `pending` gate; no cited line moved. Same day: the 40-card cap and the failed-run Discord message, then `--pending` and once-per-streak messages.
2026-10-22 (bake --local): section 7's Card bake subsection gains the `--local` bullet; no cited line moved.
2026-10-22 (cutover, bot side): section 12 (the `CARD_SYNC` switch, off until the cutover commit; commands/card_sync.py and
`card_db.sync`), pointers in sections 5, 7 and the Font check subsection, the `card_data/` row in section 8 and a
re-verification row. main.py grew by 13 lines (an import at the top, the switched texts at /pull, /push, /update, /website and
/update_stats, the option gate) and update_bot.py by 14: cited lines after those points drift; re-find them with the greps.
2026-10-22 (card guards): section 7's Card guards subsection (the CARD_GUARDS switch, dispatches, re-runs, the failure alert); no cited line moved.
2026-10-22 (card guards, review 4): the Card guards subsection gains the any-case switch, the refused dispatch, the dispatch's author, the streak artifact and the unread-comment re-run; the Card bake `--local` bullet no longer calls card-sync.yml planned; no cited line moved.
2026-10-22 (card guards, review 5): the Card guards subsection gains the dispatch outside CARD_GUARDS, `land`'s refusal of a non-`on` plan, the retried lookup, the alert's own lookup on a re-run, the re-run's window and the unstarted site build; no cited line moved.
2026-10-22 (card guards, review 6): report mode tells the PNG it would make, R11 told until the owner says yes (AUTO_FIX), a PNG pushed before its PSD, a re-run's unread branch and refresh, a cancel by hand; no cited line moved.
2026-10-23 (card guards, review 7): a new card's PNG pushed before its PSD is current too; no cited line moved.
2026-10-23 (docs drift): section 1's startup sample is the jetson journal of 10:36 (764 cards, `Repo history`, emojis
ready) with the sweep and reconnect lines; the init table gains `sweep_temp_psds` and the run-once rule; section 3's
production paragraph names the deploy window, the one-sitting rule, the fork reinstall and the journal check; section 5's
gate reads history; section 7 gains range reads, the History cleanup and recovery subsection, the new archive messages,
the .mailmap authors bullet, today's problem baselines and the bake's current refusals (soft parity, refused layers, line
splice; the jetson has the fork and fonts_cache); the card guards and card check headings say landed; section 8's temp-file,
fonts_cache and bake rows; section 12's status, step 1 done and the seed precondition; every main.py, update_bot.py,
card_data.py, psd_parser.py, query_card.py, psd_analyzer.py and analytics.py citation re-pointed at 2ada357.
2026-10-23 (BUG 2, work/fx-rebuild): section 7's Card bake subsection gains the Ability FX layers bullet (the bake runs Effect Layers.jsx's Update all; unstyled word layers still refuse); no cited line moved.
2026-10-23 (BUG 2 review, work/fx-rebuild): the Ability FX bullet names the two new refusals (a badge typed as text on a badge card; Goofy's whole-word "1") and the `1#2` rename.
2026-10-23 (BUG 2 phase 2 review, work/fx-rebuild): the typed badges bullet gains the deleted-letter badge, the typed "★", the tint refusal (any overlay, non-Normal blend), missing glyphs and edits inside own-style FX phrases; no cited line moved.
2026-10-24 (BUG 2 phase 2 review round 3, work/fx-rebuild): the typed badges bullet gains the "[fire:]" refusal; no cited line moved.

2026-10-23 (previews; revised after review the same evening: two jobs, card_preview_branch, no PNG on the branch): section 7's Card bake subsection gains the Previews bullet (card-preview.yml, `commands/card_preview.py`,
the `previews` branch; design section 16); no cited line moved.

2026-10-23 (unbatch): query_card.py citations re-pointed (prep_dataframes, init_query) after /query_ability went back to one message per card.

2026-10-24 (py314, branch work/py314): the card guards' quiet window as an environment, R7 left to font-check, G2
rendered anyway; no cited line moved.
2026-10-24 (BUG 2 phase 6, work/fx-rebuild): section 7's Card bake subsection gains the Content flags bullet (owner ruling: content problems bake, told); the FX and typed-badge bullets say told for refused; no cited line moved.
