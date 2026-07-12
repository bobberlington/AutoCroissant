---
name: autocroissant-run-and-operate
description: Day-to-day operations runbook for the AutoCroissant Discord bot. Load this when the task sounds like "run the bot", "start the bot", "stop", "restart", "deploy", "update the bot", "push/pull the pickles", "switch machines" / "handoff to the other machine", "sync commands" / "commands not showing up after deploy", "run update_stats", "set a reminder" / "schedule a command", or "where do files go" (exports, music downloads, models, temp files). Contains: the exact start command and what healthy startup output looks like line by line; on_ready init order; stopping/restarting (/stop_bot, /restart_bot, Ctrl-C, startup.sh); the per-guild command-sync model and /sync_global; the full anatomy of the git self-update flow (/push, /pull, /update, PICKLE commits, auto-merge that keeps local pickles and takes remote code, force_reset danger, every failure message and what it means) — this is the ANATOMY home; live triage of a broken update belongs to autocroissant-debugging-playbook; the multi-machine handoff runbook (pickles are the ONLY synced state); operating /update_stats (OPERATING RULE as of 2026-07-11 - always pass use_local_repo:False); artifact-location conventions; reminder operations including running bot commands on a schedule; and a guild-admin command quick table.
---

# AutoCroissant: Run and Operate

Operational runbook, verified against the code on 2026-07-11. Audience: an operator (human or model)
with zero prior context. All commands assume cwd = repo root (`/Users/michaelsrouji/Desktop/AutoCroissant`
on this Mac). "Admin" means your Discord user ID is in the `ADMINS` list in `config.py`
(`perms_check`, commands/utils.py:306 — note it returns True when you LACK permission).

Production topology (owner-stated 2026-07-11): the bot runs on the owner's Mac OR a CUDA box,
at different times — whichever machine is on hosts it. The three committed pickles
(`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are the database, and the bot's own git commands
(`/push`, `/update`) are how state moves between machines. Everything else is per-machine.

## 1. Starting the bot

Prerequisites (see autocroissant-build-and-env for building the environment):

- [ ] cwd is the repo root. Every path in the bot is relative (`stats.pkl`, `music/`, `models/`,
      `./startup.sh`). Starting from anywhere else makes the bot silently rebuild EMPTY databases —
      the tell is `stats.pkl is empty or doesn't exist, rebuilding...` in the output. Stop immediately
      if you see that on a machine that should have data.
- [ ] `config.py` exists in the repo root with a valid `TOKEN`. It is SECRET (never print or commit it).
      main.py:9 does `from config import TOKEN` — a missing/invalid config.py kills the process at import.
- [ ] No stale compiled artifacts: if `*.so` files exist from an old Cython build they shadow the `.py`
      files you just pulled. `python3 setup.py clean` first (none exist as of 2026-07-11; details in
      autocroissant-build-and-env).

Start it:

```bash
python3 main.py
```

### What healthy startup output looks like (as of 2026-07-11)

```
Git token found, API limited to 5000 requests/hour.    <- import time: StatsDatabase() (psd_analyzer.py:204-208)
Git token found, API limited to 5000 requests/hour.    <- import time: CardRepository() (query_card.py:42-45); "No git token in config..." if GIT_TOKEN unset
We have logged in as AutoCroissant#....                <- on_ready (main.py:107)
Registered 54 slash commands for reminder system.      <- slash_registry populated (main.py:109-111); 54 as of 2026-07-11
Bot initialization complete                            <- loops started (main.py:124)
Synced commands to guild: <name> (<id>)                <- one line per guild, async (main.py:101)
Trying to open reminder.pkl.                           <- init_reminder
reminder.pkl is empty or missing.                      <- or "Existing dict found in reminder.pkl, updating entries..." plus "Rescheduled reminder <id> from <t1> to <t2>" lines
Trying to open stats.pkl                               <- init_psd
Loaded existing stats from stats.pkl
Trying to open old_stats.pkl
Loaded existing old stats from old_stats.pkl
Loaded 765 cards and 48 rulebook entries               <- prep_dataframes (query_card.py:334); counts as of 2026-07-11
Stats database initialized.
Loading aliases from aliases.pkl                       <- init_query (1 GitHub tree API call follows)
INFO: No model configured for initialization           <- init_pipeline; see below for variants
```

Lines after "Bot initialization complete" come from queued async work and can interleave.
The token-presence line prints TWICE because two module singletons each print it at import.

### on_ready init order (main.py:106-124)

`on_ready` populates `slash_registry` (command name -> callback; this is what lets reminders run
bot commands), spawns a per-guild command sync task, then QUEUES four init functions and finally
starts the three queue-drain loops (main.py:121-123). Queued work only executes once the loops
start draining — so the loops effectively start first, and nothing is sent to Discord before they run.

| Init (queued in this order) | What it does | Failure/no-op behavior |
|---|---|---|
| `init_reminder` (analytics.py:26) | Loads `reminder.pkl`; reschedules past-due REPEATING reminders forward to their next future slot and saves | Missing/empty file: "reminder.pkl is empty or missing." — fine |
| `init_psd` (psd_analyzer.py:1642) | Loads `stats.pkl` + `old_stats.pkl` into `stats_db`, builds the query DataFrames | Missing pickles: rebuilds empty (dangerous if unexpected — see cwd warning) |
| `init_query` (query_card.py:627) | Loads `aliases.pkl`, fetches the TTSCardMaker GitHub file tree (exactly 1 API call) | Non-200: "Warning: Failed to populate files from repository (status N)" — /query will be broken |
| `init_pipeline` (diffusion.py:193) | Initializes the diffusion pipeline | No torch installed: "ERROR: Cannot initialize pipeline - Torch is not installed"; no `model` in config.py: "INFO: No model configured for initialization". Both are clean no-ops — the bot runs fully without AI (see autocroissant-ai-boundary) |

The three loops (main.py:823-871) each tick once per second: `process_command_queue` (also enqueues
`check_reminder` every tick — that is the reminder heartbeat), `process_dispatch_queue` (message/file
sends), `process_edit_queue` (edits). All bot output rides these queues, so replies have up to ~1 s of
latency by design. Depth on the queue model: autocroissant-architecture-contract.

## 2. Command anatomy (30-second version)

Every slash command in main.py follows one pattern: `func = to_thread(func)` then a
`@tree.command(...)` wrapper that calls it. Worker code never awaits Discord directly; it calls
`queue_message` / `queue_file` / `queue_edit` / `queue_command` (commands/utils.py:38-62) and the
loops deliver. If an interaction token has expired (Discord kills them after 15 minutes), delivery
falls back to a plain `channel.send` — so output from long jobs arrives as a normal channel message
instead of a reply. That is expected, not a bug. Admin gating is a `perms_check(interaction)` call at
the top of the wrapper. Full contract: autocroissant-architecture-contract.

## 3. Stopping and restarting

| Way | Gate | What happens |
|---|---|---|
| `/stop_bot` | admin | Stops the three loops, replies "Stopping bot!", `client.close()`, `exit(0)` (main.py:158-168) |
| `/restart_bot` | NONE (verified main.py:149-155 — no perms_check; anyone in the guild can restart) | Stops the loops, replies "Restarting bot!", then `restart_bot()` |
| Ctrl-C in the terminal | local | discord.py shuts down; main.py:880-881 prints "Bot shutting down via keyboard interrupt". Any other fatal error prints "Fatal error: ..." and exits 1 |

`restart_bot()` (commands/update_bot.py:9-13) does `execv('./startup.sh', argv)`; on
`FileNotFoundError` it falls back to `execl(python, python, main.py ...)` — i.e. re-exec the same
interpreter on main.py. The process is REPLACED, so silence after "Restarting bot!" is normal.

**startup.sh is a deploy-machine artifact, NOT in the repo** (verified absent 2026-07-11; the repo has
no startup.sh and .gitignore would not exclude it — it simply was never committed). Operators may
create one per machine, e.g. a build-then-run script (`python3 setup.py build_ext --inplace &&
exec python3 main.py`), but its contents are per-machine and UNVERIFIED — do not assume what it does
on the CUDA box. Two traps:

- Only `FileNotFoundError` triggers the fallback. A startup.sh that exists but is not executable
  raises `PermissionError`, which is NOT caught: the restart fails, the process stays alive with all
  three queue loops stopped (they were stopped before the exec), and `/restart_bot` will keep failing
  the same way. Fix: `chmod +x startup.sh` or delete it, then restart manually. `/stop_bot` still works.
- Because argv is passed through, startup.sh receives the original arguments; keep it argument-agnostic.

## 4. Command sync model

- On EVERY start, `sync_guild_commands` (main.py:97-101) runs per guild: `tree.copy_global_to(guild)`
  then `tree.sync(guild=guild)`. Guild-scoped syncs take effect essentially immediately, so after a
  restart the commands are current in every guild the bot is in (a stale Discord client may need a
  restart/reload to show them — client-side cache, not the bot).
- `/sync_global` (admin; main.py:740-746 -> management.py:74-96) clears the tree's global command set,
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

### /push — commit and push the pickles (update_bot.py:16-30)

`git add aliases.pkl stats.pkl old_stats.pkl` -> `git commit -m "PICKLE"` -> `git push`.
That is the entire commit convention: pickle snapshots are commits named PICKLE.
Success: "Successfully pushed!". Nothing changed: "Pickles are already up to date.".
Anything else: "Push failed: <err>" (then the exception re-raises).

GATE (owned by autocroissant-change-control, summarized here): pickle corruption is the #1
costliest failure class in this project's history — run the diff before every push:

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
```

### /pull — fetch and merge (update_bot.py:33-101)

Despite its Discord description text ("Does a hard reset, than a git pull" — main.py:172; help.py:10
repeats the hard-reset claim, spelled "then"), **/pull does NOT hard-reset anything**; both
descriptions are stale. Actual behavior:

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

### /update — push, pull, restart (main.py:191-204 -> update_bot.py:122-147)

Sequence: stop all three queue loops -> "Doing a complete update of the bot!" ->
"Pushing aliases.pkl, stats.pkl, and old_stats.pkl." -> git_push -> "Pulling latest changes from
remote!" -> git_pull -> "Restarting bot!" -> `restart_bot()` (process replaced; per-guild sync runs
on the way back up).

`force_reset:True` replaces push+pull with `git fetch` + `git reset --hard origin/main`
("Force reset mode: Local changes will be discarded!" / "Hard reset complete" / "Warning: All local
changes have been discarded!" — update_bot.py:104-119). **DANGEROUS: this discards local pickles,
i.e. every card parse and alias created since the last successful /push is gone.** When is it ever
right? Only when the clone is wedged (half-finished merge/rebase the bot cannot resolve) AND you have
verified the local pickles are expendable — either already on origin (diff_stats against
`git show origin/main:stats.pkl` reports 0/0/0) or known-corrupt and origin is the good copy.
Treat it as a change-control decision, not an ops reflex: autocroissant-change-control owns the gates.

### Failure messages you may see, and what they mean

| Message | Meaning / action |
|---|---|
| "Pickles are already up to date." | No pickle changes since last PICKLE commit. Normal. |
| "Push failed: <err>" | Commit/push error — usually ssh key or remote reachability. If inside /update, the update aborts. |
| "Git pull failed: <err>" | Merge machinery failed; read the git error. Rebase fallback may follow. |
| "Rebase also failed: ..." + "Manual intervention may be required." | Bot cannot self-heal the clone. SSH in, `git status`, resolve, restart. |
| "Update failed: <err>" + "Bot was NOT restarted due to errors." | /update aborted mid-way. CRITICAL QUIRK: the queue loops were stopped BEFORE the update (main.py:200-202) and nothing restarts them on failure — the bot looks alive but every queue-delivered reply (most commands) goes silent. Recover with `/restart_bot` (sends directly, still works) or a manual restart. `/pull` and `/push` alone do NOT stop the loops. |
| "Restarting bot!" then nothing | Process replaced by execv/execl. Watch the terminal/host for the fresh startup lines. |

## 6. Multi-machine handoff runbook

The pickles are the ONLY state that follows the bot. Per-machine and NOT synced (verified in
.gitignore): `config.py`, `.ssh/`, `reminder.pkl`, `music/`, `models/`, all `*.csv`/`*.txt` exports,
venvs. Both machines need their own config.py and deploy key already in place.

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

## 7. Operating /update_stats

Re-parses changed card PSDs from the TTSCardMaker repo into `stats.pkl`. Entry point
psd_analyzer.py:1173; slash wrapper main.py:351-365.

> **OPERATING RULE (as of 2026-07-11): always pass `use_local_repo:False`.** Local mode has a live
> path bug: `_process_local_files` computes the card's relative path with a `removeprefix` that
> no-ops on the absolute paths `walk()` yields (psd_analyzer.py:1002), so every card classifies from
> a top folder of "Users" -> UNKNOWN type, wrong stored paths, mass archiving — one local-mode run
> would corrupt `stats.pkl` at scale. Introduced by the 2025-12-02 cythonization commit f7c915c; the
> full story lives in autocroissant-failure-archaeology. The slash default is `use_local_repo:True`
> (main.py:361), so you must type the flag EVERY run, including inside reminder command strings.
> If a local-mode run ever happens anyway: do NOT /push; run diff_stats first (post-update checks
> are owned by autocroissant-validation-and-qa).

The four flags (slash defaults in parentheses, main.py:359-363):

| Flag | Default | What it actually does |
|---|---|---|
| `output_problematic_cards` (True) | Run `CardValidator` on each reparsed card and post the problems to the channel |
| `use_local_repo` (True — OVERRIDE TO False) | True: walk `~/Desktop/TTSCardMaker` on disk (BUGGED, see rule). False: GitHub trees API + download each changed PSD via `urlretrieve` |
| `use_local_timestamp` (True) | Local mode only: True = file mtime as the change signal; False = last-commit date from the GitHub API (extra API cost). Remote mode always uses commit dates |
| `force_update` (False) | Reparse EVERY card regardless of timestamps. (The Python-level default is True; the slash command passes False — Discord's default is what operators get) |

What a run looks like: "Updating card statistics database... This may take a while." -> the bot's
deferred reply gets EDITED to "25 cards updated.", "50 cards updated.", ... (every `UPDATE_RATE = 25`
cards EXAMINED — the count includes unchanged cards; psd_analyzer.py:29,971,1047) -> the summary
(psd_analyzer.py:1091):

```
N had newer timestamps or were new.
M did not have newer timestamps.
K cards changed location.
```

- N ("newer or new") is what gates saving: only `N > 0` triggers `prune_clean_cards()` + pickle save
  (psd_analyzer.py:1222-1224). Pruning is the deletion mechanism: cards not seen this traversal move
  to `old_stats`.
- K counts path-only moves. Verified quirk: a run with ONLY moves (N == 0) reparses and mutates the
  in-memory DB but does NOT save the pickles — the moves are lost on restart. If you need moves
  persisted, rerun with `force_update:True`.
- Then "Card statistics update complete!" and, if enabled, the problem cards: full detail first
  (path + fenced problem list, split into <=1950-char chunks), then a names-only line per card —
  all posted to the invoking channel. There are TWO problem baselines (surface distinction owned
  by autocroissant-validation-and-qa §2), as of 2026-07-11: the STORED/pickled parse-time surface
  is exactly 1 card ("20 Creature Types", MISSPELT TYPE: tornado), but the posted report re-runs
  `CardValidator.validate` per reparsed card, so a full sweep (e.g. `force_update:True`)
  legitimately posts 5 (adds Computer Virus, Anubisath Guardian, Qiraji Soldier, Silithid).
  The corruption alarm is NEW names in the report — not a count of 5 on a full sweep.

Remote-mode API budget (derivation owned by autocroissant-analysis-toolkit Recipe 4; verified
logic, psd_analyzer.py:922, 929, 1066): 2 requests of fixed overhead + 1 `get_commits` lookup per
NON-EXCLUDED PSD — the `EXCLUDE_FOLDERS` check (MDW/Markers, 91 PSDs) `continue`s BEFORE the
timestamp fetch, and the commit date IS the change detector, so it is per-card, not
per-changed-card — plus up to 1 extra request per new/author-missing card. With ~813 non-excluded
PSDs (2026-07-11) a no-change sweep is ≈ 2 + 813 ≈ 815 requests, ~16% of the 5000/hr `GIT_TOKEN`
budget (≈6 sweeps/hour theoretical; treat 2-3 as the practical ceiling, per analysis-toolkit) —
fine for a daily cadence. Without a token (60/hr) a remote run CANNOT finish (it will die mid-run;
exact failure mode untested). Check for the "Git token found" startup line.

Remote mode also downloads each changed PSD with `urlretrieve` (psd_analyzer.py:952) — see artifacts
table for where those land.

## 8. Artifact conventions — where files land

All paths relative to the CWD the bot was started from (= repo root, always).

| Artifact | Produced by | Location | Committed? |
|---|---|---|---|
| `stats.pkl`, `old_stats.pkl`, `aliases.pkl` | /update_stats, /alias, metadata commands | repo root | YES — the database, synced via /push |
| `reminder.pkl` | /set_reminder | repo root | NO (gitignored) — per-machine |
| `stats.csv` / `stats.txt` | /export_cards (`as_csv` picks which) | repo root | NO (gitignored via `*.csv`/`*.txt`) |
| `rules.txt` | /export_rulebook | repo root | NO (gitignored) |
| Music downloads | /play | `music/`, named `%(title)s-%(id)s.%(ext)s` (restrictfilenames on); playlists get a `music/<playlist title>/` subfolder (music_player.py:53,236,246) | NO (`music/` gitignored) |
| Remote-mode PSD temp files | /update_stats with `use_local_repo:False` | OS temp dir via `urlretrieve`, NEVER cleaned up (psd_analyzer.py:952) — accumulates ~a few hundred MB per full sweep; harmless but real; OS tmp cleanup reclaims it | n/a |
| Diffusion models / LoRAs | operator-provisioned | `./models/*.safetensors`, `./models/loras/` (diffusion.py:22-23) | NO (`models/` gitignored) — per-machine |
| `.ssh/` deploy key | operator-provisioned | repo root | NO (gitignored) — per-machine, SECRET |

## 9. Reminders operations

Reminder times are PST — precisely `ZoneInfo("America/Los_Angeles")` (global_config.py:9), so it
follows daylight saving. Checked once per second by the queue heartbeat (main.py:825).

- **/set_reminder** (ADMIN-gated, main.py:225) — options: `msg` (text to send), `when` (required in
  practice: `13:00`, `1PM`, or `1:30PM`; spaces/case tolerated; invalid -> "Invalid time format. Try
  `13:00` or `1PM` (PST)."), `offset` (delay added to the first run), `frequency` (repeat interval).
  `offset`/`frequency` take `<int><unit>` with units s/m/h/d/w, e.g. `30s`, `10m`, `2h`, `1d`, `1w`
  (analytics.py:87-113). `when` resolves to TODAY at that time; if that moment already passed, the
  reminder fires within ~1 second — add `offset:1d` to mean "tomorrow".
- **Command execution**: the `command` option stores a bot command that runs when the reminder fires,
  resolved through `slash_registry` and parsed with shlex + `parse_named_args` (`convert_value`
  coerces int/float/bool; utils.py:220-290). Example — the nightly stats update, with the mandatory
  flag from section 7:

  ```
  /set_reminder when:22:00 frequency:1d command:/update_stats use_local_repo:False
  ```

  **kwargs colon quirk: write `key:value` with NO space after the colon.** analytics.py:136 collapses
  every `": "` to `":"` in the stored string (added because people naturally type the space), so a
  single stray space self-heals — but any INTENDED colon-space inside a quoted value gets mangled too.
  Multi-word values need quotes: `command:/query query:"The Freezer"`. Positional args also work
  (`/update_stats True True False` maps left to right) but named args are safer.
- **/list_reminders** `[all] [hidden]` (not gated) — lists id, message, time, repeat, command per
  reminder; `all:True` = whole server, `hidden:True` = ephemeral.
- **/remove_reminder** `reminder_id` (not gated — anyone can delete) — the id is the 8-char code shown
  in backticks by /list_reminders (`uuid4` prefix, analytics.py:145).
- **Persistence**: saved to `reminder.pkl` on every change; on startup, past-due REPEATING reminders
  are rolled forward to the next future slot (analytics.py:47-53) — a one-shot reminder that came due
  while the bot was down fires immediately after startup. Per-machine file: reminders DO NOT follow a
  machine handoff (section 6 step 6).

## 10. Guild admin quick table

| Command | Gate | Notes |
|---|---|---|
| `/purge [user] [num] [bulk]` | none | Deletes messages in the current channel; defaults to the BOT's own messages, num 100 (`-1` = up to 1,000,000); `bulk:True` is faster but needs the bot to have Manage Messages (main.py:701-713) |
| `/list_guilds` | admin | Name, ID, member count per guild |
| `/leave_guild guild_id` | admin | Numeric ID required |
| `/list_guild_members guild_id` | admin | Requires the privileged members intent, which is NOT enabled — main.py:91-92 sets only `Intents.default()` + `message_content`. So as of 2026-07-11 this command always replies "Intents.members must be enabled to use this." (ClientException handler, management.py:135-136). Enabling the intent is a code+portal change: route via autocroissant-change-control |
| `/list_guild_channels guild_id` | admin | Works with default intents |
| `/get_channel_messages guild_id channel_id [limit]` | admin | Default 50, `-1` = all (1M cap); text channels only; oldest first; permission failure -> "I don't have permission to read messages in that channel." |
| `/react reactions [message_id]` | none | Space-separated emojis (`👍 👎` or `<:custom:1234567890>`); no id = most recent message; replies "Added X/Y reactions..." |
| `/sync_global` | admin | Section 4 |

## When NOT to use this skill

- Building or fixing the Python environment, ffmpeg/opus, Cython builds, requirements split ->
  **autocroissant-build-and-env**.
- Something is BROKEN (won't start, commands missing, silent bot, stats look wrong, music/diffusion
  failures) -> **autocroissant-debugging-playbook** (this skill only tells you what healthy looks like).
- Changing code, pickles-as-data decisions, what may be committed, force_reset approval ->
  **autocroissant-change-control**.
- The story behind the local-mode bug and other incidents -> **autocroissant-failure-archaeology**.
- Judging whether an update's output is GOOD (golden cards, diff review, problem-count gates) ->
  **autocroissant-validation-and-qa**.
- Script usage details for inspect_pickle/diff_stats/parse_one -> **autocroissant-diagnostics-and-tooling**.
- What a config field or module constant means -> **autocroissant-config-and-flags**.

## Provenance and maintenance

Written 2026-07-11 against the working tree at commit 284d13c ("PICKLE"). Every claim was verified by
reading main.py, commands/update_bot.py, commands/analytics.py, commands/management.py,
commands/help.py, commands/psd_analyzer.py, commands/query_card.py, commands/music_player.py,
commands/diffusion.py, commands/utils.py, global_config.py, and .gitignore. Line numbers drift —
re-find volatile facts with these before trusting them:

| Volatile fact | Re-verify with |
|---|---|
| on_ready order / loop starts / startup prints | `grep -n "queue_command(init\|process_.*_queue.start\|Bot initialization complete" main.py` |
| Slash-command count (54) | `grep -c "@tree.command" main.py` |
| Members intent still off | `grep -n "Intents" main.py` |
| /restart_bot still ungated | `grep -n -A4 'name="restart_bot"' main.py` |
| /update stops loops first | `grep -n -B2 -A6 'name="update"' main.py` |
| startup.sh still absent from repo | `ls startup.sh; git ls-files startup.sh` |
| execv/execl fallback | `grep -n -A4 "def restart_bot" commands/update_bot.py` |
| PICKLE commit convention / merge strategy | `grep -n "PICKLE\|--ours\|--theirs\|AUTO-RESOLVE" commands/update_bot.py` |
| Push remote is ssh | `git remote -v` |
| update_stats slash defaults (use_local_repo True) | `grep -n -A6 "def slash_update_stats" main.py` |
| Local-path bug still present | `grep -n "removeprefix" commands/psd_analyzer.py` (fix candidate: relpath — owned by the campaign skill) |
| UPDATE_RATE / summary line | `grep -n "UPDATE_RATE\|had newer timestamps" commands/psd_analyzer.py` |
| Save-gate on num_new (moves-only runs unsaved) | `grep -n -A2 "num_new > 0" commands/psd_analyzer.py` |
| Remote per-card API cost (exclusions skipped pre-fetch) | `grep -n "EXCLUDE_FOLDERS\|_get_remote_timestamp" commands/psd_analyzer.py` (exclude check must sit above the timestamp fetch in `_process_files_from_response`) |
| urlretrieve temp files | `grep -n "urlretrieve" commands/psd_analyzer.py` |
| Reminder colon quirk / time formats / units | `grep -n "replace(': '\|%H:%M\|timedelta(" commands/analytics.py` |
| Pickle names / timezone | `cat global_config.py` |
| Per-machine files | `cat .gitignore` |
| Music/export/model paths | `grep -n "MUSIC_BASE_DIR\|outtmpl" commands/music_player.py; grep -n "EXPORTED_\|MODELS_FOLDER\|LORAS_FOLDER" commands/psd_analyzer.py commands/diffusion.py` |
| Card/rulebook dataframe counts, newest pickle timestamp | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |

Maintenance: update the date stamps and counts above whenever the pickles are refreshed or commands
are added; if the removeprefix bug gets FIXED, rewrite section 7's operating rule (and keep the
pointer to autocroissant-failure-archaeology for the history).
