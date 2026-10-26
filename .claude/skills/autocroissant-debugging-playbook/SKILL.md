---
name: autocroissant-debugging-playbook
description: 'Symptom-to-triage playbook for live AutoCroissant failures. Load when something is BROKEN: "bot not responding" / "bot said nothing" / "message never arrived"; "stats look wrong" / "cards parsed wrong" / "everything is unknown type" / "old_stats ballooned" / pickle corruption; "slash command not showing up / stale / duplicated"; "command seems to hang" / "bot froze"; "bot won''t start" / "my edits don''t take effect"; "music silent" / "download fails" / yt-dlp errors / volume not working; "/ai stuck" / "Request queued forever" / OOM / "PyTorch is not installed." (branch archive/ai-generation only); "/update or /pull broke the bot" / merge conflict / bot never came back; "reminder didn''t fire" / scheduled command failed / "Reminder skipped its command" / "Reminders are off" / "You do not have permission" from a reminder; "emojis show as [fire] text" / "Emoji sync failed" / /sync_emojis. Contains symptom-indexed triage tables with discriminating experiments, the dispatch/edit-queue fallback chains and their silent-loss modes, and the card-data corruption triage (data commit diffs, gates, a restore dispatch; the pickle-era revert is history). Deepest coverage: pickle corruption and Discord API quirks. Live triage only - design questions go to autocroissant-architecture-contract; incident history to autocroissant-failure-archaeology; self-update flow anatomy and message meanings to autocroissant-run-and-operate.'
---

# AutoCroissant Debugging Playbook

Symptom-indexed triage for the failure modes this bot actually has, verified against the code and
git history on 2026-07-11; counts, history-step rows and line numbers re-verified 2026-10-23 (main 2ada357). Find your symptom's section, run the "First check", then use the
discriminating experiment to separate look-alike causes before touching anything.

Ground rules while debugging:

- The diagnostic scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`
  (`inspect_pickle.py`, `diff_stats.py`, `parse_one.py`, `gap_trace.py`, `dump_psd_layers.py`) are
  read-only and safe to run any time. Usage and golden outputs: **autocroissant-diagnostics-and-tooling**.
- Never debug by dispatching Card sync runs or publishing card data "to see what happens" - the card data is the
  `data` branch's `db/`, written only by the Card sync Action. Any fix that touches it routes through
  **autocroissant-change-control**.
- `config.py` is secret (tokens). Check that it exists; never print its values.
- All commands below assume cwd = repo root (`/Users/michaelsrouji/Desktop/AutoCroissant`).

Jargon used throughout: the bot drains three module-level deques the moment something is queued (a 1 s poll if a wake-up is missed; 0.1 s ticks 2026-10-21 to 10-25) from `tasks.loop`
coroutines in `main.py` - `command_queue` (deferred function calls), `dispatch_queue` (message/file
sends), `edit_queue` (edits of a command's original response). Workers run in threads and only ever
call `queue_message` / `queue_file` / `queue_edit` / `queue_command` (commands/utils.py:36-60).
Design rationale lives in **autocroissant-architecture-contract**. A "PICKLE commit" (history) was a commit of the
pickle files alone, made by the bot's `/push` until 2026-10-25.

---

## 1. "Stats/cards look wrong" (card data corruption - costliest failure class)

The database is `db/` on TTSCardMaker's `data` branch (`stats.json`, `old_stats.json`, `psd_details.json`,
`manifest.json`, `changelog/`), built by the Card sync Action with this repo's `commands/card_data.py`; the bot syncs it
into `card_data/` (one conditional request; on startup, `/update_stats` and card commands on use). Until 2026-10-25 it
was this repo's pickles, written by the jetson's `/update_stats` and moved by `/push` (that era's incidents:
**autocroissant-failure-archaeology** Entries 1, 3 and 10.32).

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| The bot shows old data | `/update_stats`: "already current: built … from TTSCardMaker@…" or a sync failure line | The data branch has not moved (Card sync not run yet, its window, a failed run) or the bot's sync failed | `gh run list -R MichaelJSr/TTSCardMaker --workflow card-sync.yml --limit 5`; the reply's notes ("card changes not in the data yet", "The last card sync failed: <run>") | Wait for the run or dispatch one (`/update_stats cards:<name>`, admins); a failed sync keeps the last good data and says why (**autocroissant-run-and-operate** §12) |
| The bot shows no cards at all after a start | console: "Card data sync failed (...)" and no "Card data loaded from card_data" line | No `card_data/` yet (a new host, the folder deleted) and GitHub unreachable | `ls card_data/` | `/update_stats` once GitHub answers; nothing else to fix |
| Many cards changed, removed or "unknown" in one data commit | `diff_stats.py origin/data~1 origin/data --git ~/Desktop/TTSCardMaker` (autocroissant-diagnostics-and-tooling Workflow 1) | A builder regression on main (the Action runs main's builder unless `CARD_DATA_BUILDER_REF` pins one) or a bad input | Field counts dominated by one field; the run's job summary and the data commit's per-card lines | Pin `CARD_DATA_BUILDER_REF` (or `gh workflow disable card-sync.yml -R MichaelJSr/TTSCardMaker`), then dispatch `restore=<good data commit>` (DATA_BUILD_ACTION.md §13); fix and re-enable |
| One card's stats/ability wrong | `inspect_pickle.py "Card Name"` (the bot's `card_data/`) - is the stored record itself wrong? | Parser issue (gaps/type injection) vs query-layer issue | Record correct but `/query` output wrong -> query layer (alias shadowing, fuzzy match at match_ratio 0.6, stale dataframes). Record wrong -> parser, or a stale read (`/update_stats cards:<name>`) | Parser root-causing recipes: **autocroissant-analysis-toolkit**; improving the parser: **autocroissant-psd-extraction-campaign** |
| old_stats grew a lot in one data commit | `diff_stats.py --history` between the two data commits (`versions added`) | Duplicate archiving regression (the build archives a version only when a card's data changes, `card_history.same_data`) | Versions added vs the run's changed/edited events | As above: pin, restore, fix |
| Card sync refused (red run, "Gate refusals" in its summary) | the run's summary and artifact (the would-be db/) | A gate (G1 floor, G2 archive ceiling, G4 same-blob re-parses, G5 quality, G3 round trip, G6 another writer) caught a change | `diff_stats.py origin/data <artifact>/db` | Review, then dispatch with the gate's input (`allow_archive`, `full`, `cards`) or fix the cause (DATA_BUILD_ACTION.md §13) |
| Query results stale right after a sync | Did `populate_files`/`prep_dataframes` run? A sync queues them (`card_sync._follow_ups`) | Command queue stopped before the refresh ran | `/query` a card you know changed; restart fixes it | Restart the bot; see section 2 for why queues stop |
| /query finds no card at all after a restart | stdout: "Card list: GitHub tree fetch for <repo> failed (...); keeping the last good list (N entries)" | GitHub was down or rate-limited at startup (3 tries, 1 s and 2 s apart); right after a start the last good list is empty | The line names the HTTP status or error | Wait for GitHub, then `/update_stats` (it refreshes the card list) or restart |

---

## 2. "Message never arrived / bot said nothing" (Discord API quirks - second costliest class)

The dispatch chain (main.py:750-887: `_fallback_send` :750, `_dispatch` :765, which also turns a `links` kwarg into
link buttons; `_edit` :782-798), drained the moment it is queued (1 s fallback poll):

```
worker thread --queue_message/queue_file--> dispatch_queue --loop-->
  0. content over 2,000 chars: split (split_long_message) and requeued at the front, files on the last chunk
  1. interaction.response.send_message(...)      if response not yet done
  2. else interaction.followup.send(...)         (attachments converted to files)
  3. on HTTPException/AttributeError: _fallback_send
       drop the 'ephemeral' flag, rewind files
       client.get_channel(interaction.channel_id).send(...)   <- plain channel message
       (channel None -> "Dropped a message: channel N not found")
  any other error, or one raised in 3 -> "Dispatch: dropped a message for channel N: <error>"; the loop goes on
```

The edit chain (`process_edit_queue`/`_edit`) tries `interaction.edit_original_response(...)` (over-long
content: first chunk edited, the rest queued as new messages after it, ephemeral if the original is); on the
same exceptions it falls back to `_fallback_send` - i.e. a NEW message instead of an edit; if that finds no
channel, the rest is dropped too. Failures log "Edit: dropped a message ...".
Since 2026-10-20 (B5 fixed) one bad message can no longer stop either loop.

Two Discord facts drive most of this: interaction tokens expire after 15 minutes (Discord-documented),
after which responses/followups/edits raise HTTPException; and fallback channel sends are ordinary
messages with no reply linkage, no ephemeral support.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| Bot says nothing, no error visible anywhere | Is the process alive? Then stdout for `Unhandled exception in internal background task` | Dispatch loop crashed, or all loops were deliberately stopped | Run a trivial command like `/help`: reply -> dispatch alive, your problem is worker-side; no reply -> loops dead | `/restart_bot` (works with dead loops: it replies via `interaction.response` directly and exec-restarts, main.py:167-175; note it has NO permission check) |
| Bot mute right after a failed `/update` | Discord for "Update failed:" + "Bot was NOT restarted due to errors." | `/update` stops all four loops BEFORE doing git work (main.py:225-228); on failure nothing restarts them (a reconnect does not either: stopped on purpose) | Those two messages arrive (sent via direct followup, not the queues) but nothing queued afterwards does | `/restart_bot`. Anything queued while loops were stopped is lost on restart (in-memory deques) |
| Reply arrived as a plain message; the slash command shows "The application did not respond" | How long did the worker run? | Work exceeded 15 min -> token expired -> fallback path 3 | Expected: reply content still arrives, unattached; the "did not respond" banner is cosmetic | Working as designed. Long jobs (`force_update` update_stats, big exports) always end like this |
| An ephemeral ("only you") reply became public or seems lost | Was the interaction old/expired? | Fallback pops `ephemeral` (main.py:872) - ephemeral cannot survive the channel.send fallback | Same reply within 15 min is ephemeral; after, it is public | Known weak point (listed in **autocroissant-architecture-contract**); don't rely on ephemeral for slow commands |
| Progress "edits" arrive as a stream of new messages | Which interaction were they attached to? | Edit-queue fallback sends new messages (`_edit`, main.py:782-798). Progress user: update_stats "N cards updated." every 25 re-parsed cards (`_progress`, repo_traverser.py:51; UPDATE_RATE card_data.py:43) | Starts as edits, degrades to sends at the 15-min mark | Cosmetic; ignore |
| Nothing sent AND stdout shows a Python traceback | Read the traceback (often logged late as "Task exception was never retrieved") | The worker crashed before queueing; `queue_*` never raise (they just append, utils.py:36-60), so anything queued BEFORE the crash still arrives | Partial output followed by silence = mid-worker crash | Fix the worker bug; file the story in **autocroissant-failure-archaeology** |
| One message missing, the rest fine | stdout: `Dispatch: dropped a message for channel N: ...` or `Dropped a message: channel N not found` | That send failed twice (e.g. Forbidden: no Send Messages; deleted channel; bad kwargs). Before 2026-10-20 this killed the loop (B5, archaeology 10.22) | The log line names the channel and error | Fix permissions or the sender; nothing to restart |
| Reminder fired but message missing | Reminder messages ALWAYS deliver via fallback path 3 (FakeInteraction -> plain Object -> AttributeError by design, utils.py:124-191) | Since 2026-10-19 the scheduler skips channels the bot can't see (`_channel_missing`), so a deleted channel no longer kills the loop through reminders; stdout shows "Reminder <id>: channel <id> not found (deleted or no access); skipped." | Check stdout for that line, or for a `Dispatch: dropped a message` line | Remove the stale reminder (`/reminders`, `/remove_reminder`) |

Silent-loss modes, summarized: (1) `queue_*` functions never raise to callers - a full deque with no
drainer is indistinguishable from success at the call site; (2) the three loops are stopped by
`/restart_bot`, `/stop_bot`, and `/update` (main.py:170-173, 183-186, 225-228 as of 2026-10-23) - a failed update
leaves them stopped; (3) a crashed loop stays crashed until a reconnect revives it (since 2026-10-21 a repeat
`on_ready` after a failed RESUME restarts every loop whose task failed: look for `Reconnected: inits skipped, revived
<loops>`) or the process restarts; loops stopped on purpose stay stopped; (4) ephemeral flags are
dropped in fallback. The discriminating experiment is always the same: compare bot stdout (worker
prints, tracebacks, loop-crash lines) against what reached Discord.

---

## 3. "Slash command missing / stale / duplicated in Discord"

Sync model: `on_ready` copies global commands to each guild and syncs per guild
(main.py:102-128; a reconnect syncs only guilds not synced yet) and prints `Synced commands to guild: <name> (<id>)` per guild plus
`Registered N slash commands for reminder system.` (47 since 2026-10-25: five Discord edit commands retired; 52 from 2026-10-21; built from `tree.get_commands()` at main.py:130-132, including
`/sync_emojis` - a quick sanity number). There is NO `on_guild_join` handler. Full ops detail: **autocroissant-run-and-operate**.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| New/changed command not in Discord after deploying code | Did the bot process restart since the code change? | Registration happens at import time; sync at on_ready | stdout has the "Synced commands to guild" lines with a fresh timestamp? | Restart the bot; per-guild sync applies within seconds |
| Command missing in ONE server only | Was the bot added to that guild after the last restart? | No on_guild_join -> new guilds get no sync until restart | Works in old guilds, missing in the new one | Restart, or `/sync_global` (admin) which also re-syncs every guild (commands/management.py:84-97) |
| Every command listed TWICE | Client shows two entries per name | Commands registered in both global and guild scope | `/sync_global` clears the global set then re-syncs guilds (management.py:87-94) | Run `/sync_global`; expect "Commands synced globally and to N guilds." |
| Stale parameter list / description in the client | Try in a different client (mobile vs desktop) | Discord client-side cache | Fresh client shows the new schema | Ctrl+R (desktop client reload); guild syncs are near-immediate, global scope can take up to ~1 hour (Discord-documented) |
| `CommandAlreadyRegistered` on startup | The failing command NAME in the traceback | Two `@tree.command` decorators with the same `name=` | - | Rename one; add-a-command checklist: **autocroissant-change-control** |
| "Were the duplicate functions in main.py the bug?" | No, and they are gone | C5: the second `slash_set_ratio` (/set_repo) and `slash_delete_song` (/delete_all_music) were renamed `slash_set_repo` / `slash_delete_all_music` on 2026-10-20; registration never depended on the def names | - | - |

---

## 4. "Command seems to hang"

Nothing in this bot blocks the event loop by design: slash handlers `await` workers wrapped by
`to_thread`, and all sends go through the queue loops. "Hangs" are therefore one of four
things:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `/update_stats` "hung" | Is it following a dispatched run (force_update or cards)? It edits the state into the reply ("waiting for its turn", "running: Build") | A dispatched Card sync run is followed for 30 minutes at most (a full re-parse continues over the next runs); a plain sync is one request | The Action's run page | Wait; the reply ends with the run's link either way. The bot itself parses nothing (the legacy parse, which took minutes, went 2026-10-25) |
| Every command mute | `/help` test from section 2 | Queue loops stopped/crashed - all dispatch shares them | See section 2 | `/restart_bot` |
| (branch `archive/ai-generation` only; main has no /ai) `/ai` says "Request queued. You'll be notified when generation starts." forever | `/ai_queue` - is the queue only ever growing? | The `in_progress` flag (diffusion.py:127) is wedged True. `diffusion()` sets it at diffusion.py:683 and clears it at :782 with NO try/finally - any exception mid-generation (OOM, bad image URL, NoneType pipeline) wedges it permanently. (`init_pipeline` DOES clear it in a finally, diffusion.py:255) | stdout has a generation traceback from the first failed request; every later `/ai` only queues | `/restart_bot`. Only one generation runs at a time by design; a try/finally is an open candidate fix - route via **autocroissant-change-control** |
| Music download "hung" | stdout for a yt-dlp traceback | Downloads run on a dedicated 2-worker pool (music_player.py:35); two stuck downloads block the third | "Downloading: url" arrived but no "Queued song:" - the completion hook only fires on finish (music_player.py:268-269) | Section 6; update yt-dlp first |

---

## 5. "Bot won't start"

Healthy startup output, line by line, is owned by **autocroissant-run-and-operate**. Triage of
unhealthy starts:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | `ls config.py` | config.py missing (it is secret and never committed) | - | Recreate it; field names (values are secret): **autocroissant-config-and-flags** |
| `Fatal error: ...` then exit 1 | The wrapped exception text (main.py:985-992 catches everything around `client.run`) | `Improper token has been passed.` = TOKEN wrong/rotated | - | Fix TOKEN in config.py |
| The process dies with no Python traceback (SIGSEGV/SIGBUS; systemd shows a signal) | `journalctl -u autocroissant` on the jetson: a traceback, or `status=11/SEGV` / `7/BUS`? | Until 2026-10-23 compiled modules ran with `boundscheck=False`: a bad index was undefined behaviour (a blank star folder read out of bounds, fixed in ba27fb7; the 2026-10-20 psd_analyzer SIGSEGV/SIGBUS was that class). Since 9919a14 a bad index in compiled code raises IndexError with a traceback, so a signal now points at a C extension (the fork's `_distance`, numpy) or at a script that replaces `datetime` before pandas is imported (the 2026-10-22 crash reports: pandas' NaT GC traverse recursion, all from rehearsal scripts, never the bot) | Traceback = a Python bug, read it; signal = check which process and what it imported first | Rebuild (`setup.py build_ext --inplace`), reinstall the fork tag if `_distance` is suspect; in scripts import pandas before patching datetime. History: **autocroissant-failure-archaeology** |
| `PrivilegedIntentsRequired` | Discord developer portal | `intents.message_content = True` (main.py:93) requires the Message Content intent enabled for the app | - | Enable the intent in the portal |
| Your code edits "don't take effect", or behavior is bizarrely old | `ls commands/*.so` | Stale compiled Cython artifacts shadow the edited .py at import time | Remove artifacts and rerun: behavior changes -> it was shadowing | `python3 setup.py clean` (custom CleanAll command). Build system detail: **autocroissant-build-and-env** |
| Startup line "card_data: a sync was cut off (...)" or "Card data sync failed (...)" | `ls card_data/`; `cat card_data/sync.json` | A sync cut off mid-write, or GitHub unreachable at start | The next sync reloads a cut-off copy whole | Nothing: `/update_stats` once GitHub answers (the bot never unpickles since 2026-10-25) |
| `ImportError`/`ModuleNotFoundError` for cv2, psd_tools, discord, git, ... | `which python3`; is the right env active? | Wrong interpreter or missing requirements | - | **autocroissant-build-and-env** (requirements.txt; the AI requirements2.txt is only on branch `archive/ai-generation`) |

---

## 6. "Music silent / download fails"

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| "Downloading: url" then nothing | stdout for a yt-dlp traceback | YouTube churn breaking yt-dlp (the perennial cause) | Run the same URL with the standalone `yt-dlp` CLI | `pip install -U yt-dlp` FIRST, before debugging anything else (it is unpinned in requirements.txt on purpose). Then retry |
| Download errors mentioning cookies / Safari | Is `browser_cookies` set in config.py? | yt-dlp uses Safari's cookies when `config.browser_cookies` is True; only old configs without that key fall back to `vram_usage == "mps"` (`cookies_from_browser`, music_player.py:36-37, applied at :55) | Toggle `browser_cookies` and retry the same URL (age-gated videos NEED cookies) | Terminal needs macOS disk access to read Safari cookies. Flag catalog: **autocroissant-config-and-flags** |
| "Now playing: **song**" but silence | stdout for `ClientException` ("ffmpeg was not found") or a voice error | ffmpeg missing, or voice deps (PyNaCl/opus) missing | The "Now playing" message is queued BEFORE the actual `vc.play` runs (music_player.py:145-150), so it proves nothing about playback | Install ffmpeg; env checklist: **autocroissant-build-and-env** |
| `/volume` refuses | The exact reply | "Volume control not available with FFmpegOpusAudio. Switch to FFmpegPCMAudio to adjust volume." (music_player.py:326-327) - all playback uses FFmpegOpusAudio (music_player.py:132,147) | - | By design; not a bug |
| Bot thinks it is/isn't in voice ("stuck") | `/disconnect` reply | `state.vc` (music_player.py:41) out of sync with reality | Reply "Not currently in a voice channel. If stuck, use `/play` then `/disconnect`." (music_player.py:427) is the built-in escape hatch | Do exactly that: `/play` (reconnects), then `/disconnect` (resets state) |
| `/play` replied "You are not in a voice channel." | Was the user in voice? | `init_vc` returns False and `/play`/`/play_all` stop there (nothing downloaded). Before 2026-10-20 it raised `TypeError` after queueing the reply (B4, archaeology 10.21) | - | Join a voice channel first |
| Song skipped with "File not found: path" | Was music/ cleaned between queueing and playing? | Queue holds paths, not files (music_player.py:141-142); it auto-advances | - | Re-queue |

---

## 7. "AI generation fails / OOM"

> **AI left main on 2026-10-20:** this section applies to branch `archive/ai-generation` (main has no AI code).


VRAM modes, model routing, and enable/disable runbooks are owned by **autocroissant-ai-boundary** (branch `archive/ai-generation`).
Triage layer only:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `/ai` replies "PyTorch is not installed." | Intentional? | torch absent = the AI boundary doing its job (diffusion.py:661-662); core bot runs fine without it | `python3 -c "import torch"` in the bot's env | Enable via requirements2.txt if wanted: **autocroissant-ai-boundary** (branch `archive/ai-generation`) |
| stdout: `INFO: No model configured for initialization` | config.py `model` field empty | `init_pipeline` returns without building pipelines when model is '' (diffusion.py:202-204) | TRAP (verified code path): with torch installed but model empty, `/ai` prints "Initializing AI pipeline...", init does nothing, then the NoneType pipeline call crashes and WEDGES `in_progress` (section 4) | Set a model (`/set_model` or config.py), `/restart_bot`, retry |
| OOM / dtype / device errors mid-generation | stdout traceback | Wrong VRAM mode for the machine | - | Mode table: **autocroissant-ai-boundary** (branch `archive/ai-generation`). Then `/restart_bot` - the failed run wedged `in_progress` (diffusion.py:683 vs :782, no try/finally) |
| "Request queued" but never starts | `/ai_queue` | Wedged `in_progress` | See section 4 row | `/restart_bot` |

---

## 8. "Self-update broke" (/pull, /update)

The canonical anatomy of the flow, normal operation, and the full failure-message table (what
each message means) live in **autocroissant-run-and-operate** §5. `force_reset` danger and
commit discipline: **autocroissant-change-control**. This section is the triage layer only
(first check / discriminating experiment per symptom); message strings verbatim from
commands/update_bot.py:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| "Git pull failed: ..." then "Update failed: ..." and "Bot was NOT restarted due to errors." | The git error text | The pull failed (auth, a timeout, a conflict it could not resolve). `/update` had ALREADY stopped the queue loops | Bot is now half-dead: direct followups worked, queued sends won't | `/restart_bot`, then fix the git issue manually on the box |
| "Merge conflict detected. Resolving automatically..." | Following messages list per-file decisions | Divergent histories (someone committed on the bot's machine) | Expected: "Accepting remote version of <file>" for every file, committed as "AUTO-RESOLVE: Merge conflict resolved (accepted remote)" | Local commits' changes in conflicting files are gone; look at `git log` on the box |
| "Git pull failed: ... divergent branches" then "Attempting to reconcile divergent branches..." | Next message | Fallback `git pull --rebase` (update_bot.py:176-190) | "Rebase successful!" = recovered; "Manual intervention may be required." = it is | SSH to the machine and resolve by hand |
| Update said "Restarting bot!" but the bot never came back | `systemctl status autocroissant` on the jetson | `restart_bot` execs `./startup.sh` if present, else re-execs main.py with the same interpreter (update_bot.py:71-75); the jetson has no startup.sh. If the pull brought NEW dependencies or changed a compiled module (stale .so), the boot dies on ImportError | The service journal shows the traceback | Install deps (`~/venvs/autocroissant/bin/python -m pip install -r requirements.txt`; **autocroissant-build-and-env**), `~/venvs/autocroissant/bin/python setup.py build_ext --inplace`, `sudo systemctl restart autocroissant` |
| "Warning: All local changes have been discarded!" | Did someone pass `force_reset:True`? | `git_reset_hard` ran - local code changes are GONE, replaced by origin/main (the card data is untouched: `card_data/` is gitignored) | `git reflog` on the box | Damage control via **autocroissant-change-control** |
| "/pull did a hard reset?!" | It did not | `git_pull` contains no reset - fetch + merge/rebase only (the description said "hard reset" until 2026-10-20, C1) | Read update_bot.py `git_pull` | Only `/update force_reset:True` resets |

---

## 9. Reminder quirks (rewritten system, 2026-10-19)

Reminders live in `reminder.json` (gitignored - NOT synced across machines; they die with
the machine, see **autocroissant-run-and-operate** §6 and §9). `check_reminder` is queued every
second by `process_reminders` (main.py:920) and run by the command drain, so reminders only fire while the loops run. The scheduler
logs to stdout; most reminder problems are visible there first.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| "✗ Reminder NOT set: ..." | The reason after the colon | `when`/`offset`/`frequency` unparseable or out of range (max 366 days), a date in the past, or the command failed its signature check (unknown option, bad value, required option missing) | The reason names the field | Fix that field; formats in **autocroissant-run-and-operate** §9 |
| "⚠ Reminder `<id>` skipped its command: ..." at fire time | `/reminders` - the stored command | The command or one of its options was renamed or removed since the reminder was made (it is re-checked on every firing) | Compare with the current `/help` entry | Re-create the reminder, then `/remove_reminder reminder_id:<id>` |
| "⚠ Reminder `<id>`: `/<name>` failed (...)" | stdout traceback "ERROR: reminder <id> command /<name> raised" | The command itself crashed | Run the same command by hand | Fix the command; the reminder keeps repeating |
| A scheduled admin command replies "You do not have permission." | `/reminders` - who is the creator? | Reminders run as their creator since 2026-10-19 and the creator is not (or no longer) in ADMINS | Creator shown as `<@id>`; check ADMINS on this host | An admin re-creates it. Legacy reminders without a creator still run as the bot |
| Every reminder command replies "✗ Reminders are off: ..." | stdout at startup: "ERROR: reminder.json could not be read" | Unreadable reminder.json (hand edit? a newer schema_version?), left untouched on purpose | `python -c "import json; json.load(open('reminder.json'))"` | Fix the file (stop the bot first), or move it aside and restart (fresh schedule). A deleted reminder.json starts an empty schedule (the reminder.pkl conversion was removed 2026-10-25) |
| A reminder shows "⚠ <problem>" in `/reminders` | stdout "is broken and was stopped" / "can't run (...)" | It crashed the scheduler or its stored data is unreadable; only that reminder stopped | - | Reschedule it (clears the problem) or delete it |
| A reminder never fires, no log line | `/reminders` - Status Off? | It was turned off (kept, frozen) | - | Turn it on: a repeating one skips its missed runs; a past one-shot needs Reschedule first |
| A panel button says "This interaction failed" | Was the bot restarted since the panel opened, or was it idle 10 minutes? | Panels don't survive a restart and close after 10 idle minutes | - | Run `/reminders` again |
| Repeating reminder "skipped" runs while bot was down | stdout at startup: `Rescheduled reminder <id> from ... to ...` | By design: it fires once (if due) and jumps to its next future slot; missed runs are not replayed. Past-due ONE-SHOT reminders fire right after startup | - | By design |
| One-shot reminder never fires, keeps moving an hour later | stdout "channel ... not found (deleted or no access); skipped." | Its channel is gone or invisible; it retries hourly (`CHANNEL_RETRY`) | - | Remove it, or restore the bot's access to the channel |
| Reminder fired at the wrong hour | What timezone did you assume? | Times are Pacific: `TIMEZONE = ZoneInfo("America/Los_Angeles")` (global_config.py), following daylight saving | The list shows a Discord relative timestamp next to the Pacific time | State times in Pacific time |
| A stored `/update_stats use_local_repo:...` changed to plain `/update_stats` | stdout at start: "Reminder <id>: migrated command ... (dropped retired `use_local_repo:False`)" | Historical: until 2026-10-25 `RETIRED_OPTIONS` stripped the removed options (local mode archived 2026-10-20); removed 2026-10-25, so such a reminder now posts "`use_local_repo` is not one of its options" when due | - | Nothing to do; operating rule in **autocroissant-run-and-operate** §7 |
| An old reminder behaves differently after the 2026-10-19 rewrite | stdout at first start: "Reminder <id>: migrated command `old` -> `new`" | Migration kept the OLD meaning (old parser read only lowercase `true` as True; pre-v2 positional values follow the old option order) | Compare old and new command strings in the log | Re-create the reminder with the options you actually want |

## 10. Type emojis (/sync_emojis, added 2026-10-19)

Untested against live Discord as of 2026-10-19 (docs/BUGS_AND_IMPROVEMENTS.md V1). Operations:
**autocroissant-run-and-operate** §11.

| Symptom | First check | Likely cause | Fix / pointer |
|---|---|---|---|
| `/query_rulebook` shows `[fire]` as text instead of an icon | stdout at startup: "Type emojis: ... ready." or "Emoji sync failed: ..." | The startup sync failed or has not finished, so the token map is empty. Since 2026-10-20 a failed GitHub listing falls back to the card list's last tree, and if that is missing too the sync still loads the existing emojis ("using the N existing emojis") | Run `/sync_emojis`; read its report |
| "Emoji sync failed: ... has no attribute 'fetch_application_emojis'" | `python3 -c "import discord; print(discord.__version__)"` | discord.py older than 2.5 | `pip install -U "discord.py>=2.5"` (**autocroissant-build-and-env** §2) |
| "Emoji sync failed: 403/404 ..." on the GitHub listing | Is GIT_TOKEN set; is `/set_repo` pointing at the card repo? | Rate limit without a token, or a wrong repo | Fix the token or `/set_repo`, then `/sync_emojis` |
| Report lists "Failed: <name> (...)" | The error text per emoji | Discord rejected an upload (name, size, or the application's emoji limit) | Fix the icon or name; rerun |
| An edited Types/ icon still shows the old image | Did an /update_stats since the edit say "Type emojis: ... updated"? | /update_stats refreshes emojis only when the default card repo's tree changed (not after `/set_repo` elsewhere); an emoji with no recorded blob for this bot (JSON missing, new identity) is adopted as current | `/sync_emojis refresh:True` |
| Emojis work on one host but not another | Which bot identity is logged in (config.py comment lines only - never print values) | Application emojis belong to the identity; each identity syncs its own set at its first startup | Let the other identity start once, or run `/sync_emojis` there |
| A renamed icon left an old emoji behind | Report line "No longer in Types/ (kept, ...)" | /update_stats deletes only emojis listed in this bot's `type_emojis.json` entry, and never when the tree has no `Types/` icons; /sync_emojis without prune never deletes | `/sync_emojis prune:True` (and see **ttscardmaker-companion** §1.7 before renaming icons) |

---

## Verify the diff, not the commit message

This history contains joke and non-descriptive messages. Verified examples: 4e03190 "am forehead"
(2026-01-15) looks like noise but actually renamed the returned counter from `num_updated` to
`num_new`, changing the save-gate semantics of update_stats - a load-bearing change; "PICKLE." /
"PICKLE" (8753489, 284d13c, cca0aaf...) are data snapshots whose content you can only judge from
`git show --stat` byte deltas; and `/pull`'s own description string misdescribes its code (section 8).
When history matters, always run `git show <hash>` and read the diff. Curated incident narratives:
**autocroissant-failure-archaeology**.

---

## When NOT to use this skill

- Planning a parser/extraction improvement (not fighting a live symptom) -> **autocroissant-psd-extraction-campaign**;
  acceptance procedure for such changes -> **autocroissant-validation-and-qa**.
- Understanding the design ("why queues?", "where does new code go?", invariants) -> **autocroissant-architecture-contract**.
- The full story behind an incident hash, dead ends, reverts -> **autocroissant-failure-archaeology**.
- How to run/interpret the diagnostic scripts, golden outputs -> **autocroissant-diagnostics-and-tooling**.
- Committing, pushing, reverting, or any pickle/commit discipline question -> **autocroissant-change-control**.
- Installing/rebuilding the environment, Cython specifics -> **autocroissant-build-and-env**.
- Normal operations (start/stop/deploy/handoff/reminder ops) -> **autocroissant-run-and-operate**.
- What a config field/flag means or how to set it -> **autocroissant-config-and-flags**.
- Enabling/disabling AI, VRAM modes, torch install -> **autocroissant-ai-boundary** (branch `archive/ai-generation`).
- Card-domain semantics (folder classification, layer rules, query DSL) -> **impossibility-cards-reference**.
- The icon font, Types/ icon naming, the card website, Photoshop scripting -> **ttscardmaker-companion**.

---

## Provenance and maintenance

Written 2026-07-11 against the working tree (branch `main`, HEAD 284d13c) and discord.py 2.7.1.
Sections 1, 2 (reminder row), 3, 9 and 10 updated 2026-10-19 against HEAD a7a4bb3 plus the
uncommitted working tree; sections 1, 3, 4, 6, 8 and all line numbers refreshed 2026-10-20 (HEAD 9fb3093); sections 1,
2, 4, 5, 8 and the line numbers again 2026-10-23 (main 2ada357).
Every claim above was verified by reading the cited code or git output on that date. Line numbers
drift; before trusting one, re-find it:

```bash
# Section 1 - card data corruption
grep -n "^def gates\|^def can_prune" commands/card_data.py              # the build's gates and its prune gate
grep -n "def prune_clean_cards" commands/card_models.py                  # deletion handling
grep -n "def startup\|def sync\|Card data sync failed" commands/card_sync.py   # the bot's load and sync
git log --format='%h %ad %s' --date=short -1 eb9aa84                     # pickle-era revert precedent 2025-11-10
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py | tail -6   # the bot's card_data/

# Section 2 - dispatch/edit chains
grep -n "except (HTTPException, AttributeError)" main.py                 # both fallbacks (_dispatch, _edit)
grep -n 'kwargs.pop("ephemeral"\|dropped a message' main.py              # ephemeral dropped; per-item logs
grep -n "process_command_queue.stop()" main.py                           # 3 stop sites (171, 184, 226)
grep -n "Reconnected: inits skipped" main.py                             # 127: a reconnect revives failed loops
grep -n "def queue_message\|def queue_command" commands/utils.py         # append-only, never raise (~42, ~59)
grep -n "BREAK_LEN = " commands/utils.py                                 # 1950 (~19)

# Section 3 - slash sync
grep -c "@tree.command" main.py                                          # 47 as of 2026-10-25
grep -n "async def slash_set_r\|async def slash_delete_" main.py        # 4 distinct names (C5 fixed)
grep -n "clear_commands" commands/management.py                          # /sync_global mechanics (~86)
grep -c "on_guild_join" main.py                                          # 0 - no join-time sync

# Sections 4/7 - hangs and AI (diffusion.py exists only on branch archive/ai-generation)
git grep -n "in_progress = " archive/ai-generation -- commands/diffusion.py    # 127, 208, 255, 683, 782; no try/finally in diffusion()
git grep -n "Request queued" archive/ai-generation -- commands/diffusion.py    # ~670
git grep -n "No model configured" archive/ai-generation -- commands/diffusion.py   # ~203
grep -n "UPDATE_RATE" commands/card_data.py                              # progress every 25 cards

# Section 6 - music
grep -n 'browser_cookies\|cookiesfrombrowser' commands/music_player.py   # ~37, ~38, ~56
grep -n "FFmpegOpusAudio" commands/music_player.py                       # playback sources + volume limitation (~119, ~134, ~315)
grep -n "If stuck" commands/music_player.py                              # ~415

# Section 8 - self-update
grep -n "startup.sh\|execl" commands/update_bot.py                       # restart fallback (73, 75)
grep -n "checkout('--theirs'" commands/update_bot.py                     # theirs on every conflict
grep -n "'--rebase'" commands/update_bot.py                              # /pull divergent fallback

# Section 9 - reminders (2026-10-19 rewrite)
grep -n "skipped its command\|Reminders are off\|crashed the scheduler\|not found (deleted or no access)" commands/analytics.py   # user-visible messages
grep -n "Rescheduled reminder\|migrated command" commands/analytics.py  # startup log lines
grep -n "^CHANNEL_RETRY" commands/analytics.py
grep -n "queue_command(check_reminder)" main.py                          # fires only while loops run (928)
grep -n "reminder" .gitignore                                            # not synced across machines

# Section 10 - type emojis
grep -n "Emoji sync failed\|No longer in Types" commands/type_emojis.py
grep -n "discord" requirements.txt                                       # discord.py>=2.5

# tasks.loop stop-on-exception (discord.py version-dependent)
python3 -c "import discord; print(discord.__version__)"                  # 2.7.1 as of 2026-07-11
```

Volatile facts to re-check when they matter: all line numbers above; the 818/446/52 counts; the
local-timestamp re-archive hazard (docs/BUGS_AND_IMPROVEMENTS.md B6 - retired with local mode on 2026-10-20; the
section 1 row stays for pickles produced by the archive);
the dispatch loop's per-item guard (B5, fixed 2026-10-20). If a cited hash stops matching the described diff, re-run `git show <hash>` and fix
the story here rather than deleting it.

2026-10-21: the queue loops tick every 0.1 s (reminders keep a 1 s loop); main.py line numbers re-pointed (review: stop() ranges, _fallback_send, reminder loop wording).

2026-10-21 (titles): card_models.py and repo_traverser.py citations re-pointed (PARSER_VERSION 2: the first /update_stats after the deploy re-parses every card once, expected, not a fault).

2026-10-21 (review): repo_traverser.py citations land on their code (`update_stats` :472, history append :371-372, `_send_progress` :434, the end-of-run queue_commands :526-527).

2026-10-21 (S7): the registered-command count is 52 (/bake_cards); main.py and repo_traverser.py citations re-pointed. A "Card bake refused: this Python has no text engine" line was then the jetson venv without the fork's psd-tools. Since 2026-10-22 the jetson has the fork (tag cards-20261023 since 2026-10-23), so on the jetson that line means the fork install is broken: reinstall the tag (autocroissant-build-and-env §4).

2026-10-21 (hook removed): no bake follows /update_stats any more (the `notice_edits` hook is gone; the user: the Action is the one live runner, the jetson a manual fallback), so a "Card bake refused: this Python has no text engine" line can only follow /bake_cards. No cited line moved.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-22 (data build): /update_stats' traversal moved to the uncompiled commands/card_data.py; the history, save-gate,
progress and path greps re-pointed there (repo_traverser.py keeps update_stats and the follow-up queue_commands).

2026-10-23 (docs drift): B9's blob half fixed (0166948: a failed file drops the ETag; only a tree 5xx stays open);
healthy-pickle numbers at a7c310e (818 / 446 versions over 354 names / 2 archived) and the expected recovery growth;
rows for the one-time History recovery, "History cleanup skipped" and "History clone ... not read"; silent-loss mode
(3) names the reconnect revive; section 5 has a hard-crash (signal) row (ba27fb7, 9919a14, the pandas/datetime
scripts); section 8 has the /pull rescue rows and the "Nothing to push" reply; the recovery procedure uses
`diff_stats.py --history`; ~812 PSDs and the followed Action run after the cutover; every main.py, update_bot.py,
card_models.py and repo_traverser.py citation re-pointed at 2ada357.

2026-10-25 (retire): 47 registered commands (/update_metadata, /mass_replace, /alias, /bake_cards, /update_tts_save retired: branch archive/discord-edits).

2026-10-25 (step 10, work/pickles): section 1 rewritten for the card data (sync, data commits, gates, restore); the startup-unpickling and /update_stats-hung rows, section 8's /push rows and the re-verify lines follow; the pickle-era revert procedure lives in autocroissant-failure-archaeology Entry 3.
