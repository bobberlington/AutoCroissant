---
name: autocroissant-debugging-playbook
description: 'Symptom-to-triage playbook for live AutoCroissant failures. Load when something is BROKEN: "bot not responding" / "bot said nothing" / "message never arrived"; "stats look wrong" / "cards parsed wrong" / "everything is unknown type" / "old_stats ballooned" / pickle corruption; "slash command not showing up / stale / duplicated"; "command seems to hang" / "bot froze"; "bot won''t start" / "my edits don''t take effect"; "music silent" / "download fails" / yt-dlp errors / volume not working; "/ai stuck" / "Request queued forever" / OOM / "PyTorch is not installed."; "/update or /pull broke the bot" / merge conflict / bot never came back; "reminder didn''t fire" / scheduled command failed. Contains symptom-indexed triage tables with discriminating experiments, the dispatch/edit-queue fallback chains and their silent-loss modes, and the pickle-corruption git-revert recovery procedure. Deepest coverage: pickle corruption and Discord API quirks. Live triage only - design questions go to autocroissant-architecture-contract; incident history to autocroissant-failure-archaeology; self-update flow anatomy and message meanings to autocroissant-run-and-operate.'
---

# AutoCroissant Debugging Playbook

Symptom-indexed triage for the failure modes this bot actually has, verified against the code and
git history on 2026-07-11. Find your symptom's section, run the "First check", then use the
discriminating experiment to separate look-alike causes before touching anything.

Ground rules while debugging:

- The diagnostic scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`
  (`inspect_pickle.py`, `diff_stats.py`, `parse_one.py`, `gap_trace.py`, `dump_psd_layers.py`) are
  read-only and safe to run any time. Usage and golden outputs: **autocroissant-diagnostics-and-tooling**.
- Never debug by running `update_stats()` or writing a pickle "to see what happens" - the pickles ARE
  the database and are committed to git. Any fix that touches them routes through **autocroissant-change-control**.
- `config.py` is secret (tokens). Check that it exists; never print its values.
- All commands below assume cwd = repo root (`/Users/michaelsrouji/Desktop/AutoCroissant`).

Jargon used throughout: the bot drains three module-level deques once per second from `tasks.loop`
coroutines in `main.py` - `command_queue` (deferred function calls), `dispatch_queue` (message/file
sends), `edit_queue` (edits of a command's original response). Workers run in threads and only ever
call `queue_message` / `queue_file` / `queue_edit` / `queue_command` (commands/utils.py:38-62).
Design rationale lives in **autocroissant-architecture-contract**. A "PICKLE commit" is a commit
containing only pickle-file changes, created by the bot's own `/push` (commands/update_bot.py:22).

---

## 1. "Stats/cards look wrong after an update" (pickle corruption - costliest failure class)

The database is `stats.pkl` (dict name -> CardInfo) plus `old_stats.pkl` (archive of prior versions)
and `aliases.pkl`. `update_stats` (commands/psd_analyzer.py:1173) traverses the TTSCardMaker card
repo in one of two modes - local disk walk or GitHub trees API - reparses changed PSDs, archives the
old version into old_stats, and saves. Corruption here silently propagates to the other machine via
`/push`/`/pull`, so triage BEFORE pushing.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| Many/all cards suddenly "unknown" type, queries return nothing | `inspect_pickle.py` summary: "entries with suspicious (non repo-relative) paths" line | A local-mode `/update_stats` ran: the live removeprefix bug corrupts every path (see below) | Suspicious-paths count > 0 AND paths start with `Users/...` instead of `Creatures/...` etc. Also: the run's summary said hundreds of "cards changed location" | Do NOT `/push`. Recover via git revert (procedure below). Operating rule: always `/update_stats` with `use_local_repo:False` - see **autocroissant-run-and-operate**; bug story owned by **autocroissant-failure-archaeology** |
| One card's stats/ability wrong | `inspect_pickle.py "Card Name"` - is the DB record itself wrong? | Parser issue (gaps/type injection) vs query-layer issue | DB record correct but `/query` output wrong -> query layer (alias shadowing, fuzzy match at match_ratio 0.6, stale dataframes). DB record wrong -> parser | Parser root-causing recipes: **autocroissant-analysis-toolkit**; improving the parser: **autocroissant-psd-extraction-campaign** |
| old_stats.pkl ballooned | `git log --oneline --stat -- old_stats.pkl \| head -30` - look for byte jumps | Duplicate archiving: every examined card archived even when unchanged | Compare versions count in `inspect_pickle.py` summary (223 versions / 218 names as of 2026-07-11) against last known good | Guard exists since e7befd5 (2026-01-31: `should_update and not is_new`, psd_analyzer.py:938 and 1020). Regression precedent: cca0aaf ballooned old_stats 3196 -> 12041 bytes, reverted same-day by eb9aa84 (2025-11-10) |
| Cards vanished from stats after an update | `diff_stats.py` HEAD vs working (command below) - "removed" section | Traversal missed files (wrong path/mode), so `prune_clean_cards` (psd_analyzer.py:244-253) archived everything it didn't see | Removed cards you did not delete = traversal problem, not deletions | Revert procedure below; verify traversal mode before rerunning |
| Update ran but nothing saved / stats stale | Bot stdout for the summary line "N had newer timestamps or were new..." (psd_analyzer.py:1091) | Save is gated: pickles only written when `num_new > 0` (psd_analyzer.py:1222-1224) | num_new = 0 in summary -> gate worked as designed (nothing changed) | This gate IS the fix for empty-stats overwrites (fb47b5d + 4e03190 "am forehead", both 2026-01-15 — the second commit switched the gate to the counter that actually works); don't "fix" it away |
| Query results stale right after update | Did `populate_files`/`prep_dataframes` run? They are queued at the END of update_stats (psd_analyzer.py:1241-1242) | Command queue stopped before the refresh ran | `/query` a card you know changed; restart fixes it | Restart the bot; see section 2 for why queues stop |

### The traversal-mode discriminating experiment (run this first, always)

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py | grep suspicious
```

- `0` suspicious paths and a sane type distribution -> stats.pkl is healthy (as of 2026-07-11: 813
  cards, creature 297 ... rulebook 48, 0 suspicious).
- `> 0`, examples starting with `Users/` -> a LOCAL-mode traversal wrote to the db: the live
  removeprefix bug (psd_analyzer.py:1002 - still open while that grep hits). The signature:
  suspicious-paths count > 0, stored paths starting `Users/`, and the run's summary flooding with
  "cards changed location". The slash default is `use_local_repo=True` (main.py:361), so one bare
  `/update_stats` triggers it - operating rule: always pass `use_local_repo:False`
  (**autocroissant-run-and-operate**). Do NOT `/push`; recover via the git-revert procedure below.
  Root-cause story: **autocroissant-failure-archaeology** Entry 1; the code fix is a labeled
  CANDIDATE there and in **autocroissant-psd-extraction-campaign** Phase 0 - do not hot-patch
  during an incident.

Remote mode cannot produce this signature: its paths come straight from the GitHub trees API and are
repo-relative by construction (psd_analyzer.py:918-925).

### Recovery: reverting a bad PICKLE commit (eb9aa84 precedent)

Precedent: cca0aaf ("PICKLE", 2025-11-10) committed a corrupted snapshot; eb9aa84 reverted it the
same day and shrank old_stats.pkl from 12041 back to 3196 bytes. Procedure (gates and commit
discipline: **autocroissant-change-control**):

1. Find the bad commit: `git log --oneline -- stats.pkl old_stats.pkl aliases.pkl | head` and inspect
   byte deltas with `git show --stat <hash>`.
2. Confirm the commit touches ONLY the three pickles (`git show --stat <hash>`). If code changed too,
   stop - a plain revert would also revert code; go to **autocroissant-change-control**.
3. Snapshot what you have: `cp stats.pkl /tmp/stats_bad.pkl` (evidence for archaeology).
4. `git revert --no-edit <hash>`. Never `reset --hard` + force-push: the other machine may already
   have pulled the bad commit, and `/pull` auto-merge assumes shared history.
5. Verify the result before pushing anything:
   ```bash
   python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py | grep -E "cards:|suspicious"
   git show HEAD:stats.pkl > /tmp/stats_head.pkl
   python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
   ```
6. Restart the bot afterwards - the running process still holds the corrupted db in memory and its
   query dataframes are built from it.

Caveat: if legitimate updates landed AFTER the bad commit, a plain revert discards those too. cca0aaf
was caught same-day; the longer you wait, the more this becomes surgery (change-control's problem).
This is why `diff_stats.py` before every pickle push is the standing gate.

---

## 2. "Message never arrived / bot said nothing" (Discord API quirks - second costliest class)

The dispatch chain (main.py:832-854), drained once per second:

```
worker thread --queue_message/queue_file--> dispatch_queue --loop-->
  1. interaction.response.send_message(...)      if response not yet done
  2. else interaction.followup.send(...)         (attachments converted to files)
  3. on HTTPException/AttributeError:
       drop the 'ephemeral' flag (main.py:847)
       client.get_channel(interaction.channel_id).send(...)   <- plain channel message
```

The edit chain (main.py:858-871) tries `interaction.edit_original_response(...)`; on the same
exceptions it falls back to `channel.send(...)` - i.e. a NEW message instead of an edit.

Two Discord facts drive most of this: interaction tokens expire after 15 minutes (Discord-documented),
after which responses/followups/edits raise HTTPException; and fallback channel sends are ordinary
messages with no reply linkage, no ephemeral support.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| Bot says nothing, no error visible anywhere | Is the process alive? Then stdout for `Unhandled exception in internal background task` | Dispatch loop crashed, or all loops were deliberately stopped | Run a trivial command like `/help`: reply -> dispatch alive, your problem is worker-side; no reply -> loops dead | `/restart_bot` (works with dead loops: it replies via `interaction.response` directly and exec-restarts, main.py:149-155; note it has NO permission check) |
| Bot mute right after a failed `/update` | Discord for "Update failed:" + "Bot was NOT restarted due to errors." | `/update` stops all three loops BEFORE doing git work (main.py:200-202); on failure nothing restarts them | Those two messages arrive (sent via direct followup, not the queues) but nothing queued afterwards does | `/restart_bot`. Anything queued while loops were stopped is lost on restart (in-memory deques) |
| Reply arrived as a plain message; the slash command shows "The application did not respond" | How long did the worker run? | Work exceeded 15 min -> token expired -> fallback path 3 | Expected: reply content still arrives, unattached; the "did not respond" banner is cosmetic | Working as designed. Long jobs (full update_stats, big exports) always end like this |
| An ephemeral ("only you") reply became public or seems lost | Was the interaction old/expired? | Fallback pops `ephemeral` (main.py:847) - ephemeral cannot survive the channel.send fallback | Same reply within 15 min is ephemeral; after, it is public | Known weak point (listed in **autocroissant-architecture-contract**); don't rely on ephemeral for slow commands |
| Progress "edits" arrive as a stream of new messages | Which interaction were they attached to? | Edit-queue fallback sends new messages (main.py:864-871). Progress users: AI previews (diffusion.py:626,633) and update_stats "N cards updated." every 25 cards (psd_analyzer.py:1077-1083, UPDATE_RATE psd_analyzer.py:29) | Starts as edits, degrades to sends at the 15-min mark | Cosmetic; ignore |
| Nothing sent AND stdout shows a Python traceback | Read the traceback (often logged late as "Task exception was never retrieved") | The worker crashed before queueing; `queue_*` never raise (they just append, utils.py:38-62), so anything queued BEFORE the crash still arrives | Partial output followed by silence = mid-worker crash | Fix the worker bug; file the story in **autocroissant-failure-archaeology** |
| Everything mute after one specific message | stdout: `Unhandled exception in internal background task 'process_dispatch_queue'` | A send raised INSIDE the fallback handler, killing the loop. discord.py 2.7.1 stops a tasks.loop permanently on any exception outside its retry set (OSError/ConnectionClosed/etc.) | Verified crash vectors (code reading): channel deleted or not in cache -> `get_channel()` returns None -> AttributeError on `.send`; or content > 2000 chars -> HTTPException in the fallback too | `/restart_bot`. Prevention for new code: always route long text through `split_long_message` (BREAK_LEN=1950, utils.py:20,236) and never queue to channels the bot cannot see |
| Reminder fired but message missing | Reminder messages ALWAYS deliver via fallback path 3 (FakeInteraction -> plain Object -> AttributeError by design, utils.py:131-198) | Deleted/invisible target channel is the loop-killer above | Check stdout for the background-task crash line | Remove the stale reminder (`/list_reminders`, `/remove_reminder`) |

Silent-loss modes, summarized: (1) `queue_*` functions never raise to callers - a full deque with no
drainer is indistinguishable from success at the call site; (2) the three loops are stopped by
`/restart_bot`, `/stop_bot`, and `/update` (main.py:151-153, 163-165, 200-202) - a failed update
leaves them stopped; (3) a crashed loop stays crashed until process restart; (4) ephemeral flags are
dropped in fallback. The discriminating experiment is always the same: compare bot stdout (worker
prints, tracebacks, loop-crash lines) against what reached Discord.

---

## 3. "Slash command missing / stale / duplicated in Discord"

Sync model: `on_ready` copies global commands to each guild and syncs per guild
(main.py:97-101, 113-114) and prints `Synced commands to guild: <name> (<id>)` per guild plus
`Registered N slash commands for reminder system.` (54 registrations as of 2026-07-11 - a quick
sanity number). There is NO `on_guild_join` handler. Full ops detail: **autocroissant-run-and-operate**.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| New/changed command not in Discord after deploying code | Did the bot process restart since the code change? | Registration happens at import time; sync at on_ready | stdout has the "Synced commands to guild" lines with a fresh timestamp? | Restart the bot; per-guild sync applies within seconds |
| Command missing in ONE server only | Was the bot added to that guild after the last restart? | No on_guild_join -> new guilds get no sync until restart | Works in old guilds, missing in the new one | Restart, or `/sync_global` (admin) which also re-syncs every guild (commands/management.py:83-96) |
| Every command listed TWICE | Client shows two entries per name | Commands registered in both global and guild scope | `/sync_global` clears the global set then re-syncs guilds (management.py:86-93) | Run `/sync_global`; expect "Commands synced globally and to N guilds." |
| Stale parameter list / description in the client | Try in a different client (mobile vs desktop) | Discord client-side cache | Fresh client shows the new schema | Ctrl+R (desktop client reload); guild syncs are near-immediate, global scope can take up to ~1 hour (Discord-documented) |
| `CommandAlreadyRegistered` on startup | The failing command NAME in the traceback | Two `@tree.command` decorators with the same `name=` | - | Rename one; add-a-command checklist: **autocroissant-change-control** |
| "Duplicate functions in main.py - is that the bug?" | No. | `slash_set_ratio` is defined twice (main.py:303 and 313 - the second is `/set_repo`'s handler) and `slash_delete_song` twice (main.py:687 and 694 - the second is `/delete_all_music`'s) | Both commands work: the decorator registers the function object at def time under distinct command names; Python's silent rebinding of the module-level name only affects later by-name references | A wart, not a fault. Do not "fix" casually - **autocroissant-architecture-contract** documents it |

---

## 4. "Command seems to hang"

Nothing in this bot blocks the event loop by design: slash handlers `await` workers wrapped by
`to_thread`, and all sends go through the 1-second queue loops. "Hangs" are therefore one of four
things:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `/update_stats` "hung" | Progress messages: "N cards updated." should tick every 25 cards | It is just slow - a full parse of ~900 PSDs takes minutes of real CPU | Progress still ticking = alive. stdout heartbeat warnings ("heartbeat blocked for more than N seconds") during heavy parsing are GIL contention noise, not a deadlock by themselves | Wait. If progress stopped AND no traceback: check section 2 (dispatch dead ≠ worker dead) |
| Every command mute | `/help` test from section 2 | Queue loops stopped/crashed - all dispatch shares them | See section 2 | `/restart_bot` |
| `/ai` says "Request queued. You'll be notified when generation starts." forever | `/ai_queue` - is the queue only ever growing? | The `in_progress` flag (diffusion.py:125) is wedged True. `diffusion()` sets it at diffusion.py:681 and clears it at :780 with NO try/finally - any exception mid-generation (OOM, bad image URL, NoneType pipeline) wedges it permanently. (`init_pipeline` DOES clear it in a finally, diffusion.py:253) | stdout has a generation traceback from the first failed request; every later `/ai` only queues | `/restart_bot`. Only one generation runs at a time by design; a try/finally is an open candidate fix - route via **autocroissant-change-control** |
| Music download "hung" | stdout for a yt-dlp traceback | Downloads run on a dedicated 2-worker pool (music_player.py:36); two stuck downloads block the third | "Downloading: url" arrived but no "Queued song:" - the completion hook only fires on finish (music_player.py:254-256) | Section 6; update yt-dlp first |

---

## 5. "Bot won't start"

Healthy startup output, line by line, is owned by **autocroissant-run-and-operate**. Triage of
unhealthy starts:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | `ls config.py` | config.py missing (it is secret and never committed) | - | Recreate it; field names (values are secret): **autocroissant-config-and-flags** |
| `Fatal error: ...` then exit 1 | The wrapped exception text (main.py:877-884 catches everything around `client.run`) | `Improper token has been passed.` = TOKEN wrong/rotated | - | Fix TOKEN in config.py |
| `PrivilegedIntentsRequired` | Discord developer portal | `intents.message_content = True` (main.py:92) requires the Message Content intent enabled for the app | - | Enable the intent in the portal |
| Your code edits "don't take effect", or behavior is bizarrely old | `ls commands/*.so` | Stale compiled Cython artifacts shadow the edited .py at import time | Remove artifacts and rerun: behavior changes -> it was shadowing | `python3 setup.py clean` (custom CleanAll command). Build system detail: **autocroissant-build-and-env** |
| Startup traceback unpickling stats.pkl (`ModuleNotFoundError`/`AttributeError` naming `commands.psd_analyzer` or `CardInfo`) | Did anything rename/move CardInfo, CardStats, or psd_analyzer.py? | Pickles store class paths; `stats_db.load` only catches EOFError/FileNotFoundError (psd_analyzer.py:222-242), so schema breakage propagates. If the rename also broke main.py's imports the process dies instantly; otherwise the bot runs with broken stats (init_psd is queued, its crash is just a task traceback) | `git log --oneline -- commands/psd_analyzer.py` for recent renames | Restore the original names. Renames are a gated change class: **autocroissant-change-control** |
| `ImportError`/`ModuleNotFoundError` for cv2, psd_tools, discord, git, ... | `which python3`; is the right env active? | Wrong interpreter or missing requirements | - | **autocroissant-build-and-env** (requirements.txt = core; requirements2.txt = AI, reorganized 2026-07-11) |

---

## 6. "Music silent / download fails"

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| "Downloading: url" then nothing | stdout for a yt-dlp traceback | YouTube churn breaking yt-dlp (the perennial cause) | Run the same URL with the standalone `yt-dlp` CLI | `pip install -U yt-dlp` FIRST, before debugging anything else (it is unpinned in requirements.txt on purpose). Then retry |
| Download errors mentioning cookies / Safari | Are you on the Mac? | yt-dlp uses Safari's cookies ONLY when `vram_usage == "mps"` (`cookies_from_browser`, music_player.py:37, applied at :55) - vram_usage doubles as a Mac switch | Same URL fails on Mac, works on the CUDA box (or vice versa for age-gated videos that NEED cookies) | Terminal needs macOS disk access to read Safari cookies. Flag catalog: **autocroissant-config-and-flags** |
| "Now playing: **song**" but silence | stdout for `ClientException` ("ffmpeg was not found") or a voice error | ffmpeg missing, or voice deps (PyNaCl/opus) missing | The "Now playing" message is queued BEFORE the actual `vc.play` runs (music_player.py:126-137), so it proves nothing about playback | Install ffmpeg; env checklist: **autocroissant-build-and-env** |
| `/volume` refuses | The exact reply | "Volume control not available with FFmpegOpusAudio. Switch to FFmpegPCMAudio to adjust volume." (music_player.py:313-314) - all playback uses FFmpegOpusAudio (music_player.py:118,133) | - | By design; not a bug |
| Bot thinks it is/isn't in voice ("stuck") | `/disconnect` reply | `state.vc` (music_player.py:44) out of sync with reality | Reply "Not currently in a voice channel. If stuck, use `/play` then `/disconnect`." (music_player.py:414) is the built-in escape hatch | Do exactly that: `/play` (reconnects), then `/disconnect` (resets state) |
| stdout: `TypeError: object NoneType can't be used in 'await' expression` in init_vc | Was the user outside a voice channel? | Known wart: `await queue_message(...)` on a sync function (music_player.py:177). The message is queued before the await explodes, so the user still sees "You are not in a voice channel." | - | Harmless stdout noise; don't chase it. Fix is a candidate via **autocroissant-change-control** |
| Song skipped with "File not found: path" | Was music/ cleaned between queueing and playing? | Queue holds paths, not files (music_player.py:127-129); it auto-advances | - | Re-queue |

---

## 7. "AI generation fails / OOM"

VRAM modes, model routing, and enable/disable runbooks are owned by **autocroissant-ai-boundary**.
Triage layer only:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| `/ai` replies "PyTorch is not installed." | Intentional? | torch absent = the AI boundary doing its job (diffusion.py:659-660); core bot runs fine without it | `python3 -c "import torch"` in the bot's env | Enable via requirements2.txt if wanted: **autocroissant-ai-boundary** |
| stdout: `INFO: No model configured for initialization` | config.py `model` field empty | `init_pipeline` returns without building pipelines when model is '' (diffusion.py:200-202) | TRAP (verified code path): with torch installed but model empty, `/ai` prints "Initializing AI pipeline...", init does nothing, then the NoneType pipeline call crashes and WEDGES `in_progress` (section 4) | Set a model (`/set_model` or config.py), `/restart_bot`, retry |
| OOM / dtype / device errors mid-generation | stdout traceback | Wrong VRAM mode for the machine | - | Mode table: **autocroissant-ai-boundary**. Then `/restart_bot` - the failed run wedged `in_progress` (diffusion.py:681 vs :780, no try/finally) |
| "Request queued" but never starts | `/ai_queue` | Wedged `in_progress` | See section 4 row | `/restart_bot` |

---

## 8. "Self-update broke" (/push, /pull, /update)

The canonical anatomy of the flow, normal operation, and the full failure-message table (what
each message means) live in **autocroissant-run-and-operate** §5. `force_reset` danger and
commit discipline: **autocroissant-change-control**. This section is the triage layer only
(first check / discriminating experiment per symptom); message strings verbatim from
commands/update_bot.py:

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| "Pickles are already up to date." | - | Nothing to commit (update_bot.py:26-27) | - | Benign |
| "Push failed: ..." then "Update failed: ..." and "Bot was NOT restarted due to errors." | The git error text | Push rejected (auth, remote ahead, ...). `/update` had ALREADY stopped the queue loops (main.py:200-202) | Bot is now half-dead: direct followups worked, queued sends won't | `/restart_bot`, then fix the git issue manually on the box |
| "Merge conflict detected. Resolving automatically..." | Following messages list per-file decisions | Divergent histories (both machines committed) | Expected resolution: "Keeping local version of <pickle>" and "Accepting remote version of <code>" - pickles ours, code theirs (update_bot.py:63-79), committed as "AUTO-RESOLVE: ..." | Verify pickles afterwards with `diff_stats.py`; the local pickles won - if the REMOTE ones were the good ones, you just kept bad data |
| "Git pull failed: ... divergent branches" then "Attempting to reconcile divergent branches..." | Next message | Fallback `git pull --rebase` (update_bot.py:92-101) | "Rebase successful!" = recovered; "Manual intervention may be required." = it is | SSH to the machine and resolve by hand |
| Update said "Restarting bot!" but the bot never came back | Is `./startup.sh` present on that machine? | `restart_bot` execs `./startup.sh`, falling back to `execl(python, main.py ...)` (update_bot.py:9-13). startup.sh exists only on deploy machines, not in the repo. The execl fallback re-runs the same interpreter: if the pull brought NEW dependencies, the boot then dies on ImportError | Check the process and its stdout on the box | Install deps (**autocroissant-build-and-env**), start manually (**autocroissant-run-and-operate**) |
| "Warning: All local changes have been discarded!" | Did someone pass `force_reset:True`? | `git_reset_hard` ran (update_bot.py:104-119) - local pickles are GONE, replaced by remote | `diff_stats.py` remote-vs-backup if you have one | Damage control via **autocroissant-change-control**; the pickles on the remote are now the only truth |
| "/pull did a hard reset?!" | It did not | `/pull`'s description string says "Does a hard reset, than a git pull" (main.py:172) but `git_pull` contains no reset - fetch + merge/rebase only | Read update_bot.py:33-101 | Stale description; doc-of-record fixes belong to **autocroissant-docs-and-style** |

---

## 9. Reminder quirks

Reminders live in `reminder.pkl` (gitignored - NOT synced across machines by `/push`; they die with
the machine, see **autocroissant-run-and-operate** for handoff implications). `check_reminder` is
queued every second by the command loop (main.py:825), so reminders only fire while the loops run.

| Symptom | First check | Likely cause | Discriminating experiment | Fix / pointer |
|---|---|---|---|---|
| Scheduled command misparsed / wrong args | The stored command string (`/list_reminders`) | Command kwargs must be `key:value` with NO space after the colon. `set_reminder` auto-repairs exactly the form `key: value` (`command.replace(': ', ':')`, analytics.py:136); any other spacing around the colon splits wrong in `parse_named_args` (utils.py:277-290) | `key : value` -> "key" becomes a positional arg and ":value" a kwarg with empty name | Rewrite the reminder as `key:value` |
| Reminder fired at the wrong hour | What timezone did you assume? | Times are PST/PDT: `TIMEZONE = ZoneInfo("America/Los_Angeles")` (global_config.py), parsed as `13:00`, `1PM`, or `1:30PM` (analytics.py:65-84) | - | State times in PST |
| Repeating reminder "skipped" runs while bot was down | stdout at startup: `Rescheduled reminder <id> from ... to ...` | Past-due repeating reminders fast-forward to the next future slot at startup (analytics.py:47-58; added by 105198f, 2025-12-02). Past-due ONE-SHOT reminders instead fire immediately on the first tick | - | By design |
| Scheduled `/update_stats` corrupted stats | The reminder's args | slash_registry maps the command name to the SLASH wrapper (main.py:109-111), so a bare scheduled `/update_stats` inherits the slash defaults - including `use_local_repo=True` (main.py:361) = the live corruption bug (section 1) | - | Any scheduled update_stats MUST include `use_local_repo:False`. Operating rule: **autocroissant-run-and-operate** |
| "Unknown command: `x`" from a reminder | Exact spelling vs `/help` | Name not in slash_registry (lookup at analytics.py:209) | - | Use the slash command's exact name, no typos; kwarg names must match the wrapper's parameter names |

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
- Enabling/disabling AI, VRAM modes, torch install -> **autocroissant-ai-boundary**.
- Card-domain semantics (folder classification, layer rules, query DSL) -> **impossibility-cards-reference**.

---

## Provenance and maintenance

Written 2026-07-11 against the working tree (branch `main`, HEAD 284d13c) and discord.py 2.7.1.
Every claim above was verified by reading the cited code or git output on that date. Line numbers
drift; before trusting one, re-find it:

```bash
# Section 1 - pickle corruption
grep -n 'removeprefix("TTSCardMaker")' commands/psd_analyzer.py          # live bug (expect ~1002)
grep -n "use_local_repo: Optional\[bool\] = True" main.py                # slash default True (~361)
grep -n "should_update and not is_new" commands/psd_analyzer.py          # e7befd5 guard (~938, ~1020)
grep -n "num_new > 0" commands/psd_analyzer.py                           # save gate (~1222)
grep -n "def prune_clean_cards" commands/psd_analyzer.py                 # deletion handling (~244)
git log --format='%h %ad %s' --date=short -1 eb9aa84                     # revert precedent 2025-11-10
git show --stat cca0aaf | tail -4                                        # old_stats 3196 -> 12041
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py | tail -6
#   expected as of 2026-07-11: 813 cards, 223 versions/218 names, 0 suspicious paths

# Section 2 - dispatch/edit chains
grep -n "except (HTTPException, AttributeError)" main.py                 # both fallbacks (~846, ~864)
grep -n 'send_kwargs.pop("ephemeral"' main.py                            # ephemeral dropped (~847)
grep -n "process_command_queue.stop()" main.py                           # 3 stop sites (~151, ~163, ~200)
grep -n "def queue_message\|def queue_command" commands/utils.py         # append-only, never raise (~43, ~60)
grep -n "BREAK_LEN = " commands/utils.py                                 # 1950 (~20)

# Section 3 - slash sync
grep -c "@tree.command" main.py                                          # 54 as of 2026-07-11
grep -n "async def slash_set_ratio\|async def slash_delete_song" main.py # duplicate names (2 hits each)
grep -n "clear_commands" commands/management.py                          # /sync_global mechanics (~86)
grep -c "on_guild_join" main.py                                          # 0 - no join-time sync

# Sections 4/7 - hangs and AI
grep -n "in_progress = " commands/diffusion.py                           # 125, 206, 253, 681, 780; no try/finally in diffusion()
grep -n "Request queued" commands/diffusion.py                           # ~668
grep -n "No model configured" commands/diffusion.py                      # ~201
grep -n "UPDATE_RATE" commands/psd_analyzer.py                           # progress every 25 cards (~29)

# Section 6 - music
grep -n 'cookiesfrombrowser\|== "mps"' commands/music_player.py          # ~37, ~55
grep -n "FFmpegOpusAudio" commands/music_player.py                       # playback sources + volume limitation (~118, ~133, ~314)
grep -n "If stuck" commands/music_player.py                              # ~414

# Section 8 - self-update
grep -n "startup.sh\|execl" commands/update_bot.py                       # restart fallback (~11, ~13)
grep -n "checkout('--ours'\|checkout('--theirs'" commands/update_bot.py  # pickles ours / code theirs (~69, ~72)
grep -n "'--rebase'" commands/update_bot.py                              # divergent fallback (~96)

# Section 9 - reminders
grep -n "replace(': ', ':')" commands/analytics.py                       # colon rule (~136)
grep -n "Rescheduled reminder" commands/analytics.py                     # fast-forward (~54)
grep -n "queue_command(check_reminder)" main.py                          # fires only while loops run (~825)
grep -n "reminder.pkl" .gitignore                                        # not synced across machines

# tasks.loop stop-on-exception (discord.py version-dependent)
python3 -c "import discord; print(discord.__version__)"                  # 2.7.1 as of 2026-07-11
```

Volatile facts to re-check when they matter: all line numbers above; the 813/223/54 counts; the
requirements split (reorganized 2026-07-11, intentionally uncommitted at time of writing); the
removeprefix bug's status (if `grep removeprefix commands/psd_analyzer.py` comes back empty, the bug
was fixed - retire section 1's local-mode warning and update **autocroissant-run-and-operate**'s
operating rule). If a cited hash stops matching the described diff, re-run `git show <hash>` and fix
the story here rather than deleting it.
