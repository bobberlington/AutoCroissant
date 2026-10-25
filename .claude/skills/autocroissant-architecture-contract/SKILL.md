---
name: autocroissant-architecture-contract
description: Load this when you need to understand or safely modify HOW AutoCroissant works as a system — "how does the bot work", "start here", "orient me", "overview of this repo", "which skill do I need" (newcomer routing map at the top), "where do I put new code", "why is this function sync", "design", "invariants" — or any task touching threading, the async event loop, the command/dispatch/edit queues, tasks.loop, to_thread, queue_message/queue_file/queue_edit/queue_command, slash-command registration, slash_registry, reminder command execution, reminders running as their creator / lacks_admin at fire time, FakeInteraction, _ReminderUser, async workers that await Discord (type_emojis), module import order/cycles, module-level singletons, or the card data (JSON on TTSCardMaker's data branch; the pickles-in-git database it replaced on 2026-10-25). Contains the execution model (discord.py event loop + three module-level deques whose loops wake on enqueue + sync workers in threads) and WHY it exists; an invariants table where each row says what breaks if you violate it; the module dependency map including the deliberate lazy-import cycle break; ASCII data flows for a slash command, the /update_stats pipeline, and a reminder firing; and the known-weak points stated plainly (ephemeral-dropping fallback, local-mode timestamp mismatch, pickle trade-offs, cross-repo icon contracts; duplicate names and the dying dispatch loop fixed 2026-10-20; the 1 s tick and in-place pickle saves 2026-10-21). Live-symptom triage belongs to autocroissant-debugging-playbook.
---

# AutoCroissant Architecture Contract

## New here? The 17-skill map

- Run / stop / deploy / machine handoff / run update_stats → autocroissant-run-and-operate
- Something is broken RIGHT NOW → autocroissant-debugging-playbook
- Install / env / Cython build → autocroissant-build-and-env ; every config axis → autocroissant-config-and-flags
- BEFORE any commit, push, or card-data touch → autocroissant-change-control ; is it safe/finished → autocroissant-validation-and-qa
- What cards/PSDs/queries mean → impossibility-cards-reference ; measuring tools → autocroissant-diagnostics-and-tooling
- How the game is played (timing tags, damage, status effects) → impossibility-simulator-rules
- Improving the parser → autocroissant-psd-extraction-campaign ; proving things → autocroissant-analysis-toolkit
- "Has this been tried?" / why is this code weird → autocroissant-failure-archaeology
- AI/diffusion: removed from main 2026-10-20; code and its skill are on branch `archive/ai-generation` ; future ideas → autocroissant-research-frontier
- "How do I know I'm right" → autocroissant-research-methodology ; help text / style / skill upkeep → autocroissant-docs-and-style
- The TTSCardMaker side (icon font, Types/ icons, card website, Photoshop scripting) → ttscardmaker-companion
- THIS skill: how the system works + the invariants. Read "The execution model" before writing code.

Facts and line numbers verified 2026-07-11 (HEAD `284d13c`); re-verified 2026-10-20 against
`main` HEAD `9fb3093`, after that day's optimization pass (module split, robustness fixes), and again on 2026-10-23
at `2ada357`.
Line numbers drift; every load-bearing one has a re-find grep in "Provenance and maintenance" at the end.

This skill is the contract: the design decisions that everything else leans on, why they were made,
the invariants you must not break, and the weak points the codebase actually has. If you are about to
write or move code in this repo, read "The execution model" and "Invariants" before touching anything.

---

## The execution model in one paragraph

`main.py` runs ONE discord.py client on ONE asyncio event loop. Every slash command's real work is a
plain **sync** function in `commands/*.py`, pushed off the loop into a worker thread via
`asyncio.to_thread`. Worker threads are forbidden from touching Discord directly; instead they append
their output to three module-level deques in `commands/utils.py` (`command_queue`, `dispatch_queue`,
`edit_queue`, utils.py:26-28). Three `tasks.loop` coroutines in `main.py` (`process_command_queue` main.py:824,
`process_dispatch_queue` main.py:848, `process_edit_queue` main.py:885) drain those deques **on the event loop** the
moment something is queued (2026-10-25: each `queue_*` wakes its loop, below; every 0.1 s from 2026-10-21, once a
second before) and perform the actual Discord I/O, with a built-in fallback chain; a fourth loop,
`process_reminders` (main.py:806-808), keeps the reminder heartbeat at 1 s. That is the whole model: loop owns
Discord, threads own work, deques are the only bridge.

## The queue trio (this skill owns this detail)

| Queue (utils.py:26-28) | Entry shape | Filled by | Drained by | What draining does |
|---|---|---|---|---|
| `command_queue` | `((args, kwargs), func)` | `queue_command(func, *a, **kw)` (utils.py:93) | `process_command_queue` (`_drain_commands`, main.py:812-826) | `create_task(run_command(...))` per entry — each func runs in a worker thread of asyncio's default pool (threads are reused: 2 alive at idle, measured 2026-10-20), or in a given threadpool (see below) |
| `dispatch_queue` | `(interaction, send_kwargs)` | `queue_message` (utils.py:75), `queue_file` (utils.py:81), `queue_any` (utils.py:69) | `process_dispatch_queue` (`_drain_dispatch`, main.py:835-850: one ordered lane task per channel, `_drain_lane`; `_dispatch` 765-779, which also turns a `links` kwarg into link buttons) | Sends via fallback chain: `interaction.response.send_message` → `interaction.followup.send` → `client.get_channel(channel_id).send` |
| `edit_queue` | `(interaction, edit_kwargs)` | `queue_edit` (utils.py:87) | `process_edit_queue` (`_drain_edits`, main.py:875-887; `_edit` 782-798) | `interaction.edit_original_response`, falling back to a fresh `channel.send` |

Mechanics you must know:

- **Wake-ups (B-1, 2026-10-25).** Each deque has a `Wake` (utils.py:35-63: `command_wake`, `dispatch_wake`,
  `edit_wake`); every `queue_*` appends, then calls `wake.set()`, which does `loop.call_soon_threadsafe(event.set)`, so a
  worker thread wakes the loop without touching it (skipped while the event is already set, so a burst costs one wake).
  Each loop is `@tasks.loop(seconds=0)`: drain, then `await wake.wait()`, which returns on the wake or after `FALLBACK`
  (1 s, utils.py:32) and clears the event before the next drain, so an item queued during a drain wakes the next wait.
  A missed wake-up is only slower (the 1 s poll), never lost. Code that appends to a deque directly must call its
  wake too (`_edit` queues its extra chunks with `queue_any`). `_stop_loops()` (main.py:895) stops the four loops and
  sets the three wakes, so a waiting drain ends at once (as with the old tick, a drain already between iterations runs once more). The `Wake` binds to the loop of its
  first `wait()` (a new loop, as in each test's `asyncio.run`, gets a new Event); before that, `set()` does nothing.
  Order within a queue and rate-limit handling are unchanged. Tests drain by calling `_drain_*` directly.
- **`to_thread(func)`** (utils.py:116-125) wraps a sync function so awaiting it runs it in a thread via
  `asyncio.to_thread`. Critical subtlety: if `func` is already a coroutine function it is returned
  **unchanged** (utils.py:118-120). This passthrough is what lets `run_command` (main.py:716-724) execute
  both sync workers and the async slash callbacks stored in `slash_registry` with one code path.
- **`to_threadpool(func, executor)`** (utils.py:128-137) is the bounded variant. `run_command` routes to
  it when the queued kwargs contain `executor` (main.py:719-720) — so **`executor` is a reserved kwarg
  name** for anything sent through `queue_command`; a worker function with its own `executor` parameter
  would have it silently popped. Sole current user: music downloads, capped at a 2-worker
  `ThreadPoolExecutor` (music_player.py:35, used at music_player.py:264 and 277) so a playlist can't
  spawn unbounded yt-dlp threads.
- **Queued commands run concurrently** (one `create_task` + thread per entry), so two workers that must not overlap
  need their own lock: card_sync's `_sync_lock` (one sync at a time) and `_run_lock` (one followed Card sync run), the
  reminders' `_check_lock`. The bot never bakes or parses PSDs: card-sync.yml is the one runner (`/bake_cards` retired
  2026-10-25; the bot's own /update_stats parse and its `bake_lock` went at step 10).
- Plain `deque` + append/popleft is safe here without locks because CPython deque append/popleft are
  atomic, producers are threads, and the only consumer is the event loop.
- The `async def slash_*` wrappers in main.py ARE allowed to await Discord directly (`defer()`,
  permission denials) — they run on the event loop. The queue discipline binds the **sync worker
  functions** in `commands/*.py`.
- **Sanctioned async workers** (they run ON the event loop because `to_thread` passes them through):
  `commands/update_bot.py` (invariant 3), parts of `management.py`, and since 2026-10-19
  `commands/type_emojis.py`. `sync_type_emojis` awaits Discord's application-emoji API directly
  (`client.fetch_application_emojis`, `create_application_emoji`, `emoji.delete`) and pushes its
  blocking work — the GitHub listing and every image download/resize — into threads with
  `asyncio.to_thread`. Its user-facing report still goes through `queue_message`. Copy that split if
  you write another async worker: await only cheap Discord calls on the loop, thread everything else.
- **The /reminders panel answers directly** (2026-10-24, `commands/reminder_panel.py`, not compiled): a View, a
  modal and `edit_message` need the live interaction, so its callbacks call `interaction.response.*` themselves
  (3 s to answer) and run every read or change of the reminders in a thread (`to_thread(snapshot | set_enabled |
  delete | reschedule | create)`: `_lock` may be held by a scheduler pass). Rights are checked in analytics.py under
  the lock, not only by the disabled buttons. Views are built on the loop, never in a worker thread.

### WHY this design (the load-bearing reasons)

1. **Thread safety.** discord.py's client and event loop are not thread-safe. A worker thread that
   calls `interaction.response.send_message(...)` gets a coroutine object it cannot await — the message
   silently never sends — and any attempt to drive the loop from the thread risks corrupting it. The
   queues mean worker code never needs the loop at all.
2. **Centralized fallbacks survive interaction-token expiry.** Discord interaction tokens die after
   15 minutes. Long jobs (full `/update_stats` sweep, playlist download) outlive them.
   The dispatch loop's chain — `response.send_message` if not yet responded, else `followup.send`, and
   on `HTTPException`/`AttributeError` a raw `channel.send` (`_fallback_send`, main.py:750-762) — means a late result
   degrades to a plain channel message instead of vanishing. Every send written as `queue_*` gets this
   for free; every send written by hand does not.
3. **Serialization + self-throttling.** One drain point per queue (woken on enqueue since 2026-10-25; every 0.1 s
   from 2026-10-21, 1 s before) keeps each queue's order; discord.py's own rate-limit handling covers the rest without
   explicit rate-limit code. Cost since the wake-ups: about 1 ms per hop (simulated 2026-10-25: reply hop mean
   0.4-1.0 ms, p95 1-3 ms; a reminder's two hops about 1 ms; the 0.1 s tick gave 50 ms mean / 95 ms p95 per hop and
   150 ms for two hops).
4. **Deferred init without blocking login.** `on_ready` queues five commands instead of running them:
   `init_reminder`, `init_psd` (the card data: `card_data/`, then a sync), `init_query`, `refresh_history`,
   `init_type_emojis(client)`. (Until 2026-10-25 a `sweep_temp_psds` of the parse's temp PSDs ran first.) The card data
   load and the GitHub tree fetch happen in worker threads after the bot is already up. `init_type_emojis` is async: it runs on the loop and threads
   its own downloads. The inits run **once** (the `_started` flag, 778c2ce, 2026-10-21): discord.py calls `on_ready`
   again after a failed RESUME, and that repeat only syncs any guild not synced yet, restarts the loops whose task
   failed and prints `Reconnected: inits skipped, revived <loops>` (main.py:125-128). Loops stopped on purpose (by
   `/update`) stay stopped. main.py also calls `bind_reminder_client(client)` at import time (main.py:236) so the
   scheduler can tell when a reminder's channel is gone.

## Invariants — violate this and X happens

| # | Invariant | Violate it and... |
|---|---|---|
| 1 | **No awaits inside queued sync functions.** Workers are plain `def`; there is no event loop in their thread. | `await` in a `def` is a SyntaxError — so the tempting "fix" is making the worker `async def`. Do that and `to_thread` passes it through unchanged (utils.py:118-120): your "worker" now runs ON the event loop, and its blocking calls (PSD parse, yt-dlp, GitPython) freeze the entire bot — heartbeats, all three queue loops, everything — until it finishes. |
| 2 | **All Discord output from worker code goes through `queue_message`/`queue_file`/`queue_edit`/`queue_any`/`queue_command`** (utils.py:69-96). | Direct calls from a thread produce never-awaited coroutines (message silently lost) or thread-unsafe loop access. You also forfeit the fallback chain: your send dies permanently at the 15-minute token expiry instead of degrading to `channel.send`, and reminders (which run your command with a FakeInteraction) break because FakeInteraction only implements the queue-facing surface. |
| 3 | **The three queue loops must be running for ANYTHING to send.** Started once in `on_ready` through `_start()` (main.py:143-144; `_loops()`/`_stop_loops()`/`_start()` at main.py:890-911); a repeat `on_ready` (reconnect) restarts only the loops whose task failed. Deliberately STOPPED through `_stop_loops()` (which also wakes the waiting drains, so the stop is immediate) by `/restart_bot` (main.py:207), `/stop_bot` (main.py:217), `/update` (main.py:242) so no queued output races process replacement; a reconnect does not revive those. | With a loop stopped (or `on_ready` never reached), `queue_*` calls accumulate silently in the deques forever — the bot looks alive and says nothing. This is also why `commands/update_bot.py` is the one module that bypasses the queues entirely: its functions are `async def` awaiting `interaction.followup.send` directly (`git_pull` update_bot.py:33, `update_bot` :119), because during `/update` the dispatch loop is already stopped and queued progress reports would never flush. |
| 4 | **Registration pattern:** `func = to_thread(func)` rebind FIRST, then the `@tree.command` async wrapper that awaits it (43 occurrences as of 2026-10-25). `on_ready` stores `slash_registry[cmd.name] = cmd.callback` (main.py:129-130), which powers reminder command execution (analytics.py `resolve_command`, `_fire`). | Forget the rebind line → the wrapper's `await func(...)` executes the sync function inline ON the event loop (bot frozen for its whole duration), then raises `TypeError` awaiting the plain return value. Since 2026-10-19 reminders bind their stored `/name option:value` text against the callback's real signature at creation AND at every firing, so: rename a command's `name=` → saved reminders post "⚠ Reminder `<id>` skipped its command: `/<name>` is not a bot command" (a command removed on purpose fails the same way; the "was retired" shim was removed 2026-10-25); rename or remove a slash **parameter** → "`<option>` is not one of its options" (the option-dropping shim, used for /update_stats's `use_local_repo`/`use_local_timestamp`, was removed 2026-10-25); change a parameter's type → values that no longer convert are refused. The reminder itself keeps repeating; its creator must re-create it. The wrapper function's Python name, by contrast, is decorative — see weak point 1. |
| 5 | **One bot per process; state is module-level singletons.** `stats_db` (card_models.py:409, re-exported by psd_analyzer), `card_repo` (query_card.py:990), `state`/`music_queue`/`prev_music`/`executor` (music_player.py:32-45), `reminders` + `_client` + `_lock`/`_check_lock` (analytics.py:61-66), `type_emojis` (type_emojis.py:31, token → emoji map filled by the sync), `slash_registry` + the three deques (utils.py:26-30). | A second client in-process, or importing a commands module under a second name, gives you split-brain state. Import order is load-bearing: the singletons are constructed at import time (cheap — heavy loading is deferred to the queued `init_*` functions), and (since 2026-10-20) none of them prints: the single git-token-presence line comes from `print_token_status()` in main.py. |
| 6 | **Importing `commands.*` requires `config.py` to exist.** utils and music_player `import config` at module level and github_api reads it in a try/except (so `python -m commands.tts_save` runs without it); the facade reaches utils: psd_analyzer → query_card → type_emojis → utils. | Any tool importing the facade or main needs a config.py present (the tests supply a fake). The card data needs none: card_db, card_models and card_data import no config (the Card sync Action runs without one), and the diagnostics read db/ JSON and historical pickles through card_db's allow-list unpickler. config.py is SECRET — never print or read its values. |
| 7 | **`check_reminder` is re-queued every second — reminder-path work must stay cheap, and stay single-flight.** `process_reminders` (main.py:806-808, a 1 s `tasks.loop` started and stopped with the drains) does `queue_command(check_reminder)`; the woken command loop starts it at once, so it runs ~once/second in a pool thread (analytics.py:819). Since 2026-10-19 it takes `_check_lock` with a non-blocking acquire (a pass that finds another still running just returns) and does all work under `_lock` (an RLock shared with the slash shortcuts and the /reminders panel, which run in their own threads). Commands of due reminders are queued onward (`queue_command(_run_reminder_command, ...)`), never run inline; reminder.json is saved only when something changed, atomically. | Anything slow in `check_reminder` costs a pool thread every second and holds `_lock`, stalling `/set_reminder`, `/reminders` and `/remove_reminder`. Remove the single-flight lock and two overlapping passes can fire the same reminder twice. Run a reminder's command inline and one slow command blocks every other reminder. |
| 8 | **`lacks_admin` (named `perms_check` until 2026-10-20) returns True when the user is NOT an admin:** `return interaction.user.id not in ADMINS` (utils.py). Callers write `if lacks_admin(i): deny-and-return` (e.g. main.py:180-182). | Flip the polarity and you either lock the admins out or open an admin command to everyone. When adding an admin-gated command, copy an existing gate verbatim. |
| 9 | **Admin gates live in the registered slash callback, and reminders fire as their creator** (2026-10-19). A firing reminder's FakeInteraction carries `_ReminderUser(creator_id)` (analytics.py:95, set in `_reminder_interaction`), so the callback's own `lacks_admin` re-checks ADMINS at fire time. Legacy reminders without a creator keep the default FakeUser — the bot itself (`LEGACY_RUNS_AS_BOT`, analytics.py:37). | Put an admin check anywhere other than the callback stored in `slash_registry` (e.g. only in a worker, or in a pre-dispatch hook) and scheduled runs either skip it or check the wrong user. Make the FakeInteraction default to the bot again and every reminder passes admin checks (the bot's id is in ADMINS on the dev Mac — archaeology Entry 10.8). |
| 10 | **The bot reads card data and never writes it: one sync at a time (`card_sync._sync_lock`), the database swapped whole.** No bot module imports card_bake, psd_edit or card_data's build (2026-10-25: the legacy /update_stats parse, its `bake_lock` and the pickles are gone). | Write card data from the bot and there are two writers (Card sync is the one, gate G6). Import card_bake at module level and the bot fails to start wherever the fork's psd-tools is absent (the Mac's stock venv). |

## Data-flow diagrams

### (a) A slash command, end to end

```
Discord user runs /query
      |
      v  (event loop, discord.py)
tree dispatch -> async wrapper slash_query_name (main.py:289-292)
      |   await query_name(...)        # query_name was rebound: to_thread(query_name), main.py:283
      v
asyncio.to_thread -> WORKER THREAD runs sync query_name()
      |   does the work; never awaits; never sends
      |   queue_message(interaction, ...) / queue_file(...)      # appends to dispatch_queue
      v
dispatch_queue (deque, utils.py:26)
      |
      |   ... and dispatch_wake.set(): call_soon_threadsafe wakes the loop
      v  (at once, back on the event loop; 1 s poll if a wake were missed)
process_dispatch_queue (main.py:848) -> the channel's lane
      |- interaction.response.send_message(**kw)     if response not yet done
      |- interaction.followup.send(**kw)             if already responded/deferred
      '- client.get_channel(channel_id).send(**kw)   on HTTPException/AttributeError
      |                                              (token expired, or fake interaction;
      v                                               NOTE: drops `ephemeral` — weak point 3)
Discord
```

### (b) The card data build (TTSCardMaker's Card sync Action, on the runner)

```
card push / website edit / schedule / an admin's dispatch
      v
card-sync.yml: python -m commands.card_data (this repo's builder, checked out sparse: card_data.SPARSE_FILES)
      |  db/ before (the data branch), the tree and index from git (GitBuild: ls-tree, log --raw; no commits API)
      |  changed PSDs only (new, moved, another blob, an older PARSER_VERSION, cards, full): blobs from the clone, else
      |    range reads or REST within the budget; one open for PSDParser.parse and psd_details.gather; one bad PSD fails
      |    its card only
      |  authors, Card Edits.json, duplicates, created dates, the archive (dated from git), versions (card_history)
      |  changelog events, gates G1-G6 (a refusal commits nothing)
      v
db/ JSON (stats.json, old_stats.json, psd_details.json, manifest.json, changelog/) -> one commit on the data branch;
the site build dispatched; the posts; the bake and the TTS save on main
```

Design and stages: docs/Campaigns/DATA_BUILD_ACTION.md §6-§7. Until 2026-10-25 the jetson ran a "legacy mode" of the same
traversal inside `/update_stats` and saved the pickles (`card_data.update`, repo_traverser.py): gone at step 10.

### (b2) `/update_stats` (the bot's sync)

```
/update_stats  (wrapper: with force_update or cards, admins only; then defers)
      v
WORKER THREAD: card_sync.update()   (uncompiled)
      |  force_update / cards: POST actions/workflows/card-sync.yml/dispatches (ACTIONS_TOKEN), then GET the run every
      |    15 s (state edited into the reply, queue_edit), 30 min at most; a cancelled run: dispatched once more or said
      v
card_sync.sync()  (one at a time: _sync_lock)
      |  card_db.sync(card_data/): GET commits/data with If-None-Match (304: free, nothing else) -> new commit:
      |    raw db/manifest.json, then only the files whose sha256 differ, checked and parsed before any is written,
      |    each replaced atomically, manifest and sync.json last
      |  new commit: card_db.read_db -> stats_db.stats / old_stats swapped whole (readers hold the old dict or the new)
      v
the reply (say), then queue_command(refresh_history), queue_command(_refresh_cards: populate_files, prep_dataframes,
      then queue_command(sync_changed_emojis) over that tree)
```

Nothing parses and nothing is written but `card_data/` (gitignored). Startup (`init_psd` -> `card_sync.startup`) loads
`card_data/` first and syncs once; card commands queue the same sync on use (at most every 2 minutes, `card_sync.on_use`).

### (c) A reminder firing (slash_registry + FakeInteraction), as of 2026-10-19

```
process_reminders 1 s tick (main.py:806-808) -> queue_command(check_reminder)
      v  the woken command loop (main.py:824)
check_reminder() in a worker thread (analytics.py:1210)
      |  _check_lock non-blocking acquire (overlapping pass -> return) ; then under _lock:
      |  for each due reminder that is on and has no problem -> _process_due -> _fire (analytics.py:1137):
      |- channel gone? (_channel_missing: bot ready AND get_channel is None) -> skip, log
      |- fake = make_fake_interaction(channel, guild); fake.user = _ReminderUser(creator_id)
      |         (legacy reminder without creator: fake.user stays the bot's FakeUser)
      |- msg?     queue_message(fake, chunk) per chunk
      '- command? resolve_command(stored text): shlex-split, slash_registry.get(name),
                  bind name:value options against the callback's signature and convert them
                  -> queue_command(_run_reminder_command, callback, fake, values, id, name)
                        |
                        v  at once: run_command (main.py:716) -> coroutine passthrough
                  _run_reminder_command awaits callback(fake, **values) on the event loop;
                  the callback runs lacks_admin(fake) as the CREATOR (invariant 9), then awaits
                  its to_thread'd worker exactly like a real invocation. Exceptions are reported
                  in the channel ("⚠ Reminder <id>: /<name> failed ...").
      then: repeating -> when = next_run(when, frequency, now) (one jump, no replay);
            one-shot delivered -> removed; one-shot whose channel is gone -> retry in 1 h;
            any change -> save_reminders() (reminder.json: temp file + fsync + rename)
      an off reminder (`enabled: false`) or one with a `problem` is skipped: no fire, no reschedule
```

**FakeInteraction (utils.py:160-227) — what and why.** Both send-queues store
`(interaction, kwargs)` and the drain loops call interaction methods on them. A reminder fires with
no real interaction — no token, no response object. `make_fake_interaction(channel_id, guild_id)`
builds an object with just enough surface (`response.is_done/send_message/defer`, `followup.send`,
`edit_original_response`, `user`, `channel_id`) so worker code written against real interactions runs
unmodified. Its default `user` is a FakeUser with the bot's own id (1011264680683110433); reminders
replace it with `_ReminderUser` (invariant 9), which also carries `name`, `display_name`, `mention`
and `bot=False`. Its delivery trick is deliberate: `FakeResponse.send_message` doesn't send — it
**re-queues** the kwargs via `queue_any` with a bare `discord.Object` carrying only `channel_id`
(utils.py:185-191). When the dispatch loop pops that Object, `interaction.response` raises
`AttributeError`, which lands in the fallback branch → `client.get_channel(channel_id).send(...)` —
the one send that needs no interaction. FakeInteraction rides the error path **by design**; do not
"clean up" the dispatch loop's `AttributeError` handling without rerouting reminders first.

## Module dependency map

```
config.py  (SECRET - tokens; names only)      global_config.py  (paths, the data branch, TIMEZONE)
     ^                                              ^
     | import config (module level)                 |
     |  utils, music_player                         |  main, analytics, card_models, github_api,
     |  (github_api: try/except, getattr defaults;  |  psd_analyzer, psd_parser, query_card,
     |   everyone else reads it through github_api) |  repo_history, card_data, tts_save,
     |                                              |  type_emojis, update_bot
     |
main.py  (registration + queue loops; imports every commands module)
 |- commands.utils         queues, to_thread, say/queue_long_message, lacks_admin, FakeInteraction  <- everyone's base
 |- commands.help          -> utils
 |- commands.management    -> utils
 |- commands.analytics     -> utils, global_config
 |- commands.music_player  -> utils, config
 |- commands.github_api    -> global_config, config (optional): GIT_TOKEN, fetch_tree/fetch_blob, get_with_retry
 |                            [2026-10-20; NOT compiled; the one home for GitHub access]
 |- commands.query_card    -> utils, global_config, github_api, type_emojis (module level, 2026-10-19), repo_history,
 |                            card_models (stats_db), card_edits, card_history (start);
 |                            psd_analyzer LAZILY (the facade-only names)
 |- commands.type_emojis   -> utils, global_config, github_api, repo_history; query_card.card_repo LAZILY inside _list_type_icons
 |                            [2026-10-19; NOT in setup.py INCLUDE_FILES, so never compiled]
 |- commands.repo_history  -> global_config ONLY  [2026-10-20; commit-pinned image links; NOT compiled]
 |- commands.git_commit    -> github_api, repo_history, global_config: one commit on the card repo (Git Data API; git objects in the
 |                            history clone pushed over SSH when the token can't write), shared by the edit pipeline's card_bake
 |                            and card_guards (Actions) [2026-10-21; NOT compiled; no discord or config]
 |- commands.tts_save      -> git_commit, repo_history, card_models, card_edits, global_config: `python -m commands.tts_save
 |                            <save.json>`, no bot module imports it (TTSCardMaker's
 |                            card-sync.yml's TTS save step does (tts-save.yml until 2026-10-24), with the AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE env
 |                            overrides global_config reads) [2026-10-20; /update_tts_save retired 2026-10-25; NOT compiled]
 |- commands.card_models   -> global_config ONLY: CardType, CardStats, CardInfo, StatsDatabase, stats_db,
 |                            card_name  [2026-10-20 split; compiled]
 |- commands.card_edits    -> card_models: CardAuthors, CardEdits  [compiled]
 |- commands.psd_parser    -> card_models, global_config, psd_tools: PSDParser, CardClassifier, CardValidator,
 |                            IconVocabulary  [compiled; no Discord, network or pandas]
 |- commands.psd_details   -> font_check, psd_regions, psd_parser, card_models, github_api, global_config, psd_tools (Pillow's ImageCms
 |                            optional): gather, load/save/prune, the clone CLI  [2026-10-21; NOT compiled; no discord or config;
 |                            card_data imports it LAZILY inside its functions so fontTools stays off the startup path]
 |- commands.card_data     -> card_models, card_edits, psd_parser, github_api, global_config; psd_details, card_history, card_db,
 |                            changelog LAZILY: CardBuild (the parse and REST blobs), GitBuild (the clone), build (the Card sync
 |                            Action's db/ JSON)  [2026-10-22; NOT compiled; no discord, pandas, config or utils: card-sync.yml
 |                            checks out card_data.SPARSE_FILES alone]
 |- commands.card_sync     -> card_db, github_api, psd_analyzer (stats_db, the facade's), utils, global_config, config (optional);
 |                            query_card, type_emojis, repo_history, analytics, psd_details, font_check LAZILY: the bot's card
 |                            data (sync, /update_stats, admins' Card sync runs, /fonts from data). main imports it; psd_analyzer
 |                            LAZILY (init_psd) [2026-10-22; NOT compiled]
 |- commands.psd_analyzer  -> the three above, query_card, github_api, utils, pandas: the FACADE. Re-exports every card name, holds the
 |                            metadata/export commands and init_psd. Code outside imports card classes from HERE only.
 |- commands.frankenstein  -> utils, query_card (card_repo)
 '- commands.update_bot    -> repo_history (batch_overrides);
                              GitPython LAZILY inside its functions  (no utils: bypasses queues on purpose, invariant 3)

Outside the bot's import graph (CLIs and TTSCardMaker Actions only; no bot module imports them): card_check,
card_guards, card_standards, png_export, psd_diff, txt2_repair, tts_save and card_dates
(since 2026-10-25 only `created`, the creation-dates CLI; /update_metadata, /mass_replace and /alias are on branch
archive/discord-edits). card_db, card_history and changelog are in it, reached from card_data, card_sync and query_card.

(not in main) branch archive/local-update-stats: archive/local_update_stats.py  (2026-10-20; NOT compiled)
     -> commands.psd_analyzer, commands.query_card, global_config   [the archived local /update_stats mode]
```

**The deliberate lazy-import cycle breaks (two of them).** The facade `psd_analyzer` imports
`card_repo` from `query_card` at module level because it uses the repo object throughout. `query_card` takes
`stats_db` from `card_models` at module level (query_card.py:20, since 5835139, 2026-10-21):
card_models imports nothing of the facade, so that is no cycle, and every deploy rebuilds every `.so`, so the old
"two `stats_db` objects" risk of a stale compiled facade is gone. The names only the facade holds
(`get_stats_as_dict`, `CardEdits` through it, `_format_metadata`) it imports **inside the function**
(query_card.py:441, 510, 598, 606, 733, 762: `_populate_edit_names`, `prep_dataframes`, the card lookups). The same shape repeats since 2026-10-19: `query_card` imports `render_type_tokens` from
`type_emojis` at module level, and `type_emojis` needs `card_repo` only when listing icons, so `_list_type_icons`
imports it inside the function. Hoist either import to module level and you get a circular-import crash at startup
(psd_analyzer → query_card → psd_analyzer before it finished executing). The lazy side is the one
whose need is deferred anyway — keep it that way.

## Why the card data is JSON on a git branch (and was pickles in this repo)

The bot runs on the jetson (production, systemd `autocroissant.service`) or the owner's Mac — whichever machine is on,
never both (user-stated 2026-07-11). Until 2026-10-25 the database was three pickles (`stats.pkl`, `old_stats.pkl`,
`aliases.pkl`) **committed to this repo** and moved by the bot's own `/push` and `/update`: one transport, zero extra
infrastructure, git history for free (a bad `old_stats.pkl` was cleanly reverted, cca0aaf → eb9aa84, 2025-11-10). What it
traded away: no locking, no merges (binary; take-ours on a conflict), no schema migration (366c8d9), Python-only access
(the website stubbed the classes to unpickle them), and corruption replicated by design — the top-ranked costliest
failure class.

**Now (DATA_BUILD_ACTION.md; cutover 2026-10-24 23:02, step 10 2026-10-25):** TTSCardMaker's Card sync Action is the one
writer. It builds the card data with this repo's `commands/card_data.py` and commits `db/` (JSON, one card per line, a
sha256 manifest) to the orphan branch `data`: readable diffs per card, gates before every commit (G1-G6), a `restore`
dispatch for rollback, and no host that must be on. The bot only reads it (data flow b2) into `card_data/` and memory; the
website builds from it too. The pickles stay in this repo's history (the build's seed and the diagnostics read them
there). The change discipline (the data commit's review) is owned by **autocroissant-change-control**.

## Known-weak points, stated plainly

No sugarcoating. Each verified 2026-07-11; items 4, 5, 8 and 9 updated or added 2026-10-19; item 5
updated and line numbers re-checked 2026-10-20; items 1, 7, 8 updated and 10-11 added after that day's
optimization pass.

1. **(Fixed 2026-10-20) `main.py` used to define two function names twice.** The second `slash_set_ratio`
   (the `/set_repo` handler) is now `slash_set_repo`, and the second `slash_delete_song` (`/delete_all_music`)
   is `slash_delete_all_music`. Registration never depended on the names (`@tree.command` captures each
   function at `def` time and `slash_registry` keys on `cmd.name`); `tests/test_robustness.py` asserts every
   top-level def is unique. Changing `name=` is still not safe (invariant 4). Archaeology Entry 10.25.
2. **`shuffle(music_queue)` shuffles a deque** (music_player.py:338; `music_queue` is a `deque`,
   music_player.py:32). `random.shuffle` does indexed swaps and deque middle-indexing is O(n), so the
   shuffle is O(n²)-ish. Correct, and fine at hobby-scale queue lengths; would hurt at tens of
   thousands of entries. Not worth fixing until it is.
3. **The dispatch fallback drops `ephemeral`** (main.py:752, in `_fallback_send`: `kwargs.pop("ephemeral", None)`
   before `channel.send`). A reply the user expected to be private — e.g. an ephemeral
   reply — posts publicly to the channel if the send lands in the fallback path (token
   expired, or any FakeInteraction send). `channel.send` simply has no ephemeral concept.
4. **(Fixed 2026-10-19) Remote traversal used to leak temp files.** Each downloaded temp file is now
   removed right after parsing (`_release` in card_data; `_download` removes its own on a failed write). The bot parses
   no PSD since 2026-10-25 (the startup sweep, `sweep_temp_psds`, went with it). Kept here as a pointer: archaeology
   Entry 10.2.
5. **(Retired 2026-10-20) Local mode's change signal did not match the stored data.** The local-mode PATH
   bug (removeprefix, 2025-12-02 → 2026-10-19) was fixed with `relpath` (story:
   **autocroissant-failure-archaeology** Entry 1). What remained: with `use_local_timestamp:True` (the
   default) local mode compared file mtimes with timestamps that came from GitHub commit dates, and every
   clone's mtimes are newer (813/813 on 2026-10-19), so a default local run archived every card into
   old_stats.pkl (docs/BUGS_AND_IMPROVEMENTS.md B6). The default became remote on 2026-10-20 and the same day local
   mode was archived: `/update_stats` and `update_stats()` no longer take `use_local_repo` /
   `use_local_timestamp`, the traversal lives on branch `archive/local-update-stats` (not in main, so the bot
   can't import it; not compiled), and stored reminders naming them fail binding (the dropping shim was removed 2026-10-25). Revive it only
   after fixing the change signal.
6. **(Gone 2026-10-25) Pickles-as-database implications** — see the section above; the card data's gates live in the
   build (DATA_BUILD_ACTION.md §13) and its review in **autocroissant-change-control**.
7. **`/pull` blocks the event loop.** `git_pull` is `async def` running blocking GitPython network
   calls directly on the loop (update_bot.py:33; `to_thread` passes coroutine functions through,
   utils.py:118-120). For `/update` and `/restart_bot` that's fine — the queues are stopped and the
   process is about to be replaced — but `/pull` (main.py:226-232) does NOT stop the queues, so all
   sends and heartbeats stall for the duration of the fetch/merge. Deliberate simplicity; know it's
   why the bot "freezes" briefly during a pull. Since 2026-10-20 each git network call has
   `kill_after_timeout=NET_TIMEOUT` (120 s, update_bot.py:10), so a stalled fetch or push ends; since 2026-10-21
   git runs non-interactively (`_git()`/`_repo()` set `repo_history.batch_overrides`: GIT_TERMINAL_PROMPT=0, ssh
   BatchMode), so a host-key or credential prompt fails instead of hanging.
8. **(Fixed 2026-10-20) The dispatch and edit loops used to die for good** (was B5). Each drained item now
   runs in its own `try` that logs one line (`Dispatch: dropped a message for channel N: ...`); the
   fallback (`_fallback_send`) rewinds files, and skips a channel `client.get_channel` can't find (logged);
   content over 2,000 characters is split with `split_long_message` before sending (edits keep the first
   chunk and post the rest). `run_command` logs a failed queued function with its name and traceback.
   Archaeology Entry 10.22.
9. **Icon names are a cross-repo contract.** The parser (`ICON_FONT_PREFIX`, `ICON_MANIFEST_PATH`,
   `IconVocabulary`), the emoji names (`type_emojis.emoji_name`) and the website all derive names from
   TTSCardMaker's `Types/` file stems and the stoogedings manifest. Since 2026-10-20 the parser accepts
   exactly the font's spellings; emoji names and the website still normalize differently. Renaming or
   deleting an icon PNG breaks things on both sides without an error, because the manifest keeps the old
   entry (ttscardmaker-companion §1.6-1.7; docs/BUGS_AND_IMPROVEMENTS.md C13).
10. **The queue tick was most of a reply's latency** until 2026-10-21 (measured 2026-10-20, Mac:
   `queue_message` to send 502 ms mean, p95 966 ms, against 3.9 ms of /query work), and a search sent one
   message per result (85 per /query_ability on average, max 763) through the single dispatch loop. The
   drains ticked every 0.1 s from 2026-10-21 and wake on enqueue since 2026-10-25 (B-1; reminders keep a 1 s
   heartbeat). Simulated with the real loops on the Mac (`~/Documents/Tools/wt/scratch/opt/bot/instant/`): reply hop
   50 ms mean / 95 ms p95 before, 0.4-1.0 ms / 1-3 ms after; the four loops' idle CPU 1.67 → 0.54 ms/s. /query_ability
   listed name and image link 10 per message from 2026-10-21 until the owner reverted it on 2026-10-23
   (Discord previews only a few images per message): it sends one message per card again, the image URL alone
   (archaeology 10.30; `docs/Finished/optimization-pass-2026-10-20/profile.md` items 1-2). What remains:
   the dispatch loop awaits sends one after another, so a big result set (`limit` caps it) holds other
   replies behind it.
11. **(Gone 2026-10-25) Pickle saves.** The bot writes no card data; its copy in `card_data/` is replaced file by file
   atomically by `card_db.sync` (temp file, fsync, `os.replace`; the manifest and `sync.json` last, so a cut-off sync
   is reloaded whole by the next; archaeology 10.29 for the pickle era).

## Where do I put new code (quick answers)

- **New slash command**: a plain sync function in the topical `commands/*.py` module; every output via
  `queue_*`; then in main.py the `func = to_thread(func)` rebind + `@tree.command` wrapper (invariant
  4). Full checklist with gates: **autocroissant-change-control**.
- **New recurring/background work**: do NOT add a fourth `tasks.loop`. One-shot deferred work →
  `queue_command(fn, ...)` (runs at once in a thread). Recurring work on a schedule → prefer a
  reminder that runs a slash command (`/set_reminder ... frequency:1d command:/update_stats` is the
  usual example); only if that can't work, follow the `check_reminder` pattern: cheap per-tick
  body, single-flight, heavy work queued onward (invariant 7).
- **Long-running work** (parsing sweeps, downloads): it already runs in a thread — report
  progress with `queue_edit`, expect your interaction token to expire, and rely on the fallback chain
  (write nothing that only works pre-expiry).
- **Work that must await Discord APIs other than sending messages** (emoji, guild management): an
  `async def` worker in the style of `sync_type_emojis` — await the Discord calls, thread every blocking
  step with `asyncio.to_thread`, report through `queue_message`.
- **Bounded parallelism**: pass `executor=<ThreadPoolExecutor>` through `queue_command` (main.py:719-720)
  as the music downloader does; remember `executor` is a reserved kwarg name.
- **Retiring a feature (owner's rule, 2026-10-20: archives go on branches)**: delete the code from main and
  commit it to its own branch `archive/<topic>`, made from main in a separate worktree so uncommitted work on
  main is untouched (`git worktree add /tmp/wt -b archive/<topic> HEAD`, add the file, one commit,
  `git worktree remove /tmp/wt`). Never merge it. Main carries no `archive/` folder. The archived file's header
  says when and why, and docs point at the branch. Example: local mode on branch `archive/local-update-stats`
  (list archive branches with `git branch --list 'archive/*'`). **Dead code with no callers is just deleted**,
  not archived (owner, 2026-10-20). The old reminder argument parser and the unused image helpers were deleted
  that way; their short-lived archive branches were removed too. Old code stays readable in history
  (`git show a7a4bb3:commands/utils.py`).
  Read one with `git show archive/local-update-stats:archive/local_update_stats.py`. Cut decorators with the
  function they decorate (Entry 10.13 in failure-archaeology). If a slash option goes away, stored reminders naming it fail binding when due (invariant 4; the
  `RETIRED_OPTIONS` shim was removed 2026-10-25).

## When NOT to use this skill

- **A live symptom to triage** (bot silent, commands missing, stats look wrong, music dead) →
  **autocroissant-debugging-playbook**. This skill explains why the design is what it is; that one
  gets you from symptom to cause.
- **Card-domain semantics** (folder classification, PSD layer rules, type injection, query DSL,
  exclusion lists) → **impossibility-cards-reference**.
- **Building the environment, Cython** → **autocroissant-build-and-env**.
- **Making a change** (checklists, the data gate, what may never be committed) →
  **autocroissant-change-control**.
- **Incident history and the removeprefix bug's full story** → **autocroissant-failure-archaeology**.
- **Operating reminders and type emojis day to day** → **autocroissant-run-and-operate** §9, §11.
- **The TTSCardMaker side of icons, the font manifest and the website** → **ttscardmaker-companion**.
- **AI/diffusion doctrine and enablement** → **autocroissant-ai-boundary** (branch `archive/ai-generation`).

## Provenance and maintenance

Line numbers were verified 2026-07-11 at HEAD `284d13c`, re-verified 2026-10-20 at HEAD `9fb3093`, re-mapped after that
day's optimization pass (psd_analyzer split, `refs6.py`) and re-verified 2026-10-23 at `2ada357`
(expected values below are the 2026-10-23 readings). They WILL drift.
Re-find before citing:

| Fact | Re-verify with (cwd = repo root) |
|---|---|
| Three woken drains + the 1 s reminder loop | `grep -n -e "tasks.loop" -e "async def process_" main.py` (806-807 reminders at 1 s; 823-824 / 847-848 / 884-885 at `seconds=0`, paced by the wait) |
| Wake-ups: every queue_* sets its wake; the stop wakes them | `grep -n "^class Wake\|_wake.set()\|^FALLBACK" commands/utils.py` (35, 4 sets, 32); `grep -n "_stop_loops" main.py` (3 calls, def 895) |
| check_reminder queued by its own 1 s loop | `grep -n "queue_command(check_reminder)" main.py` (808, inside `process_reminders`) |
| Loops started once in on_ready (revived on a reconnect) / stopped in restart/stop/update | `grep -n "_start(loop)\|def _loops" main.py` (129 revive, 144 start, 890); `grep -n "_stop_loops()" main.py` (207 / 217 / 242) |
| Five queued inits; run once; reminder client binding | `grep -n "queue_command(init\|queue_command(refresh_history\|bind_reminder_client(client)\|_started = True" main.py` |
| The three deques | `grep -n "= deque()" commands/utils.py` (26-28) |
| queue_* helpers | `grep -n "^def queue_" commands/utils.py` (69/75/81/87/93/99) |
| to_thread coroutine passthrough | `grep -n -e "def to_thread" -e "iscoroutinefunction" commands/utils.py` (116, passthrough 118-120) |
| run_command + reserved `executor` kwarg | `grep -n "async def run_command" main.py` (716); `grep -rn "executor=executor" commands/music_player.py` |
| Registration rebind count | `grep -c "= to_thread(" main.py` (43 on 2026-10-25) |
| One sync at a time; no bot module imports the bake | `grep -n "_sync_lock\|_run_lock" commands/card_sync.py`; `grep -ln "card_bake" main.py commands/card_sync.py commands/psd_analyzer.py` (empty) |
| slash_registry population | `grep -n "slash_registry\[cmd.name\]" main.py` (134) |
| Reminder resolve/bind/queue | `grep -n "slash_registry.get\|queue_command(_run_reminder_command" commands/analytics.py` |
| Reminders fire as creator; single-flight | `grep -n "_ReminderUser(\|_check_lock\|^LEGACY_RUNS_AS_BOT" commands/analytics.py` |
| FakeInteraction + default bot user id | `grep -n "def make_fake_interaction\|1011264680683110433" commands/utils.py` (160, 174) |
| lacks_admin polarity | `grep -n -A2 "def lacks_admin" commands/utils.py` (returns `not in ADMINS`) |
| Unique handler names (was duplicate defs) | `grep -n -e "async def slash_set_re" -e "async def slash_delete_" main.py` (4 distinct names) |
| Deque shuffle | `grep -n "shuffle(music_queue)" commands/music_player.py` (338) |
| Ephemeral dropped in fallback; get_channel None guarded | `grep -n 'pop("ephemeral"\|def _fallback_send\|not found")' main.py` (752, 750, 759) |
| Temp files removed after parse (weak point 4 fixed) | `grep -n "mkstemp(\|remove(" commands/card_data.py` |
| Local mode archived 2026-10-20 (weak point 5) | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `grep -rn "use_local" main.py commands/` (finds nothing since 2026-10-25) |
| Module-level cycles + lazy breaks | `grep -n "from commands.query_card import card_repo" commands/psd_analyzer.py commands/type_emojis.py` (module level in the first; inside a function in type_emojis); `grep -n "from commands.psd_analyzer import" commands/query_card.py` (all inside functions: 442, 511, 599, 607, 734, 763); `grep -n "^from commands" commands/query_card.py` (18-24: card_edits, card_history, card_models, github_api, repo_history, type_emojis, utils) |
| Singletons | `grep -n "stats_db = StatsDatabase()" commands/card_models.py` (409); `grep -n "card_repo = CardRepository()" commands/query_card.py` (999); `grep -n "state = MusicState()" commands/music_player.py`; `grep -n "^reminders = \|^type_emojis:" commands/analytics.py commands/type_emojis.py` |
| config import chain | `grep -ln "^import config" commands/*.py` (music_player, utils); `grep -n "import config" commands/github_api.py` (try/except) |
| type_emojis uncompiled | `grep -c "type_emojis" setup.py` (0) |
| update_bot bypasses queues | `grep -n "followup.send" commands/update_bot.py`; `grep -n "from commands" commands/update_bot.py` (6: repo_history at module level; 38: card_models inside `_rescue_merged_cards`); `grep -n "from git" commands/update_bot.py` (19, 27: inside `_git`/`_repo`) |
| History step before the traversal (data flow b) | `grep -n "line = cleaned\|history_index()\|again = cleaned\|db.save()" commands/card_data.py` (1137-1138, 1153, 1159) |

Maintenance rule: if any grep above stops matching, the architecture moved — update this contract in
the same change, and check whether the corresponding invariant row still holds before assuming it does.

2026-10-20 (later): profile.md, cited in weak point 10, moved from `~/Documents/ac-optimization/` to `docs/Finished/optimization-pass-2026-10-20/`.

2026-10-21: the drains tick every 0.1 s and `process_reminders` keeps the 1 s reminder heartbeat (weak point 10 done, invariant 7; reminder data flow (c) starts at that loop; review re-pointed run_command, executor, _fallback_send, invariant 4
reworded); card pickle saves atomic and locked through `card_models.save_pickle` (weak point 11 rewritten); main.py, card_models.py,
query_card.py and help.py line numbers re-pointed.

2026-10-21 (audit fixes): commands/git_commit.py added to the module map (the commit plumbing tts_save and card_bake used to carry
twice); tts_save's dependency line re-pointed (github_api now reached through git_commit).

2026-10-21 (titles): card_models.py and repo_traverser.py citations re-pointed (`CardInfo.parser_version`; `_psd_changed` re-parses older stamps).

2026-10-21 (review): repo_traverser.py `update_stats` :472.

2026-10-21 (S7): commands/card_runner.py in the module map; invariant 10 (one Lock for update_stats and the card bake) and the queue-trio bullet on concurrent queued commands; data flow (b) ends with the edits-file notice; rebind count 50; every main.py, repo_traverser.py and help.py citation re-pointed by the old-to-new line map (main.py +1 from :21, +17 from :399; repo_traverser.py +1 from :22, +14 inside update_stats; help.py +5 from :89).

2026-10-21 (hook removed): `notice_edits` and `_last_edits_sha` deleted from card_runner.py, the call at repo_traverser.py:541-542 with them (the user: the Action is the one live runner, the jetson a manual fallback). Data flow (b) now ends with the dataframe refresh; the queue-trio bullet, invariant 10 (bake_lock stays), the module map and the re-verification row updated; no other cited line moved.

2026-10-21 (details): commands/psd_details.py in the module map (lazy from repo_traverser); data flow (b) gathers the Details record per parsed card and saves psd_details.json beside the pickles; the downstream line names the file. repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-22 (data build): /update_stats' traversal moved to the uncompiled commands/card_data.py (`CardBuild`,
`card_data.update`); repo_traverser.py is the Discord wrapper (bake lock, messages, follow-ups). The /update_stats data
flow, the bake-lock and temp-file citations and the module map re-pointed; card_data's own entry (no discord, pandas,
config or utils) added. The lazy-import cycle break is unchanged (repo_traverser still imports card_repo at module level).

2026-10-22 (cutover, bot side): data flow (b2) for /update_stats with `CARD_SYNC` on, commands/card_sync.py in the module map,
invariant 10's after-the-cutover note, and the successor paragraph under "Why pickles-in-git". main.py grew by 13 lines (an
import, the switched texts, the option gate) and repo_traverser.py by 5, so their cited line numbers drift by up to that;
re-find them by grep.

2026-10-23 (docs drift): 17-skill map (impossibility-simulator-rules added); WHY #4 and invariant 3 name sweep_temp_psds,
the run-once inits and the reconnect revive (`_loops()`/`_start()`); invariant 10 says the jetson has the fork; the lazy-import
section says query_card takes stats_db from card_models at module level; module map: query_card, discord_edits, update_bot
(repo_history; card_models and GitPython lazily) and the CLI/Action-only modules; data flow (b) starts with the history
cleanup/recovery; (b2) and the successor paragraph: merged and deployed off, shadow schedule on; weak point 7: git never
prompts; every main.py, card_models.py, query_card.py, card_data.py and update_bot.py citation re-pointed at 2ada357
(main.py had drifted by about 50 lines, not 13).

2026-10-23 (unbatch): invariant 10 says /query_ability sends one message per card again (the 10-per-message listing reverted); query_card.py citations re-pointed (-1, -9 after query_ability).

2026-10-25 (retire): /update_metadata, /mass_replace, /alias, /bake_cards and /update_tts_save retired (branch archive/discord-edits); invariant 10, the module map (card_runner, tts_save, card_dates (was discord_edits) now outside the bot) and data flow (b) updated.

2026-10-25 (step 10, work/pickles): the pickles, /push, `CARD_SYNC`, repo_traverser.py (the legacy /update_stats) and card_runner.py (bake_lock, the startup sweep) removed: data flows (b) and (b2), invariants 6 and 10, on_ready's five inits, the module map, "Why the card data is JSON on a git branch" and weak points 4, 6 and 11 rewritten.

2026-10-25 (B-1, work/instant): the drains wake on enqueue (`Wake`, `call_soon_threadsafe`, 1 s fallback poll) instead of
the 0.1 s tick; `_stop_loops()`; the execution model, the queue trio, WHY #3, invariants 3 and 7, data flows (a) and (c),
weak point 10 and the re-verify rows updated; main.py and utils.py citations re-pointed.
