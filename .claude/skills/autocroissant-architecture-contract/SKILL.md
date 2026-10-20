---
name: autocroissant-architecture-contract
description: Load this when you need to understand or safely modify HOW AutoCroissant works as a system — "how does the bot work", "start here", "orient me", "overview of this repo", "which skill do I need" (newcomer routing map at the top), "where do I put new code", "why is this function sync", "design", "invariants" — or any task touching threading, the async event loop, the command/dispatch/edit queues, tasks.loop, to_thread, queue_message/queue_file/queue_edit/queue_command, slash-command registration, slash_registry, reminder command execution, reminders running as their creator / perms_check at fire time, FakeInteraction, _ReminderUser, async workers that await Discord (type_emojis), module import order/cycles, module-level singletons, or the pickles-in-git database decision. Contains the execution model (discord.py event loop + three module-level deques drained by 1-second loops + sync workers in threads) and WHY it exists; an invariants table where each row says what breaks if you violate it; the module dependency map including the deliberate lazy-import cycle break; ASCII data flows for a slash command, the /update_stats pipeline, and a reminder firing; and the known-weak points stated plainly (duplicate function names in main.py, ephemeral-dropping fallback, the dispatch loop that can die, local-mode timestamp mismatch, pickle trade-offs, cross-repo icon contracts). Design/invariant questions land here; live-symptom triage belongs to autocroissant-debugging-playbook.
---

# AutoCroissant Architecture Contract

## New here? The 16-skill map in 11 lines

- Run / stop / deploy / machine handoff / run update_stats → autocroissant-run-and-operate
- Something is broken RIGHT NOW → autocroissant-debugging-playbook
- Install / env / Cython build → autocroissant-build-and-env ; every config axis → autocroissant-config-and-flags
- BEFORE any commit, push, or pickle touch → autocroissant-change-control ; is it safe/finished → autocroissant-validation-and-qa
- What cards/PSDs/queries mean → impossibility-cards-reference ; measuring tools → autocroissant-diagnostics-and-tooling
- Improving the parser → autocroissant-psd-extraction-campaign ; proving things → autocroissant-analysis-toolkit
- "Has this been tried?" / why is this code weird → autocroissant-failure-archaeology
- AI/diffusion: removed from main 2026-10-20; code and its skill are on branch `archive/ai-generation` ; future ideas → autocroissant-research-frontier
- "How do I know I'm right" → autocroissant-research-methodology ; help text / style / skill upkeep → autocroissant-docs-and-style
- The TTSCardMaker side (icon font, Types/ icons, card website, Photoshop scripting) → ttscardmaker-companion
- THIS skill: how the system works + the invariants. Read "The execution model" before writing code.

Facts and line numbers verified 2026-07-11 (HEAD `284d13c`); re-verified 2026-10-20 against
`main` HEAD `9fb3093`.
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
`edit_queue`, utils.py:26-28). Three `@tasks.loop(seconds=1)` coroutines in `main.py`
(`process_command_queue` main.py:750, `process_dispatch_queue` main.py:759, `process_edit_queue`
main.py:791) drain those deques once per second **on the event loop** and perform the actual Discord
I/O, with a built-in fallback chain. That is the whole model: loop owns Discord, threads own work,
deques are the only bridge.

## The queue trio (this skill owns this detail)

| Queue (utils.py:26-28) | Entry shape | Filled by | Drained by | What draining does |
|---|---|---|---|---|
| `command_queue` | `((args, kwargs), func)` | `queue_command(func, *a, **kw)` (utils.py:59) | `process_command_queue` (main.py:750-754) | `create_task(run_command(...))` per entry — each func runs in a fresh thread (or threadpool, see below) |
| `dispatch_queue` | `(interaction, send_kwargs)` | `queue_message` (utils.py:42), `queue_file` (utils.py:48), `queue_any` (utils.py:37) | `process_dispatch_queue` (main.py:759-786) | Sends via fallback chain: `interaction.response.send_message` → `interaction.followup.send` → `client.get_channel(channel_id).send` |
| `edit_queue` | `(interaction, edit_kwargs)` | `queue_edit` (utils.py:54) | `process_edit_queue` (main.py:791-803) | `interaction.edit_original_response`, falling back to a fresh `channel.send` |

Mechanics you must know:

- **`to_thread(func)`** (utils.py:67-76) wraps a sync function so awaiting it runs it in a thread via
  `asyncio.to_thread`. Critical subtlety: if `func` is already a coroutine function it is returned
  **unchanged** (utils.py:69-71). This passthrough is what lets `run_command` (main.py:740-745) execute
  both sync workers and the async slash callbacks stored in `slash_registry` with one code path.
- **`to_threadpool(func, executor)`** (utils.py:79-88) is the bounded variant. `run_command` routes to
  it when the queued kwargs contain `executor` (main.py:742-743) — so **`executor` is a reserved kwarg
  name** for anything sent through `queue_command`; a worker function with its own `executor` parameter
  would have it silently popped. Sole current user: music downloads, capped at a 2-worker
  `ThreadPoolExecutor` (music_player.py:36, used at music_player.py:251 and 265) so a playlist can't
  spawn unbounded yt-dlp threads.
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

### WHY this design (the load-bearing reasons)

1. **Thread safety.** discord.py's client and event loop are not thread-safe. A worker thread that
   calls `interaction.response.send_message(...)` gets a coroutine object it cannot await — the message
   silently never sends — and any attempt to drive the loop from the thread risks corrupting it. The
   queues mean worker code never needs the loop at all.
2. **Centralized fallbacks survive interaction-token expiry.** Discord interaction tokens die after
   15 minutes. Long jobs (full `/update_stats` sweep, playlist download) outlive them.
   The dispatch loop's chain — `response.send_message` if not yet responded, else `followup.send`, and
   on `HTTPException`/`AttributeError` a raw `channel.send` (main.py:778-786) — means a late result
   degrades to a plain channel message instead of vanishing. Every send written as `queue_*` gets this
   for free; every send written by hand does not.
3. **Serialization + self-throttling.** One drain point per second batches sends and naturally stays
   under Discord rate limits without any explicit rate-limit code. Cost: up to ~1s latency per hop,
   and a reply that traverses two queue hops (see the reminder diagram) takes ~2s. That is the
   accepted trade at hobby scale.
4. **Deferred init without blocking login.** `on_ready` (main.py:101-119) queues `init_reminder`,
   `init_psd`, `init_query`, `init_type_emojis(client)` as commands (main.py:111-114)
   instead of running them — pickle loads and the GitHub tree fetch happen in worker threads after the
   bot is already up. `init_type_emojis` is async: it runs on the loop and threads its own downloads.
   main.py also calls `bind_reminder_client(client)` at import time (main.py:204) so the scheduler can
   tell when a reminder's channel is gone.

## Invariants — violate this and X happens

| # | Invariant | Violate it and... |
|---|---|---|
| 1 | **No awaits inside queued sync functions.** Workers are plain `def`; there is no event loop in their thread. | `await` in a `def` is a SyntaxError — so the tempting "fix" is making the worker `async def`. Do that and `to_thread` passes it through unchanged (utils.py:69-71): your "worker" now runs ON the event loop, and its blocking calls (PSD parse, yt-dlp, GitPython) freeze the entire bot — heartbeats, all three queue loops, everything — until it finishes. |
| 2 | **All Discord output from worker code goes through `queue_message`/`queue_file`/`queue_edit`/`queue_any`/`queue_command`** (utils.py:37-61). | Direct calls from a thread produce never-awaited coroutines (message silently lost) or thread-unsafe loop access. You also forfeit the fallback chain: your send dies permanently at the 15-minute token expiry instead of degrading to `channel.send`, and reminders (which run your command with a FakeInteraction) break because FakeInteraction only implements the queue-facing surface. |
| 3 | **The three queue loops must be running for ANYTHING to send.** Started in `on_ready` (main.py:116-118); deliberately STOPPED by `/restart_bot` (main.py:145-147), `/stop_bot` (main.py:157-159), `/update` (main.py:194-196) so no queued output races process replacement. | With a loop stopped (or `on_ready` never reached), `queue_*` calls accumulate silently in the deques forever — the bot looks alive and says nothing. This is also why `commands/update_bot.py` is the one module that bypasses the queues entirely: its functions are `async def` awaiting `interaction.followup.send` directly (update_bot.py:16, 42, 131), because during `/update` the dispatch loop is already stopped and queued progress reports would never flush. |
| 4 | **Registration pattern:** `func = to_thread(func)` rebind FIRST, then the `@tree.command` async wrapper that awaits it (48 occurrences as of 2026-10-20; e.g. main.py:251, 352). `on_ready` stores `slash_registry[cmd.name] = cmd.callback` (main.py:104-105), which powers reminder command execution (analytics.py `_resolve_command`, `_fire`). | Forget the rebind line → the wrapper's `await func(...)` executes the sync function inline ON the event loop (bot frozen for its whole duration), then raises `TypeError` awaiting the plain return value. Since 2026-10-19 reminders bind their stored `/name option:value` text against the callback's real signature at creation AND at every firing, so: rename a command's `name=` → saved reminders post "⚠ Reminder `<id>` skipped its command: `/<name>` is not a bot command"; rename or remove a slash **parameter** → "`<option>` is not one of its options", UNLESS you list it in `RETIRED_OPTIONS` (analytics.py:41; with the command's old option order), which makes stored reminders drop it at load and at fire time — done for /update_stats's `use_local_repo`/`use_local_timestamp` on 2026-10-20; change a parameter's type → values that no longer convert are refused. The reminder itself keeps repeating; its creator must re-create it. The wrapper function's Python name, by contrast, is decorative — see weak point 1. |
| 5 | **One bot per process; state is module-level singletons.** `stats_db` (psd_analyzer.py:2288), `card_repo` (query_card.py:714), `state`/`music_queue`/`prev_music`/`executor` (music_player.py:33-45), `reminders` + `_client` + `_lock`/`_check_lock` (analytics.py:61-66), `type_emojis` (type_emojis.py:26, token → emoji map filled by the sync), `slash_registry` + the three deques (utils.py:26-30). | A second client in-process, or importing a commands module under a second name, gives you split-brain state. Import order is load-bearing: the singletons are constructed at import time (cheap — heavy loading is deferred to the queued `init_*` functions), and constructing them prints the git-token-presence lines (query_card.py:48-51; same pattern in StatsDatabase). |
| 6 | **Importing `commands.*` requires `config.py` to exist.** Five modules `import config` at module level (utils, query_card, psd_analyzer, music_player, and since 2026-10-19 type_emojis), and psd_analyzer pulls in query_card at import (psd_analyzer.py:26), which pulls in type_emojis (query_card.py:15). | Any tool that unpickles `stats.pkl` must have this repo on `sys.path` AND a config.py present — `CardInfo`/`CardStats` are dataclasses defined in `commands.psd_analyzer`, so unpickling imports it, and the chain reaches `config`. This is why the diagnostics scripts run from the repo root; code outside this repo must stub the classes, as TTSCardMaker's website build does (`_Unpickler` in site/build/build_site.py). config.py is SECRET — never print or read its values. |
| 7 | **`check_reminder` is re-queued every tick — reminder-path work must stay cheap, and stay single-flight.** The first statement of `process_command_queue` is `queue_command(check_reminder)` (main.py:751); the entry drains on the next tick, so it runs ~once/second in a fresh thread (analytics.py:835). Since 2026-10-19 it takes `_check_lock` with a non-blocking acquire (a pass that finds another still running just returns) and does all work under `_lock` (an RLock shared with set/list/remove, which run in their own threads). Commands of due reminders are queued onward (`queue_command(_run_reminder_command, ...)`), never run inline; reminder.pkl is saved only when something changed, atomically. | Anything slow in `check_reminder` costs a thread-spawn per second and holds `_lock`, stalling `/set_reminder`, `/list_reminders` and `/remove_reminder`. Remove the single-flight lock and two overlapping passes can fire the same reminder twice. Run a reminder's command inline and one slow command blocks every other reminder. |
| 8 | **`perms_check` is inverted:** `return interaction.user.id not in ADMINS` — it returns **True when the user LACKS permission** (utils.py:249-259). Callers write `if perms_check(i): deny-and-return` (e.g. main.py:154-156). | Read it as "has permission" and you either lock the admins out or open an admin command to everyone. The name reads positive; the docstring says the truth. When adding an admin-gated command, copy an existing gate verbatim. |
| 9 | **Admin gates live in the registered slash callback, and reminders fire as their creator** (2026-10-19). A firing reminder's FakeInteraction carries `_ReminderUser(creator_id)` (analytics.py:69, set in `_reminder_interaction`), so the callback's own `perms_check` re-checks ADMINS at fire time. Legacy reminders without a creator keep the default FakeUser — the bot itself (`LEGACY_RUNS_AS_BOT`, analytics.py:37). | Put an admin check anywhere other than the callback stored in `slash_registry` (e.g. only in a worker, or in a pre-dispatch hook) and scheduled runs either skip it or check the wrong user. Make the FakeInteraction default to the bot again and every reminder passes admin checks (the bot's id is in ADMINS on the dev Mac — archaeology Entry 10.8). |

## Data-flow diagrams

### (a) A slash command, end to end

```
Discord user runs /query
      |
      v  (event loop, discord.py)
tree dispatch -> async wrapper slash_query_name (main.py:256-258)
      |   await query_name(...)        # query_name was rebound: to_thread(query_name), main.py:251
      v
asyncio.to_thread -> WORKER THREAD runs sync query_name()
      |   does the work; never awaits; never sends
      |   queue_message(interaction, ...) / queue_file(...)      # appends to dispatch_queue
      v
dispatch_queue (deque, utils.py:26)
      |
      v  (next 1-second tick, back on the event loop)
process_dispatch_queue (main.py:759)
      |- interaction.response.send_message(**kw)     if response not yet done
      |- interaction.followup.send(**kw)             if already responded/deferred
      '- client.get_channel(channel_id).send(**kw)   on HTTPException/AttributeError
      |                                              (token expired, or fake interaction;
      v                                               NOTE: drops `ephemeral` — weak point 3)
Discord
```

### (b) `/update_stats` pipeline (PSDs -> parser -> pickles -> dataframes)

```
/update_stats  (wrapper defers, main.py:363)
      v
WORKER THREAD: update_stats() (psd_analyzer.py:1827)
      |  GitHub trees API with If-None-Match: 304 -> "Nothing changed in the card repo since the
      |    last update." and the run ends (ETag cache in memory only; psd_analyzer.py:1460-1464)
      |  build the icon vocabulary: Types/ + Types/Stars/ names + the stoogedings manifest
      |    (MDW/Fonts/stoogedings-src/stoogedings.json; impossibility-cards-reference §4.1)
      |  pick PSDs (GitHub only; local mode archived 2026-10-20, weak point 5): new, moved, blob sha
      |    != card.blob_sha (_psd_changed, psd_analyzer.py:1379-1391; commits-since fallback for
      |    cards without blob_sha), named in cards:, or force_update
      |  DOWNLOAD_WORKERS=4 stream blobs (git/blobs API) into mkstemp files ahead of parsing
      |    (psd_analyzer.py:1429-1440, 1657-1701)
      v
per picked PSD: one commit lookup (_get_remote_timestamp); PSDParser.parse; remove temp file;
  old CardInfo -> old_stats only if its blob changed or it moved; CardValidator
      v
stats_db singleton (psd_analyzer.py:2288) --save()--> stats.pkl / old_stats.pkl   (the DATABASE)
      |  (save only when num_new > 0 or traverser.changed)
      v
queue_command(card_repo.populate_files); queue_command(card_repo.prep_dataframes)
      |                                            (psd_analyzer.py:1888-1889)
      v  (subsequent command-queue ticks, fresh threads)
card_repo dataframes rebuilt  ->  /query and /query_ability now serve the new data
      |
      '--- progress and problem reports stream back throughout via queue_edit / queue_message

Downstream, outside this process: /push commits the pickles; TTSCardMaker's website build reads
them from this repo's main (ttscardmaker-companion §2).
```

### (c) A reminder firing (slash_registry + FakeInteraction), as of 2026-10-19

```
process_command_queue tick (main.py:751)
      v
check_reminder() in a worker thread (analytics.py:835)
      |  _check_lock non-blocking acquire (overlapping pass -> return) ; then under _lock:
      |  for each due, enabled reminder -> _process_due -> _fire (analytics.py:662):
      |- channel gone? (_channel_missing: bot ready AND get_channel is None) -> skip, log
      |- fake = make_fake_interaction(channel, guild); fake.user = _ReminderUser(creator_id)
      |         (legacy reminder without creator: fake.user stays the bot's FakeUser)
      |- msg?     queue_message(fake, chunk) per chunk
      '- command? _resolve_command(stored text): shlex-split, slash_registry.get(name),
                  drop RETIRED_OPTIONS (/update_stats use_local_repo/use_local_timestamp),
                  bind name:value options against the callback's signature and convert them
                  -> queue_command(_run_reminder_command, callback, fake, values, id, name)
                        |
                        v  next tick: run_command (main.py:740) -> coroutine passthrough
                  _run_reminder_command awaits callback(fake, **values) on the event loop;
                  the callback runs perms_check(fake) as the CREATOR (invariant 9), then awaits
                  its to_thread'd worker exactly like a real invocation. Exceptions are reported
                  in the channel ("⚠ Reminder <id>: /<name> failed ...").
      then: repeating -> when = _next_run(when, frequency, now) (one jump, no replay);
            one-shot delivered -> removed; one-shot whose channel is gone -> retry in 1 h;
            any change -> save_reminders() (temp file + fsync + rename)
```

**FakeInteraction (utils.py:107-174) — what and why.** Both send-queues store
`(interaction, kwargs)` and the drain loops call interaction methods on them. A reminder fires with
no real interaction — no token, no response object. `make_fake_interaction(channel_id, guild_id)`
builds an object with just enough surface (`response.is_done/send_message/defer`, `followup.send`,
`edit_original_response`, `user`, `channel_id`) so worker code written against real interactions runs
unmodified. Its default `user` is a FakeUser with the bot's own id (1011264680683110433); reminders
replace it with `_ReminderUser` (invariant 9), which also carries `name`, `display_name`, `mention`
and `bot=False`. Its delivery trick is deliberate: `FakeResponse.send_message` doesn't send — it
**re-queues** the kwargs via `queue_any` with a bare `discord.Object` carrying only `channel_id`
(utils.py:132-138). When the dispatch loop pops that Object, `interaction.response` raises
`AttributeError`, which lands in the fallback branch → `client.get_channel(channel_id).send(...)` —
the one send that needs no interaction. FakeInteraction rides the error path **by design**; do not
"clean up" the dispatch loop's `AttributeError` handling without rerouting reminders first.

## Module dependency map

```
config.py  (SECRET - tokens; names only)      global_config.py  (paths, pickle names, TIMEZONE)
     ^                                              ^
     | import config (module level)                 |
     |  utils, query_card, psd_analyzer,            |  analytics, query_card,
     |  music_player, type_emojis                   |  psd_analyzer, update_bot
     |
main.py  (registration + queue loops; imports every commands module)
 |- commands.utils         queues, to_thread, perms_check, FakeInteraction  <- everyone's base
 |- commands.help          -> utils
 |- commands.management    -> utils
 |- commands.analytics     -> utils, global_config
 |- commands.music_player  -> utils, config
 |- commands.query_card    -> utils, config, global_config, type_emojis (module level, 2026-10-19)
 |- commands.type_emojis   -> utils, config; query_card.card_repo LAZILY inside _list_type_icons
 |                            [2026-10-19; NOT in setup.py INCLUDE_FILES, so never compiled]
 |- commands.psd_analyzer  -> utils, config, global_config, query_card (module level!)
 |- commands.frankenstein  -> utils, query_card (card_repo)
 '- commands.update_bot    -> global_config ONLY  (no utils: bypasses queues on purpose, invariant 3)

(not in main) branch archive/local-update-stats: archive/local_update_stats.py  (2026-10-20; NOT compiled)
     -> commands.psd_analyzer, commands.query_card, global_config   [the archived local /update_stats mode]
```

**The deliberate lazy-import cycle breaks (two of them).** `psd_analyzer` imports `card_repo` from `query_card` at
module level (psd_analyzer.py:26) because it uses the repo object throughout. `query_card` needs
`psd_analyzer.get_stats_as_dict` only when dataframes are actually rebuilt, so
`prep_dataframes` imports it **inside the function** (query_card.py:379). The same shape repeats since
2026-10-19: `query_card` imports `render_type_tokens` from `type_emojis` at module level
(query_card.py:15), and `type_emojis` needs `card_repo` only when listing icons, so `_list_type_icons`
imports it inside the function (type_emojis.py:60). Hoist that import to
module level and you get a circular-import crash at startup (psd_analyzer → query_card → psd_analyzer
before psd_analyzer finished executing). The lazy side is the one whose need is deferred anyway —
keep it that way.

## Why pickles-in-git is the database

The bot runs on the jetson (production, systemd `autocroissant.service`) or the owner's Mac —
whichever machine is on, never both (user-stated 2026-07-11). State must follow the process across machines. A DB server needs
an always-on host (defeats the point); a cloud DB is ops and cost for a hobby; sqlite still needs a
sync mechanism. The chosen design: the pickles (`stats.pkl`, `old_stats.pkl`, `aliases.pkl`) are
**committed to the same git repo the bot already syncs for code**, moved by the bot's own `/push` and
`/update` commands. One transport, one flow (`/update` = push pickles, pull code, restart), zero
extra infrastructure — and git history for free, which has already paid off: a bad `old_stats.pkl`
snapshot was committed and cleanly reverted (cca0aaf → eb9aa84, 2025-11-10).

What it trades away, plainly: **no locking** (concurrent writers = last-write-wins; survivable only
because there is one process per machine and one machine at a time); **no merges** (pickles are
binary — conflict strategy is take-ours, owned by autocroissant-run-and-operate); **no schema
migration** (the 2025-10-20 "Big massive refactor" 366c8d9 changed the pickle schema; pre-refactor
pickles need pre-refactor code); **Python-only access**, and outside this repo only through a class-stubbing unpickler (invariant 6); and it makes
pickle corruption the top-ranked costliest failure class — a corrupt commit replicates to the other
machine *by design*. The survival discipline (diff before push, PICKLE commit convention) is owned by
**autocroissant-change-control**.

## Known-weak points, stated plainly

No sugarcoating. Each verified 2026-07-11; items 4, 5, 8 and 9 updated or added 2026-10-19; item 5
updated and line numbers re-checked 2026-10-20.

1. **`main.py` defines two function names twice.** `slash_set_ratio` at main.py:304 (the `/set_ratio`
   handler) and again at main.py:314 (the `/set_repo` handler); `slash_delete_song` at main.py:597
   (`/delete_song`) and again at main.py:604 (`/delete_all_music`) — 2026-10-20 lines. Both commands in each pair work —
   `@tree.command` captured each function object at `def` time; the module-level name just silently
   rebound to the second. It works **by decorator side effect**, not by intent. Trap: anything that
   resolves these by module attribute name (hot-reload, a refactor iterating module globals) sees only
   the second def. Renaming the wrapper is safe (invariant 4: wrapper names are decorative); changing
   `name=` is not.
2. **`shuffle(music_queue)` shuffles a deque** (music_player.py:326; `music_queue` is a `deque`,
   music_player.py:33). `random.shuffle` does indexed swaps and deque middle-indexing is O(n), so the
   shuffle is O(n²)-ish. Correct, and fine at hobby-scale queue lengths; would hurt at tens of
   thousands of entries. Not worth fixing until it is.
3. **The dispatch fallback drops `ephemeral`** (main.py:779: `send_kwargs.pop("ephemeral", None)`
   before `channel.send`). A reply the user expected to be private — e.g. `/list_reminders`
   `hidden:True` — posts publicly to the channel if the send lands in the fallback path (token
   expired, or any FakeInteraction send). `channel.send` simply has no ephemeral concept.
4. **(Fixed 2026-10-19) Remote traversal used to leak temp files.** Each downloaded temp file is now
   removed in a `finally` right after parsing, and files left by an error at the end (psd_analyzer.py:1667-1668, 1695-1701). Kept here as a pointer:
   archaeology Entry 10.2.
5. **(Retired 2026-10-20) Local mode's change signal did not match the stored data.** The local-mode PATH
   bug (removeprefix, 2025-12-02 → 2026-10-19) was fixed with `relpath` (story:
   **autocroissant-failure-archaeology** Entry 1). What remained: with `use_local_timestamp:True` (the
   default) local mode compared file mtimes with timestamps that came from GitHub commit dates, and every
   clone's mtimes are newer (813/813 on 2026-10-19), so a default local run archived every card into
   old_stats.pkl (BUGS_AND_IMPROVEMENTS B6). The default became remote on 2026-10-20 and the same day local
   mode was archived: `/update_stats` and `update_stats()` no longer take `use_local_repo` /
   `use_local_timestamp`, the traversal lives on branch `archive/local-update-stats` (not in main, so the bot
   can't import it; not compiled), and stored reminders drop the options (`RETIRED_OPTIONS`, analytics.py:41). Revive it only
   after fixing the change signal.
6. **Pickles-as-database implications** — no locking, no merges, no migrations, git as replication;
   see the section above for the full accounting. The mitigation gates live in
   **autocroissant-change-control**.
7. **`/pull` blocks the event loop.** `git_pull` is `async def` running blocking GitPython network
   calls directly on the loop (update_bot.py:42; `to_thread` passes coroutine functions through,
   utils.py:69-71). For `/update` and `/restart_bot` that's fine — the queues are stopped and the
   process is about to be replaced — but `/pull` (main.py:165-172) does NOT stop the queues, so all
   sends and heartbeats stall for the duration of the fetch/merge. Deliberate simplicity; know it's
   why the bot "freezes" briefly during a pull.
8. **The dispatch and edit loops can die for good** (BUGS_AND_IMPROVEMENTS B5). Their `except` catches
   only `HTTPException`/`AttributeError`, and the fallback itself is unguarded:
   `client.get_channel(channel_id)` returns `None` for a deleted or invisible channel, and `.send` on
   `None` raises inside the `except` block. An uncaught exception stops a `tasks.loop` permanently, and
   then nothing the bot queues is ever sent. Since 2026-10-19 reminders check `_channel_missing` before
   firing, so the scheduler no longer feeds the loop dead channels; every other path still can.
9. **Icon names are a cross-repo contract.** The parser (`ICON_FONT_PREFIX`, `ICON_MANIFEST_PATH`,
   `IconVocabulary`), the emoji names (`type_emojis.emoji_name`) and the website all derive names from
   TTSCardMaker's `Types/` file stems and the stoogedings manifest. Since 2026-10-20 the parser accepts
   exactly the font's spellings; emoji names and the website still normalize differently. Renaming or
   deleting an icon PNG breaks things on both sides without an error, because the manifest keeps the old
   entry (ttscardmaker-companion §1.6-1.7; BUGS_AND_IMPROVEMENTS C13).

## Where do I put new code (quick answers)

- **New slash command**: a plain sync function in the topical `commands/*.py` module; every output via
  `queue_*`; then in main.py the `func = to_thread(func)` rebind + `@tree.command` wrapper (invariant
  4). Full checklist with gates: **autocroissant-change-control**.
- **New recurring/background work**: do NOT add a fourth `tasks.loop`. One-shot deferred work →
  `queue_command(fn, ...)` (runs next tick in a thread). Recurring work on a schedule → prefer a
  reminder that runs a slash command (`/set_reminder ... frequency:1w command:/sync_emojis` is the
  2026-10-19 example); only if that can't work, follow the `check_reminder` pattern: cheap per-tick
  body, single-flight, heavy work queued onward (invariant 7).
- **Long-running work** (parsing sweeps, downloads): it already runs in a thread — report
  progress with `queue_edit`, expect your interaction token to expire, and rely on the fallback chain
  (write nothing that only works pre-expiry).
- **Work that must await Discord APIs other than sending messages** (emoji, guild management): an
  `async def` worker in the style of `sync_type_emojis` — await the Discord calls, thread every blocking
  step with `asyncio.to_thread`, report through `queue_message`.
- **Bounded parallelism**: pass `executor=<ThreadPoolExecutor>` through `queue_command` (main.py:742-743)
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
  function they decorate (Entry 10.13 in failure-archaeology). If a slash option goes away, add it to `RETIRED_OPTIONS` in analytics.py so saved reminders keep running
  (invariant 4).

## When NOT to use this skill

- **A live symptom to triage** (bot silent, commands missing, stats look wrong, music dead) →
  **autocroissant-debugging-playbook**. This skill explains why the design is what it is; that one
  gets you from symptom to cause.
- **Card-domain semantics** (folder classification, PSD layer rules, type injection, query DSL,
  exclusion lists) → **impossibility-cards-reference**.
- **Building the environment, Cython** → **autocroissant-build-and-env**.
- **Making a change** (checklists, pickle-push gates, what may never be committed) →
  **autocroissant-change-control**.
- **Incident history and the removeprefix bug's full story** → **autocroissant-failure-archaeology**.
- **Operating reminders and type emojis day to day** → **autocroissant-run-and-operate** §9, §11.
- **The TTSCardMaker side of icons, the font manifest and the website** → **ttscardmaker-companion**.
- **AI/diffusion doctrine and enablement** → **autocroissant-ai-boundary** (branch `archive/ai-generation`).

## Provenance and maintenance

Line numbers were verified 2026-07-11 at HEAD `284d13c` and re-verified 2026-10-20 at HEAD `9fb3093`
(expected values below are the 2026-10-20 readings). They WILL drift.
Re-find before citing:

| Fact | Re-verify with (cwd = repo root) |
|---|---|
| Three 1s loops + drain functions | `grep -n -e "tasks.loop" -e "async def process_" main.py` (749-750 / 758-759 / 790-791) |
| check_reminder re-queued every tick | `grep -n "queue_command(check_reminder)" main.py` (751) |
| Loops started in on_ready / stopped in restart/stop/update | `grep -n "process_.*_queue.st" main.py` (116-118 / 145-147 / 157-159 / 194-196) |
| Four queued inits incl. type emojis; reminder client binding | `grep -n "queue_command(init\|bind_reminder_client(client)" main.py` (111-114, 204) |
| The three deques | `grep -n "= deque()" commands/utils.py` (26-28) |
| queue_* helpers | `grep -n "^def queue_" commands/utils.py` (37/42/48/54/59) |
| to_thread coroutine passthrough | `grep -n -e "def to_thread" -e "iscoroutinefunction" commands/utils.py` (67, passthrough 69-71) |
| run_command + reserved `executor` kwarg | `grep -n "async def run_command" main.py` (740); `grep -rn "executor=executor" commands/music_player.py` |
| Registration rebind count | `grep -c "= to_thread(" main.py` (48) |
| slash_registry population | `grep -n "slash_registry\[cmd.name\]" main.py` (105) |
| Reminder resolve/bind/queue | `grep -n "slash_registry.get\|queue_command(_run_reminder_command" commands/analytics.py` |
| Reminders fire as creator; single-flight | `grep -n "_ReminderUser(\|_check_lock\|^LEGACY_RUNS_AS_BOT" commands/analytics.py` |
| FakeInteraction + default bot user id | `grep -n "def make_fake_interaction\|1011264680683110433" commands/utils.py` (107, 121) |
| perms_check inversion | `grep -n "def perms_check" commands/utils.py` (249); read the return at 259 |
| Duplicate defs | `grep -n -e "async def slash_set_ratio" -e "async def slash_delete_song" main.py` (304/314, 597/604) |
| Deque shuffle | `grep -n "shuffle(music_queue)" commands/music_player.py` (326) |
| Ephemeral dropped in fallback; unguarded get_channel | `grep -n 'pop("ephemeral"\|client.get_channel(interaction.channel_id).send' main.py` (779; 786, 803) |
| Temp files removed after parse (weak point 4 fixed) | `grep -n "mkstemp(\|remove(local_file)" commands/psd_analyzer.py` (1433, 1668) |
| Local mode archived 2026-10-20 (weak point 5) | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS`, analytics.py:41-44) |
| Module-level cycles + lazy breaks | `grep -n "from commands.query_card import card_repo" commands/psd_analyzer.py commands/type_emojis.py` (psd_analyzer 26 module level; type_emojis 60 inside a function); `grep -n "from commands.psd_analyzer import\|from commands.type_emojis" commands/query_card.py` (334, 379 prep_dataframes, 461 all inside functions; 15 module level) |
| Singletons | `grep -n "stats_db = StatsDatabase()" commands/psd_analyzer.py` (2288); `grep -n "card_repo = CardRepository()" commands/query_card.py` (714); `grep -n "state = MusicState()" commands/music_player.py` (45); `grep -n "^reminders = \|^type_emojis:" commands/analytics.py commands/type_emojis.py` (61, 26) |
| config import chain | `grep -ln "^import config" commands/*.py` (music_player, psd_analyzer, query_card, type_emojis, utils) |
| type_emojis uncompiled | `grep -c "type_emojis" setup.py` (0) |
| update_bot bypasses queues | `grep -n "followup.send" commands/update_bot.py`; `grep -n "from commands" commands/update_bot.py` (empty) |

Maintenance rule: if any grep above stops matching, the architecture moved — update this contract in
the same change, and check whether the corresponding invariant row still holds before assuming it does.
