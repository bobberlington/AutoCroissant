---
name: autocroissant-architecture-contract
description: Load this when you need to understand or safely modify HOW AutoCroissant works as a system — "how does the bot work", "start here", "orient me", "overview of this repo", "which skill do I need" (newcomer routing map at the top), "where do I put new code", "why is this function sync", "design", "invariants" — or any task touching threading, the async event loop, the command/dispatch/edit queues, tasks.loop, to_thread, queue_message/queue_file/queue_edit/queue_command, slash-command registration, slash_registry, reminder command execution, reminders running as their creator / lacks_admin at fire time, FakeInteraction, _ReminderUser, async workers that await Discord (type_emojis), module import order/cycles, module-level singletons, or the pickles-in-git database decision. Contains the execution model (discord.py event loop + three module-level deques drained by 0.1 s loops + sync workers in threads) and WHY it exists; an invariants table where each row says what breaks if you violate it; the module dependency map including the deliberate lazy-import cycle break; ASCII data flows for a slash command, the /update_stats pipeline, and a reminder firing; and the known-weak points stated plainly (ephemeral-dropping fallback, local-mode timestamp mismatch, pickle trade-offs, cross-repo icon contracts; duplicate names and the dying dispatch loop fixed 2026-10-20; the 1 s tick and in-place pickle saves 2026-10-21). Live-symptom triage belongs to autocroissant-debugging-playbook.
---

# AutoCroissant Architecture Contract

## New here? The 17-skill map

- Run / stop / deploy / machine handoff / run update_stats → autocroissant-run-and-operate
- Something is broken RIGHT NOW → autocroissant-debugging-playbook
- Install / env / Cython build → autocroissant-build-and-env ; every config axis → autocroissant-config-and-flags
- BEFORE any commit, push, or pickle touch → autocroissant-change-control ; is it safe/finished → autocroissant-validation-and-qa
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
`edit_queue`, utils.py:25-27). Three `@tasks.loop(seconds=0.1)` coroutines in `main.py`
(`process_command_queue` main.py:932, `process_dispatch_queue` main.py:945, `process_edit_queue`
main.py:957) drain those deques ten times a second **on the event loop** (once a second until
2026-10-21) and perform the actual Discord I/O, with a built-in fallback chain; a fourth loop,
`process_reminders` (main.py:926-928), keeps the reminder heartbeat at 1 s. That is the whole model: loop owns
Discord, threads own work, deques are the only bridge.

## The queue trio (this skill owns this detail)

| Queue (utils.py:25-27) | Entry shape | Filled by | Drained by | What draining does |
|---|---|---|---|---|
| `command_queue` | `((args, kwargs), func)` | `queue_command(func, *a, **kw)` (utils.py:58) | `process_command_queue` (main.py:932-941) | `create_task(run_command(...))` per entry — each func runs in a worker thread of asyncio's default pool (threads are reused: 2 alive at idle, measured 2026-10-20), or in a given threadpool (see below) |
| `dispatch_queue` | `(interaction, send_kwargs)` | `queue_message` (utils.py:41), `queue_file` (utils.py:47), `queue_any` (utils.py:36) | `process_dispatch_queue` (main.py:945-954; `_dispatch` 885-899, which also turns a `links` kwarg into link buttons) | Sends via fallback chain: `interaction.response.send_message` → `interaction.followup.send` → `client.get_channel(channel_id).send` |
| `edit_queue` | `(interaction, edit_kwargs)` | `queue_edit` (utils.py:53) | `process_edit_queue` (main.py:957-964; `_edit` 902-923) | `interaction.edit_original_response`, falling back to a fresh `channel.send` |

Mechanics you must know:

- **`to_thread(func)`** (utils.py:80-89) wraps a sync function so awaiting it runs it in a thread via
  `asyncio.to_thread`. Critical subtlety: if `func` is already a coroutine function it is returned
  **unchanged** (utils.py:82-84). This passthrough is what lets `run_command` (main.py:836-844) execute
  both sync workers and the async slash callbacks stored in `slash_registry` with one code path.
- **`to_threadpool(func, executor)`** (utils.py:92-101) is the bounded variant. `run_command` routes to
  it when the queued kwargs contain `executor` (main.py:839-840) — so **`executor` is a reserved kwarg
  name** for anything sent through `queue_command`; a worker function with its own `executor` parameter
  would have it silently popped. Sole current user: music downloads, capped at a 2-worker
  `ThreadPoolExecutor` (music_player.py:35, used at music_player.py:264 and 277) so a playlist can't
  spawn unbounded yt-dlp threads.
- **Queued commands run concurrently** (one `create_task` + thread per entry), so two long workers that must not overlap
  need their own lock: `card_runner.bake_lock` (2026-10-21) is held by `update_stats` for its whole run (repo_traverser.py:84-87)
  and by the card bake around `card_bake.run`; the one that finds it held says so and blocks only its own thread. Nothing
  queues a bake on its own: the `notice_edits` hook at the end of `update_stats` was removed 2026-10-21 (the user: the Action is
  the one live runner, `/bake_cards` the fallback by hand).
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
   on `HTTPException`/`AttributeError` a raw `channel.send` (`_fallback_send`, main.py:870-882) — means a late result
   degrades to a plain channel message instead of vanishing. Every send written as `queue_*` gets this
   for free; every send written by hand does not.
3. **Serialization + self-throttling.** One drain point (every 0.1 s since 2026-10-21, 1 s before)
   batches sends; discord.py's own rate-limit handling covers the rest without explicit rate-limit code.
   Cost: up to ~100 ms latency per hop, and a reply that traverses two queue hops (see the reminder
   diagram) up to ~200 ms. That is the accepted trade at hobby scale.
4. **Deferred init without blocking login.** `on_ready` (main.py:115-144) queues six commands
   (main.py:134-139) instead of running them: `sweep_temp_psds` first (2026-10-22, c0fc117: deletes the mkstemp
   `tmp*.psd` files older than 1 h that a cut-off run left, under `card_runner.bake_lock`), then `init_reminder`,
   `init_psd`, `init_query`, `refresh_history`, `init_type_emojis(client)`. Pickle loads and the GitHub tree fetch
   happen in worker threads after the bot is already up. `init_type_emojis` is async: it runs on the loop and threads
   its own downloads. The inits run **once** (the `_started` flag, 778c2ce, 2026-10-21): discord.py calls `on_ready`
   again after a failed RESUME, and that repeat only syncs any guild not synced yet, restarts the loops whose task
   failed and prints `Reconnected: inits skipped, revived <loops>` (main.py:125-128). Loops stopped on purpose (by
   `/update`) stay stopped. main.py also calls `bind_reminder_client(client)` at import time (main.py:236) so the
   scheduler can tell when a reminder's channel is gone.

## Invariants — violate this and X happens

| # | Invariant | Violate it and... |
|---|---|---|
| 1 | **No awaits inside queued sync functions.** Workers are plain `def`; there is no event loop in their thread. | `await` in a `def` is a SyntaxError — so the tempting "fix" is making the worker `async def`. Do that and `to_thread` passes it through unchanged (utils.py:82-84): your "worker" now runs ON the event loop, and its blocking calls (PSD parse, yt-dlp, GitPython) freeze the entire bot — heartbeats, all three queue loops, everything — until it finishes. |
| 2 | **All Discord output from worker code goes through `queue_message`/`queue_file`/`queue_edit`/`queue_any`/`queue_command`** (utils.py:36-60). | Direct calls from a thread produce never-awaited coroutines (message silently lost) or thread-unsafe loop access. You also forfeit the fallback chain: your send dies permanently at the 15-minute token expiry instead of degrading to `channel.send`, and reminders (which run your command with a FakeInteraction) break because FakeInteraction only implements the queue-facing surface. |
| 3 | **The three queue loops must be running for ANYTHING to send.** Started once in `on_ready` through `_start()` (main.py:141-142; `_loops()`/`_start()` at main.py:967-979); a repeat `on_ready` (reconnect) restarts only the loops whose task failed. Deliberately STOPPED by `/restart_bot` (main.py:170-173), `/stop_bot` (main.py:183-186), `/update` (main.py:225-228) so no queued output races process replacement; a reconnect does not revive those. | With a loop stopped (or `on_ready` never reached), `queue_*` calls accumulate silently in the deques forever — the bot looks alive and says nothing. This is also why `commands/update_bot.py` is the one module that bypasses the queues entirely: its functions are `async def` awaiting `interaction.followup.send` directly (`git_push` update_bot.py:78, `git_pull` :113, `update_bot` :212), because during `/update` the dispatch loop is already stopped and queued progress reports would never flush. |
| 4 | **Registration pattern:** `func = to_thread(func)` rebind FIRST, then the `@tree.command` async wrapper that awaits it (50 occurrences as of 2026-10-23; e.g. main.py:283, 405). `on_ready` stores `slash_registry[cmd.name] = cmd.callback` (main.py:130-131), which powers reminder command execution (analytics.py `_resolve_command`, `_fire`). | Forget the rebind line → the wrapper's `await func(...)` executes the sync function inline ON the event loop (bot frozen for its whole duration), then raises `TypeError` awaiting the plain return value. Since 2026-10-19 reminders bind their stored `/name option:value` text against the callback's real signature at creation AND at every firing, so: rename a command's `name=` → saved reminders post "⚠ Reminder `<id>` skipped its command: `/<name>` is not a bot command"; rename or remove a slash **parameter** → "`<option>` is not one of its options", UNLESS you list it in `RETIRED_OPTIONS` (analytics.py:41; with the command's old option order), which makes stored reminders drop it at load and at fire time — done for /update_stats's `use_local_repo`/`use_local_timestamp` on 2026-10-20; change a parameter's type → values that no longer convert are refused. The reminder itself keeps repeating; its creator must re-create it. The wrapper function's Python name, by contrast, is decorative — see weak point 1. |
| 5 | **One bot per process; state is module-level singletons.** `stats_db` (card_models.py:409, re-exported by psd_analyzer), `card_repo` (query_card.py:990), `state`/`music_queue`/`prev_music`/`executor` (music_player.py:32-45), `reminders` + `_client` + `_lock`/`_check_lock` (analytics.py:61-66), `type_emojis` (type_emojis.py:31, token → emoji map filled by the sync), `slash_registry` + the three deques (utils.py:25-29). | A second client in-process, or importing a commands module under a second name, gives you split-brain state. Import order is load-bearing: the singletons are constructed at import time (cheap — heavy loading is deferred to the queued `init_*` functions), and (since 2026-10-20) none of them prints: the single git-token-presence line comes from `print_token_status()` in main.py. |
| 6 | **Importing `commands.*` requires `config.py` to exist.** utils and music_player `import config` at module level and github_api reads it in a try/except (so `python -m commands.tts_save` runs without it); every card module reaches utils: psd_analyzer → repo_traverser → query_card → type_emojis → utils (since the 2026-10-20 split). | Any tool that unpickles `stats.pkl` must have this repo on `sys.path` AND a config.py present — `CardInfo`/`CardStats` are dataclasses defined in `commands.card_models` with `__module__` pinned to `commands.psd_analyzer` (the facade re-exports them), so unpickling imports the facade and the chain reaches `config`. This is why the diagnostics scripts run from the repo root; code outside this repo must stub the classes, as TTSCardMaker's website build does (`_Unpickler` in site/build/build_site.py). config.py is SECRET — never print or read its values. |
| 7 | **`check_reminder` is re-queued every second — reminder-path work must stay cheap, and stay single-flight.** `process_reminders` (main.py:926-928, a 1 s `tasks.loop` started and stopped with the drains) does `queue_command(check_reminder)`; the command loop drains it within 0.1 s, so it runs ~once/second in a pool thread (analytics.py:819). Since 2026-10-19 it takes `_check_lock` with a non-blocking acquire (a pass that finds another still running just returns) and does all work under `_lock` (an RLock shared with set/list/remove, which run in their own threads). Commands of due reminders are queued onward (`queue_command(_run_reminder_command, ...)`), never run inline; reminder.pkl is saved only when something changed, atomically. | Anything slow in `check_reminder` costs a pool thread every second and holds `_lock`, stalling `/set_reminder`, `/list_reminders` and `/remove_reminder`. Remove the single-flight lock and two overlapping passes can fire the same reminder twice. Run a reminder's command inline and one slow command blocks every other reminder. |
| 8 | **`lacks_admin` (named `perms_check` until 2026-10-20) returns True when the user is NOT an admin:** `return interaction.user.id not in ADMINS` (utils.py). Callers write `if lacks_admin(i): deny-and-return` (e.g. main.py:180-182). | Flip the polarity and you either lock the admins out or open an admin command to everyone. When adding an admin-gated command, copy an existing gate verbatim. |
| 9 | **Admin gates live in the registered slash callback, and reminders fire as their creator** (2026-10-19). A firing reminder's FakeInteraction carries `_ReminderUser(creator_id)` (analytics.py:69, set in `_reminder_interaction`), so the callback's own `lacks_admin` re-checks ADMINS at fire time. Legacy reminders without a creator keep the default FakeUser — the bot itself (`LEGACY_RUNS_AS_BOT`, analytics.py:37). | Put an admin check anywhere other than the callback stored in `slash_registry` (e.g. only in a worker, or in a pre-dispatch hook) and scheduled runs either skip it or check the wrong user. Make the FakeInteraction default to the bot again and every reminder passes admin checks (the bot's id is in ADMINS on the dev Mac — archaeology Entry 10.8). |
| 10 | **`update_stats` and the card bake never overlap: both take `card_runner.bake_lock`** (2026-10-21; after the cutover /update_stats parses nothing and takes no lock). `update_stats` wraps its whole run in it (repo_traverser.py:79-82); `card_runner.bake_cards` takes it around `card_bake.run`. The bake imports card_bake and psd_edit inside the function and refuses with one line when the fork's text engine is missing. | Drop the lock and a bake (a PSD render and save, GBs of RAM: design section 8) runs beside a full re-parse on the jetson's 4 cores, and `/update_stats` can re-read a tree the bake is committing to mid-run. Import card_bake at module level and the bot fails to start wherever the fork's psd-tools is absent (the Mac's stock venv, the Actions that install stock psd-tools); the jetson has had the fork since 2026-10-22. |

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
dispatch_queue (deque, utils.py:25)
      |
      v  (next 0.1 s tick, back on the event loop)
process_dispatch_queue (main.py:945)
      |- interaction.response.send_message(**kw)     if response not yet done
      |- interaction.followup.send(**kw)             if already responded/deferred
      '- client.get_channel(channel_id).send(**kw)   on HTTPException/AttributeError
      |                                              (token expired, or fake interaction;
      v                                               NOTE: drops `ephemeral` — weak point 3)
Discord
```

### (b) `/update_stats` pipeline (PSDs -> parser -> pickles -> dataframes)

```
/update_stats  (wrapper defers, main.py:422)
      v
WORKER THREAD: update_stats() (repo_traverser.py:61; the traversal is card_data.update, card_data.py:1117, since 2026-10-22)
      |  one-time history cleanup/recovery first (card_data.cleaned, 2026-10-23): HISTORY_CLONE (a full clone)
      |    + the bot's PICKLE commits (BOT_CLONE, else this checkout); recover, normalize, unknown_values, misreads,
      |    dedupe, date_archive, commits_of on a COPY; one "History: ..." line, or "History cleanup skipped (...)":
      |    an error never fails the run. The recovery runs once (recovered_before)
      |  GitHub trees API with If-None-Match: 304 -> "Nothing changed in the card repo since the
      |    last update." and the run ends (ETag cache in memory only; card_data.py:267-272)
      |  build the icon vocabulary: Types/ + Types/Stars/ names + the stoogedings manifest
      |    (MDW/Fonts/stoogedings-src/stoogedings.json; impossibility-cards-reference §4.1)
      |  pick PSDs (GitHub only; local mode archived 2026-10-20, weak point 5): new, moved, blob sha
      |    != card.blob_sha (_psd_changed, card_data.py:237-249; commits-since fallback for
      |    cards without blob_sha), named in cards:, or force_update
      |  DOWNLOAD_WORKERS=4 fetch up to AHEAD=8 PSDs ahead of parsing: with a lazy psd-tools, range
      |    requests at the tree's commit (RemotePSD, ~2.5% of the bytes); else, or on any failure, the
      |    whole blob (git/blobs API) into a mkstemp file (card_data _fetch_psd, _download)
      v
per picked PSD: one commit lookup (_get_remote_timestamp); one PSDImage open for PSDParser.parse and
  psd_details.gather (the website's Details record, by blob sha; a failure is one console line); release; a lookup
  or parse that raises fails that card only (it keeps its last version; no ETag kept, so the next run retries);
  old CardInfo -> old_stats only if its blob changed or it moved and its data did (after edits); CardValidator
      v
      |  then the history cleanup once more over what the run read (a second History line when it finds something)
      v
stats_db singleton (card_models.py:409) --save()--> stats.pkl / old_stats.pkl   (the DATABASE)
      |  (save only when something changed: num_changed, archives, build.changed, a History line, stale archive
      |   tags; card_data.py:1158)
      |  psd_details.json: pruned to the tree's blobs and saved when a record was gathered or pruned
      |    (loaded at the start of the run; a 304 touches neither file)
      v
changed tree only: queue_command(sync_changed_emojis)   (until 2026-10-20 also a TTS save run when images, the
      |   rules, Card Edits.json or the save changed; TTSCardMaker's tts-save.yml Action does that now)
queue_command(refresh_history); queue_command(card_repo.populate_files); queue_command(card_repo.prep_dataframes)
      |                                            (repo_traverser.py:115-122)
      |   (no bake is queued here: the edits-file notice was removed 2026-10-21; /bake_cards is by hand, the Action the live runner)
      v  (subsequent command-queue ticks, pool threads)
card_repo dataframes rebuilt  ->  /query and /query_ability now serve the new data
      |
      '--- progress and problem reports stream back throughout via queue_edit / queue_message

Downstream, outside this process: /push commits the pickles and psd_details.json (a PICKLE commit); /pull may add one
more after a merge that dropped a card from both pickles (the rescue, 2026-10-22); TTSCardMaker's website build reads
them from this repo's main (ttscardmaker-companion §2).
```

### (b2) `/update_stats` after the cutover (`CARD_SYNC` on; merged and deployed off 2026-10-23, flips at the cutover)

```
/update_stats  (wrapper: with force_update or cards, admins only; then defers)
      v
WORKER THREAD: update_stats() (repo_traverser.py) -> global_config.CARD_SYNC? -> card_sync.update()   (uncompiled)
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

Nothing parses and nothing is written but `card_data/` (gitignored): no bake_lock, no pickle, no /push. Startup
(`init_psd` -> `card_sync.startup`) loads `card_data/` first and syncs once; the root pickles only when neither works.

### (c) A reminder firing (slash_registry + FakeInteraction), as of 2026-10-19

```
process_reminders 1 s tick (main.py:874-876) -> queue_command(check_reminder)
      v  next command-drain tick (main.py:880)
check_reminder() in a worker thread (analytics.py:819)
      |  _check_lock non-blocking acquire (overlapping pass -> return) ; then under _lock:
      |  for each due, enabled reminder -> _process_due -> _fire (analytics.py:647):
      |- channel gone? (_channel_missing: bot ready AND get_channel is None) -> skip, log
      |- fake = make_fake_interaction(channel, guild); fake.user = _ReminderUser(creator_id)
      |         (legacy reminder without creator: fake.user stays the bot's FakeUser)
      |- msg?     queue_message(fake, chunk) per chunk
      '- command? _resolve_command(stored text): shlex-split, slash_registry.get(name),
                  drop RETIRED_OPTIONS (/update_stats use_local_repo/use_local_timestamp),
                  bind name:value options against the callback's signature and convert them
                  -> queue_command(_run_reminder_command, callback, fake, values, id, name)
                        |
                        v  next tick: run_command (main.py:784) -> coroutine passthrough
                  _run_reminder_command awaits callback(fake, **values) on the event loop;
                  the callback runs lacks_admin(fake) as the CREATOR (invariant 9), then awaits
                  its to_thread'd worker exactly like a real invocation. Exceptions are reported
                  in the channel ("⚠ Reminder <id>: /<name> failed ...").
      then: repeating -> when = _next_run(when, frequency, now) (one jump, no replay);
            one-shot delivered -> removed; one-shot whose channel is gone -> retry in 1 h;
            any change -> save_reminders() (temp file + fsync + rename)
```

**FakeInteraction (utils.py:124-191) — what and why.** Both send-queues store
`(interaction, kwargs)` and the drain loops call interaction methods on them. A reminder fires with
no real interaction — no token, no response object. `make_fake_interaction(channel_id, guild_id)`
builds an object with just enough surface (`response.is_done/send_message/defer`, `followup.send`,
`edit_original_response`, `user`, `channel_id`) so worker code written against real interactions runs
unmodified. Its default `user` is a FakeUser with the bot's own id (1011264680683110433); reminders
replace it with `_ReminderUser` (invariant 9), which also carries `name`, `display_name`, `mention`
and `bot=False`. Its delivery trick is deliberate: `FakeResponse.send_message` doesn't send — it
**re-queues** the kwargs via `queue_any` with a bare `discord.Object` carrying only `channel_id`
(utils.py:149-155). When the dispatch loop pops that Object, `interaction.response` raises
`AttributeError`, which lands in the fallback branch → `client.get_channel(channel_id).send(...)` —
the one send that needs no interaction. FakeInteraction rides the error path **by design**; do not
"clean up" the dispatch loop's `AttributeError` handling without rerouting reminders first.

## Module dependency map

```
config.py  (SECRET - tokens; names only)      global_config.py  (paths, pickle names, TIMEZONE)
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
 |                            card_models (save_pickle, stats_db), card_edits (ALIASES_MOVED), card_history (start);
 |                            psd_analyzer LAZILY (the facade-only names)
 |- commands.type_emojis   -> utils, global_config, github_api, repo_history; query_card.card_repo LAZILY inside _list_type_icons
 |                            [2026-10-19; NOT in setup.py INCLUDE_FILES, so never compiled]
 |- commands.repo_history  -> global_config ONLY  [2026-10-20; commit-pinned image links; NOT compiled]
 |- commands.git_commit    -> github_api, repo_history, global_config: one commit on the card repo (Git Data API; git objects in the
 |                            history clone pushed over SSH when the token can't write), shared by tts_save and the edit pipeline's
 |                            card_bake [2026-10-21; NOT compiled; no discord or config]
 |- commands.card_runner   -> utils, global_config (discord.File); card_bake + psd_edit LAZILY inside bake_cards: bake_lock,
 |                            bake_cards (/bake_cards)  [2026-10-21; NOT compiled; the notice_edits hook removed the same day]
 |- commands.tts_save      -> git_commit, repo_history, card_models, card_edits, global_config; utils + discord LAZILY inside the
 |                            posting functions, so `python -m commands.tts_save <save.json>` runs standalone (TTSCardMaker's
 |                            tts-save.yml Action does, with the AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE env
 |                            overrides global_config reads) [2026-10-20; TTS save updater + /update_tts_save; NOT compiled]
 |- commands.card_models   -> global_config ONLY: CardType, CardStats, CardInfo (the pickled classes, __module__ pinned to
 |                            commands.psd_analyzer), StatsDatabase, stats_db, card_name  [2026-10-20 split; compiled]
 |- commands.card_edits    -> card_models: CardAuthors, CardEdits  [compiled]
 |- commands.psd_parser    -> card_models, global_config, psd_tools: PSDParser, CardClassifier, CardValidator,
 |                            IconVocabulary  [compiled; no Discord, network or pandas]
 |- commands.psd_details   -> font_check, psd_regions, psd_parser, card_models, github_api, global_config, psd_tools (Pillow's ImageCms
 |                            optional): gather, load/save/prune, the backfill CLI  [2026-10-21; NOT compiled; no discord or config;
 |                            card_data imports it LAZILY inside its functions so fontTools stays off the startup path]
 |- commands.card_data     -> card_models, card_edits, psd_parser, github_api, global_config; psd_details, card_history, card_db,
 |                            changelog LAZILY: CardBuild (the traversal), update (legacy /update_stats), build (the Card sync
 |                            Action's db/ JSON)  [2026-10-22; NOT compiled; no discord, pandas, config or utils: card-sync.yml
 |                            checks out card_data.SPARSE_FILES alone]
 |- commands.card_sync     -> card_db, github_api, psd_analyzer (stats_db, the facade's), utils, global_config, config (optional);
 |                            query_card, type_emojis, repo_history, analytics, psd_details, font_check LAZILY: the bot after the
 |                            cutover (sync, /update_stats, Card sync dispatches, /fonts from the data). Imported LAZILY by
 |                            repo_traverser, psd_analyzer and font_check, only when global_config.CARD_SYNC is on
 |                            [2026-10-22; NOT compiled]
 |- commands.repo_traverser -> card_data, card_edits, card_models, card_runner (bake_lock), utils, query_card, repo_history,
 |                            type_emojis (module level!; tts_save no longer, 2026-10-20): RepositoryTraverser (CardBuild with the
 |                            bot's aliases), update_stats, _tree_etags  [compiled; a thin Discord wrapper since 2026-10-22]
 |- commands.psd_analyzer  -> the four above, query_card, github_api, utils, pandas: the FACADE. Re-exports every card name, holds the
 |                            metadata/export commands and init_psd. Code outside imports card classes from HERE only.
 |- commands.discord_edits -> card_edits, card_models, github_api, global_config; card_bake, card_runner, card_data,
 |                            card_history, psd_analyzer, query_card, utils LAZILY: /update_metadata, /mass_replace, /alias
 |                            write Card Edits.json (main.py:23)  [2026-10-22; NOT compiled]
 |- commands.frankenstein  -> utils, query_card (card_repo)
 '- commands.update_bot    -> global_config, repo_history (batch_overrides); card_models LAZILY (the /pull rescue);
                              GitPython LAZILY inside its functions  (no utils: bypasses queues on purpose, invariant 3)

Outside the bot's import graph (CLIs and TTSCardMaker Actions only; no bot module imports them): card_check,
card_guards, card_standards, png_export, psd_diff, txt2_repair, discord_archive, discord_match. card_db, card_history and
changelog are in it, reached from card_data, card_sync, query_card and discord_edits (the data build's modules).

(not in main) branch archive/local-update-stats: archive/local_update_stats.py  (2026-10-20; NOT compiled)
     -> commands.psd_analyzer, commands.query_card, global_config   [the archived local /update_stats mode]
```

**The deliberate lazy-import cycle breaks (two of them).** `repo_traverser` (and the facade `psd_analyzer`) import
`card_repo` from `query_card` at module level because they use the repo object throughout. `query_card` takes
`stats_db` and `save_pickle` from `card_models` at module level (query_card.py:20, since 5835139, 2026-10-21):
card_models imports nothing of the facade, so that is no cycle, and every deploy rebuilds every `.so`, so the old
"two `stats_db` objects" risk of a stale compiled facade is gone. The names only the facade holds
(`get_stats_as_dict`, `CardEdits` through it, `_format_metadata`) it imports **inside the function**
(query_card.py:441, 510, 598, 606, 733, 762: `_populate_edit_names`, `prep_dataframes`, the card lookups). The same shape repeats since 2026-10-19: `query_card` imports `render_type_tokens` from
`type_emojis` at module level, and `type_emojis` needs `card_repo` only when listing icons, so `_list_type_icons`
imports it inside the function. Hoist either import to module level and you get a circular-import crash at startup
(psd_analyzer → repo_traverser → query_card → psd_analyzer before it finished executing). The lazy side is the one
whose need is deferred anyway — keep it that way.

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

**Its successor: merged and deployed off 2026-10-23; the Card sync shadow schedule is on (TTSCardMaker 96d61368);
CARD_SYNC flips at the cutover.** The Card sync Action in TTSCardMaker builds the card data as JSON on that repo's `data`
branch (docs/Campaigns/DATA_BUILD_ACTION.md); the bot only reads it into `card_data/` and memory (data flow b2). The bot
side (work/cutover-bot, 7b99438) is on main with `CARD_SYNC = False`. One constant, `CARD_SYNC` in global_config.py,
flipped in the cutover commit, moves the bot from this section's design to that one; until then the pickles above stay
the database, and step 10 deletes them.

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
3. **The dispatch fallback drops `ephemeral`** (main.py:872, in `_fallback_send`: `kwargs.pop("ephemeral", None)`
   before `channel.send`). A reply the user expected to be private — e.g. `/list_reminders`
   `hidden:True` — posts publicly to the channel if the send lands in the fallback path (token
   expired, or any FakeInteraction send). `channel.send` simply has no ephemeral concept.
4. **(Fixed 2026-10-19) Remote traversal used to leak temp files.** Each downloaded temp file is now
   removed right after parsing (`_release`, card_data.py:733-738, called from the `finally` blocks at :644-645 and
   :677-686; `_download` removes its own on a failed write, :176-186). Since 2026-10-22 startup also sweeps the ones a
   cut-off run left (`sweep_temp_psds`, invariant table WHY #4). Kept here as a pointer: archaeology Entry 10.2.
5. **(Retired 2026-10-20) Local mode's change signal did not match the stored data.** The local-mode PATH
   bug (removeprefix, 2025-12-02 → 2026-10-19) was fixed with `relpath` (story:
   **autocroissant-failure-archaeology** Entry 1). What remained: with `use_local_timestamp:True` (the
   default) local mode compared file mtimes with timestamps that came from GitHub commit dates, and every
   clone's mtimes are newer (813/813 on 2026-10-19), so a default local run archived every card into
   old_stats.pkl (docs/BUGS_AND_IMPROVEMENTS.md B6). The default became remote on 2026-10-20 and the same day local
   mode was archived: `/update_stats` and `update_stats()` no longer take `use_local_repo` /
   `use_local_timestamp`, the traversal lives on branch `archive/local-update-stats` (not in main, so the bot
   can't import it; not compiled), and stored reminders drop the options (`RETIRED_OPTIONS`, analytics.py:41). Revive it only
   after fixing the change signal.
6. **Pickles-as-database implications** — no locking, no merges, no migrations, git as replication;
   see the section above for the full accounting. The mitigation gates live in
   **autocroissant-change-control**.
7. **`/pull` blocks the event loop.** `git_pull` is `async def` running blocking GitPython network
   calls directly on the loop (update_bot.py:113; `to_thread` passes coroutine functions through,
   utils.py:82-84). For `/update` and `/restart_bot` that's fine — the queues are stopped and the
   process is about to be replaced — but `/pull` (main.py:190-198) does NOT stop the queues, so all
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
   drains tick every 0.1 s since 2026-10-21 (reminders keep a 1 s heartbeat; idle 1.3 ms CPU/s). /query_ability
   listed name and image link 10 per message from 2026-10-21 until the owner reverted it on 2026-10-23
   (Discord previews only a few images per message): it sends one message per card again, the image URL alone
   (archaeology 10.30; `docs/Finished/optimization-pass-2026-10-20/profile.md` items 1-2). What remains:
   the dispatch loop awaits sends one after another, so a big result set (`limit` caps it) holds other
   replies behind it.
11. **Pickle saves are atomic and locked, but not transactional across pickles.** Since 2026-10-21
   `StatsDatabase.save` (card_models.py:170) and `save_aliases` (query_card.py:435) go through
   `save_pickle` (card_models.py:22-37: temp file beside the target, flush + fsync, `os.replace`) under one
   `threading.Lock` per pickle, like reminder.pkl; a kill mid-write leaves the old file whole. Still true:
   the two pickles are saved one after the other (old_stats.pkl first since 2026-10-21, ae1d2c5, so a kill between them never
   leaves a card unlisted and untagged), and a `/push` during a save commits whichever complete file is there
   (archaeology 10.29).

## Where do I put new code (quick answers)

- **New slash command**: a plain sync function in the topical `commands/*.py` module; every output via
  `queue_*`; then in main.py the `func = to_thread(func)` rebind + `@tree.command` wrapper (invariant
  4). Full checklist with gates: **autocroissant-change-control**.
- **New recurring/background work**: do NOT add a fourth `tasks.loop`. One-shot deferred work →
  `queue_command(fn, ...)` (runs next tick in a thread). Recurring work on a schedule → prefer a
  reminder that runs a slash command (`/set_reminder ... frequency:1d command:/update_stats` is the
  usual example); only if that can't work, follow the `check_reminder` pattern: cheap per-tick
  body, single-flight, heavy work queued onward (invariant 7).
- **Long-running work** (parsing sweeps, downloads): it already runs in a thread — report
  progress with `queue_edit`, expect your interaction token to expire, and rely on the fallback chain
  (write nothing that only works pre-expiry).
- **Work that must await Discord APIs other than sending messages** (emoji, guild management): an
  `async def` worker in the style of `sync_type_emojis` — await the Discord calls, thread every blocking
  step with `asyncio.to_thread`, report through `queue_message`.
- **Bounded parallelism**: pass `executor=<ThreadPoolExecutor>` through `queue_command` (main.py:787-788)
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

Line numbers were verified 2026-07-11 at HEAD `284d13c`, re-verified 2026-10-20 at HEAD `9fb3093`, re-mapped after that
day's optimization pass (psd_analyzer split, `refs6.py`) and re-verified 2026-10-23 at `2ada357`
(expected values below are the 2026-10-23 readings). They WILL drift.
Re-find before citing:

| Fact | Re-verify with (cwd = repo root) |
|---|---|
| Three 0.1 s drains + the 1 s reminder loop | `grep -n -e "tasks.loop" -e "async def process_" main.py` (926-927 reminders at 1 s; 932-933 / 945-946 / 957-958 at 0.1 s) |
| check_reminder queued by its own 1 s loop | `grep -n "queue_command(check_reminder)" main.py` (928, inside `process_reminders`) |
| Loops started once in on_ready (revived on a reconnect) / stopped in restart/stop/update | `grep -n "_start(loop)\|def _loops" main.py` (126 revive, 142 start, 967); `grep -n "process_.*\.stop" main.py` (170-173 / 183-186 / 225-228) |
| Six queued inits (sweep_temp_psds first); run once; reminder client binding | `grep -n "queue_command(sweep_temp_psds\|queue_command(init\|queue_command(refresh_history\|bind_reminder_client(client)\|_started = True" main.py` (134-139, 143, 236) |
| The three deques | `grep -n "= deque()" commands/utils.py` (25-27) |
| queue_* helpers | `grep -n "^def queue_" commands/utils.py` (36/41/47/53/58/63) |
| to_thread coroutine passthrough | `grep -n -e "def to_thread" -e "iscoroutinefunction" commands/utils.py` (80, passthrough 82-84) |
| run_command + reserved `executor` kwarg | `grep -n "async def run_command" main.py` (836); `grep -rn "executor=executor" commands/music_player.py` |
| Registration rebind count | `grep -c "= to_thread(" main.py` (50 on 2026-10-23) |
| update_stats and the card bake share one Lock; the bake is lazy about the fork | `grep -n "bake_lock" commands/repo_traverser.py commands/card_runner.py`; `grep -c "notice_edits" commands/repo_traverser.py commands/card_runner.py` (0 and 0); `grep -n "^import commands.card_bake\|^from commands.card_bake" commands/card_runner.py` (empty: imported inside `bake_cards`) |
| slash_registry population | `grep -n "slash_registry\[cmd.name\]" main.py` (131) |
| Reminder resolve/bind/queue | `grep -n "slash_registry.get\|queue_command(_run_reminder_command" commands/analytics.py` |
| Reminders fire as creator; single-flight | `grep -n "_ReminderUser(\|_check_lock\|^LEGACY_RUNS_AS_BOT" commands/analytics.py` |
| FakeInteraction + default bot user id | `grep -n "def make_fake_interaction\|1011264680683110433" commands/utils.py` (124, 138) |
| lacks_admin polarity | `grep -n -A2 "def lacks_admin" commands/utils.py` (returns `not in ADMINS`) |
| Unique handler names (was duplicate defs) | `grep -n -e "async def slash_set_re" -e "async def slash_delete_" main.py` (4 distinct names) |
| Deque shuffle | `grep -n "shuffle(music_queue)" commands/music_player.py` (338) |
| Ephemeral dropped in fallback; get_channel None guarded | `grep -n 'pop("ephemeral"\|def _fallback_send\|not found")' main.py` (872, 870, 879) |
| Temp files removed after parse (weak point 4 fixed); startup sweep | `grep -n "mkstemp(\|remove(" commands/card_data.py` (176, 186, 738); `grep -n "def sweep_temp_psds\|^TEMP_" commands/card_runner.py` (27-28, 103) |
| Local mode archived 2026-10-20 (weak point 5) | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS`, analytics.py:41-44) |
| Module-level cycles + lazy breaks | `grep -n "from commands.query_card import card_repo" commands/repo_traverser.py commands/psd_analyzer.py commands/type_emojis.py` (module level in the first two; inside a function in type_emojis); `grep -n "from commands.psd_analyzer import" commands/query_card.py` (all inside functions: 442, 511, 599, 607, 734, 763); `grep -n "^from commands" commands/query_card.py` (18-24: card_edits, card_history, card_models, github_api, repo_history, type_emojis, utils) |
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
