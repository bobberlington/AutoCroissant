---
name: autocroissant-architecture-contract
description: Load this when you need to understand or safely modify HOW AutoCroissant works as a system — "how does the bot work", "start here", "orient me", "overview of this repo", "which skill do I need" (newcomer routing map at the top), "where do I put new code", "why is this function sync", "design", "invariants" — or any task touching threading, the async event loop, the command/dispatch/edit queues, tasks.loop, to_thread, queue_message/queue_file/queue_edit/queue_command, slash-command registration, slash_registry, reminder command execution, reminders running as their creator / lacks_admin at fire time, FakeInteraction, _ReminderUser, async workers that await Discord (type_emojis), module import order/cycles, module-level singletons, or the pickles-in-git database decision. Contains the execution model (discord.py event loop + three module-level deques drained by 1-second loops + sync workers in threads) and WHY it exists; an invariants table where each row says what breaks if you violate it; the module dependency map including the deliberate lazy-import cycle break; ASCII data flows for a slash command, the /update_stats pipeline, and a reminder firing; and the known-weak points stated plainly (ephemeral-dropping fallback, local-mode timestamp mismatch, pickle trade-offs, cross-repo icon contracts; duplicate names and the dying dispatch loop fixed 2026-10-20). Live-symptom triage belongs to autocroissant-debugging-playbook.
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
`main` HEAD `9fb3093`, and again after that day's optimization pass (module split, robustness fixes).
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
`edit_queue`, utils.py:25-27). Three `@tasks.loop(seconds=1)` coroutines in `main.py`
(`process_command_queue` main.py:839, `process_dispatch_queue` main.py:853, `process_edit_queue`
main.py:865) drain those deques once per second **on the event loop** and perform the actual Discord
I/O, with a built-in fallback chain. That is the whole model: loop owns Discord, threads own work,
deques are the only bridge.

## The queue trio (this skill owns this detail)

| Queue (utils.py:25-27) | Entry shape | Filled by | Drained by | What draining does |
|---|---|---|---|---|
| `command_queue` | `((args, kwargs), func)` | `queue_command(func, *a, **kw)` (utils.py:58) | `process_command_queue` (main.py:838-849) | `create_task(run_command(...))` per entry — each func runs in a worker thread of asyncio's default pool (threads are reused: 2 alive at idle, measured 2026-10-20), or in a given threadpool (see below) |
| `dispatch_queue` | `(interaction, send_kwargs)` | `queue_message` (utils.py:41), `queue_file` (utils.py:47), `queue_any` (utils.py:36) | `process_dispatch_queue` (main.py:852-861) | Sends via fallback chain: `interaction.response.send_message` → `interaction.followup.send` → `client.get_channel(channel_id).send` |
| `edit_queue` | `(interaction, edit_kwargs)` | `queue_edit` (utils.py:53) | `process_edit_queue` (main.py:864-875; `_edit` 814-832) | `interaction.edit_original_response`, falling back to a fresh `channel.send` |

Mechanics you must know:

- **`to_thread(func)`** (utils.py:80-89) wraps a sync function so awaiting it runs it in a thread via
  `asyncio.to_thread`. Critical subtlety: if `func` is already a coroutine function it is returned
  **unchanged** (utils.py:82-84). This passthrough is what lets `run_command` (main.py:748-756) execute
  both sync workers and the async slash callbacks stored in `slash_registry` with one code path.
- **`to_threadpool(func, executor)`** (utils.py:92-101) is the bounded variant. `run_command` routes to
  it when the queued kwargs contain `executor` (main.py:751-752) — so **`executor` is a reserved kwarg
  name** for anything sent through `queue_command`; a worker function with its own `executor` parameter
  would have it silently popped. Sole current user: music downloads, capped at a 2-worker
  `ThreadPoolExecutor` (music_player.py:35, used at music_player.py:264 and 277) so a playlist can't
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
   on `HTTPException`/`AttributeError` a raw `channel.send` (main.py:782-794) — means a late result
   degrades to a plain channel message instead of vanishing. Every send written as `queue_*` gets this
   for free; every send written by hand does not.
3. **Serialization + self-throttling.** One drain point per second batches sends and naturally stays
   under Discord rate limits without any explicit rate-limit code. Cost: up to ~1s latency per hop,
   and a reply that traverses two queue hops (see the reminder diagram) takes ~2s. That is the
   accepted trade at hobby scale.
4. **Deferred init without blocking login.** `on_ready` (main.py:106-126) queues `init_reminder`,
   `init_psd`, `init_query`, `refresh_history`, `init_type_emojis(client)` as commands (main.py:117-121)
   instead of running them — pickle loads and the GitHub tree fetch happen in worker threads after the
   bot is already up. `init_type_emojis` is async: it runs on the loop and threads its own downloads.
   main.py also calls `bind_reminder_client(client)` at import time (main.py:211) so the scheduler can
   tell when a reminder's channel is gone.

## Invariants — violate this and X happens

| # | Invariant | Violate it and... |
|---|---|---|
| 1 | **No awaits inside queued sync functions.** Workers are plain `def`; there is no event loop in their thread. | `await` in a `def` is a SyntaxError — so the tempting "fix" is making the worker `async def`. Do that and `to_thread` passes it through unchanged (utils.py:82-84): your "worker" now runs ON the event loop, and its blocking calls (PSD parse, yt-dlp, GitPython) freeze the entire bot — heartbeats, all three queue loops, everything — until it finishes. |
| 2 | **All Discord output from worker code goes through `queue_message`/`queue_file`/`queue_edit`/`queue_any`/`queue_command`** (utils.py:36-60). | Direct calls from a thread produce never-awaited coroutines (message silently lost) or thread-unsafe loop access. You also forfeit the fallback chain: your send dies permanently at the 15-minute token expiry instead of degrading to `channel.send`, and reminders (which run your command with a FakeInteraction) break because FakeInteraction only implements the queue-facing surface. |
| 3 | **The three queue loops must be running for ANYTHING to send.** Started in `on_ready` (main.py:123-125); deliberately STOPPED by `/restart_bot` (main.py:152-154), `/stop_bot` (main.py:164-166), `/update` (main.py:201-203) so no queued output races process replacement. | With a loop stopped (or `on_ready` never reached), `queue_*` calls accumulate silently in the deques forever — the bot looks alive and says nothing. This is also why `commands/update_bot.py` is the one module that bypasses the queues entirely: its functions are `async def` awaiting `interaction.followup.send` directly (update_bot.py:18, 47, 136), because during `/update` the dispatch loop is already stopped and queued progress reports would never flush. |
| 4 | **Registration pattern:** `func = to_thread(func)` rebind FIRST, then the `@tree.command` async wrapper that awaits it (48 occurrences as of 2026-10-20; e.g. main.py:258, 359). `on_ready` stores `slash_registry[cmd.name] = cmd.callback` (main.py:110-111), which powers reminder command execution (analytics.py `_resolve_command`, `_fire`). | Forget the rebind line → the wrapper's `await func(...)` executes the sync function inline ON the event loop (bot frozen for its whole duration), then raises `TypeError` awaiting the plain return value. Since 2026-10-19 reminders bind their stored `/name option:value` text against the callback's real signature at creation AND at every firing, so: rename a command's `name=` → saved reminders post "⚠ Reminder `<id>` skipped its command: `/<name>` is not a bot command"; rename or remove a slash **parameter** → "`<option>` is not one of its options", UNLESS you list it in `RETIRED_OPTIONS` (analytics.py:41; with the command's old option order), which makes stored reminders drop it at load and at fire time — done for /update_stats's `use_local_repo`/`use_local_timestamp` on 2026-10-20; change a parameter's type → values that no longer convert are refused. The reminder itself keeps repeating; its creator must re-create it. The wrapper function's Python name, by contrast, is decorative — see weak point 1. |
| 5 | **One bot per process; state is module-level singletons.** `stats_db` (card_models.py:206, re-exported by psd_analyzer), `card_repo` (query_card.py:750), `state`/`music_queue`/`prev_music`/`executor` (music_player.py:32-45), `reminders` + `_client` + `_lock`/`_check_lock` (analytics.py:61-66), `type_emojis` (type_emojis.py:32, token → emoji map filled by the sync), `slash_registry` + the three deques (utils.py:25-29). | A second client in-process, or importing a commands module under a second name, gives you split-brain state. Import order is load-bearing: the singletons are constructed at import time (cheap — heavy loading is deferred to the queued `init_*` functions), and (since 2026-10-20) none of them prints: the single git-token-presence line comes from `print_token_status()` in main.py. |
| 6 | **Importing `commands.*` requires `config.py` to exist.** utils and music_player `import config` at module level and github_api reads it in a try/except (so `python -m commands.tts_save` runs without it); every card module reaches utils: psd_analyzer → repo_traverser → query_card → type_emojis → utils (since the 2026-10-20 split). | Any tool that unpickles `stats.pkl` must have this repo on `sys.path` AND a config.py present — `CardInfo`/`CardStats` are dataclasses defined in `commands.card_models` with `__module__` pinned to `commands.psd_analyzer` (the facade re-exports them), so unpickling imports the facade and the chain reaches `config`. This is why the diagnostics scripts run from the repo root; code outside this repo must stub the classes, as TTSCardMaker's website build does (`_Unpickler` in site/build/build_site.py). config.py is SECRET — never print or read its values. |
| 7 | **`check_reminder` is re-queued every tick — reminder-path work must stay cheap, and stay single-flight.** The first statement of `process_command_queue` is `queue_command(check_reminder)` (main.py:840); the entry drains on the next tick, so it runs ~once/second in a pool thread (analytics.py:819). Since 2026-10-19 it takes `_check_lock` with a non-blocking acquire (a pass that finds another still running just returns) and does all work under `_lock` (an RLock shared with set/list/remove, which run in their own threads). Commands of due reminders are queued onward (`queue_command(_run_reminder_command, ...)`), never run inline; reminder.pkl is saved only when something changed, atomically. | Anything slow in `check_reminder` costs a pool thread every second and holds `_lock`, stalling `/set_reminder`, `/list_reminders` and `/remove_reminder`. Remove the single-flight lock and two overlapping passes can fire the same reminder twice. Run a reminder's command inline and one slow command blocks every other reminder. |
| 8 | **`lacks_admin` (named `perms_check` until 2026-10-20) returns True when the user is NOT an admin:** `return interaction.user.id not in ADMINS` (utils.py). Callers write `if lacks_admin(i): deny-and-return` (e.g. main.py:161-163). | Flip the polarity and you either lock the admins out or open an admin command to everyone. When adding an admin-gated command, copy an existing gate verbatim. |
| 9 | **Admin gates live in the registered slash callback, and reminders fire as their creator** (2026-10-19). A firing reminder's FakeInteraction carries `_ReminderUser(creator_id)` (analytics.py:69, set in `_reminder_interaction`), so the callback's own `lacks_admin` re-checks ADMINS at fire time. Legacy reminders without a creator keep the default FakeUser — the bot itself (`LEGACY_RUNS_AS_BOT`, analytics.py:37). | Put an admin check anywhere other than the callback stored in `slash_registry` (e.g. only in a worker, or in a pre-dispatch hook) and scheduled runs either skip it or check the wrong user. Make the FakeInteraction default to the bot again and every reminder passes admin checks (the bot's id is in ADMINS on the dev Mac — archaeology Entry 10.8). |

## Data-flow diagrams

### (a) A slash command, end to end

```
Discord user runs /query
      |
      v  (event loop, discord.py)
tree dispatch -> async wrapper slash_query_name (main.py:263-265)
      |   await query_name(...)        # query_name was rebound: to_thread(query_name), main.py:258
      v
asyncio.to_thread -> WORKER THREAD runs sync query_name()
      |   does the work; never awaits; never sends
      |   queue_message(interaction, ...) / queue_file(...)      # appends to dispatch_queue
      v
dispatch_queue (deque, utils.py:25)
      |
      v  (next 1-second tick, back on the event loop)
process_dispatch_queue (main.py:853)
      |- interaction.response.send_message(**kw)     if response not yet done
      |- interaction.followup.send(**kw)             if already responded/deferred
      '- client.get_channel(channel_id).send(**kw)   on HTTPException/AttributeError
      |                                              (token expired, or fake interaction;
      v                                               NOTE: drops `ephemeral` — weak point 3)
Discord
```

### (b) `/update_stats` pipeline (PSDs -> parser -> pickles -> dataframes)

```
/update_stats  (wrapper defers, main.py:370)
      v
WORKER THREAD: update_stats() (repo_traverser.py:469)
      |  GitHub trees API with If-None-Match: 304 -> "Nothing changed in the card repo since the
      |    last update." and the run ends (ETag cache in memory only; repo_traverser.py:162-164)
      |  build the icon vocabulary: Types/ + Types/Stars/ names + the stoogedings manifest
      |    (MDW/Fonts/stoogedings-src/stoogedings.json; impossibility-cards-reference §4.1)
      |  pick PSDs (GitHub only; local mode archived 2026-10-20, weak point 5): new, moved, blob sha
      |    != card.blob_sha (_psd_changed, repo_traverser.py:73-85; commits-since fallback for
      |    cards without blob_sha), named in cards:, or force_update
      |  DOWNLOAD_WORKERS=4 stream blobs (git/blobs API) into mkstemp files ahead of parsing
      |    (repo_traverser.py:127-140, 352-400)
      v
per picked PSD: one commit lookup (_get_remote_timestamp); PSDParser.parse; remove temp file;
  old CardInfo -> old_stats only if its blob changed or it moved; CardValidator
      v
stats_db singleton (card_models.py:206) --save()--> stats.pkl / old_stats.pkl   (the DATABASE)
      |  (save only when num_changed > 0 or traverser.changed)
      v
changed tree only: queue_command(sync_changed_emojis)   (until 2026-10-20 also a TTS save run when images, the
      |   rules, Card Edits.json or the save changed; TTSCardMaker's tts-save.yml Action does that now)
queue_command(card_repo.populate_files); queue_command(card_repo.prep_dataframes)
      |                                            (repo_traverser.py:517-524)
      v  (subsequent command-queue ticks, pool threads)
card_repo dataframes rebuilt  ->  /query and /query_ability now serve the new data
      |
      '--- progress and problem reports stream back throughout via queue_edit / queue_message

Downstream, outside this process: /push commits the pickles; TTSCardMaker's website build reads
them from this repo's main (ttscardmaker-companion §2).
```

### (c) A reminder firing (slash_registry + FakeInteraction), as of 2026-10-19

```
process_command_queue tick (main.py:840)
      v
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
                        v  next tick: run_command (main.py:748) -> coroutine passthrough
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
     |   everyone else reads it through github_api) |  repo_history, repo_traverser, tts_save,
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
 |- commands.query_card    -> utils, global_config, github_api, type_emojis (module level, 2026-10-19), repo_history
 |- commands.type_emojis   -> utils, global_config, github_api, repo_history; query_card.card_repo LAZILY inside _list_type_icons
 |                            [2026-10-19; NOT in setup.py INCLUDE_FILES, so never compiled]
 |- commands.repo_history  -> global_config ONLY  [2026-10-20; commit-pinned image links; NOT compiled]
 |- commands.tts_save      -> repo_history, github_api, card_models, card_edits, global_config; utils + discord LAZILY inside the
 |                            posting functions, so `python -m commands.tts_save <save.json>` runs standalone (TTSCardMaker's
 |                            tts-save.yml Action does, with the AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE env
 |                            overrides global_config reads) [2026-10-20; TTS save updater + /update_tts_save; NOT compiled]
 |- commands.card_models   -> global_config ONLY: CardType, CardStats, CardInfo (the pickled classes, __module__ pinned to
 |                            commands.psd_analyzer), StatsDatabase, stats_db, card_name  [2026-10-20 split; compiled]
 |- commands.card_edits    -> card_models: CardAuthors, CardEdits  [compiled]
 |- commands.psd_parser    -> card_models, global_config, psd_tools: PSDParser, CardClassifier, CardValidator,
 |                            IconVocabulary  [compiled; no Discord, network or pandas]
 |- commands.repo_traverser -> card_models, card_edits, psd_parser, github_api, utils, query_card, repo_history,
 |                            type_emojis (module level!; tts_save no longer, 2026-10-20): RepositoryTraverser, update_stats,
 |                            _tree_etags  [compiled]
 |- commands.psd_analyzer  -> the four above, query_card, github_api, utils, pandas: the FACADE. Re-exports every card name, holds the
 |                            metadata/export commands and init_psd. Code outside imports card classes from HERE only.
 |- commands.frankenstein  -> utils, query_card (card_repo)
 '- commands.update_bot    -> global_config ONLY  (no utils: bypasses queues on purpose, invariant 3)

(not in main) branch archive/local-update-stats: archive/local_update_stats.py  (2026-10-20; NOT compiled)
     -> commands.psd_analyzer, commands.query_card, global_config   [the archived local /update_stats mode]
```

**The deliberate lazy-import cycle breaks (two of them).** `repo_traverser` (and the facade `psd_analyzer`) import
`card_repo` from `query_card` at module level because they use the repo object throughout. `query_card` needs
`psd_analyzer.get_stats_as_dict`, `stats_db`, `card_name` and `CardEdits` only when dataframes are rebuilt or a card
is looked up, so `prep_dataframes`, `_stats_card` and `_populate_edit_names` import them **inside the function**, and
always from the facade, never from `card_models` directly (a stale compiled psd_analyzer .so would otherwise leave two
`stats_db` objects). The same shape repeats since 2026-10-19: `query_card` imports `render_type_tokens` from
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
3. **The dispatch fallback drops `ephemeral`** (main.py:784, in `_fallback_send`: `kwargs.pop("ephemeral", None)`
   before `channel.send`). A reply the user expected to be private — e.g. `/list_reminders`
   `hidden:True` — posts publicly to the channel if the send lands in the fallback path (token
   expired, or any FakeInteraction send). `channel.send` simply has no ephemeral concept.
4. **(Fixed 2026-10-19) Remote traversal used to leak temp files.** Each downloaded temp file is now
   removed in a `finally` right after parsing, and files left by an error at the end (repo_traverser.py:362-363, 392-400). Kept here as a pointer:
   archaeology Entry 10.2.
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
   calls directly on the loop (update_bot.py:47; `to_thread` passes coroutine functions through,
   utils.py:82-84). For `/update` and `/restart_bot` that's fine — the queues are stopped and the
   process is about to be replaced — but `/pull` (main.py:172-179) does NOT stop the queues, so all
   sends and heartbeats stall for the duration of the fetch/merge. Deliberate simplicity; know it's
   why the bot "freezes" briefly during a pull. Since 2026-10-20 each git network call has
   `kill_after_timeout=NET_TIMEOUT` (120 s, update_bot.py:8), so a stalled fetch or push ends; an SSH
   host-key prompt is still not suppressed (network-robustness audit item 4).
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
10. **The 1 s tick is most of a reply's latency** (measured 2026-10-20, Mac: `queue_message` to send
   502 ms mean, p95 966 ms, against 3.9 ms of /query work; a reminder takes two hops, ~1 s), and a search
   sends one message per result (85 per /query_ability on average, max 763) through the single dispatch
   loop, delaying everyone else's replies. Waking the drain on enqueue, or batching up to 10 embeds per
   message, changes the execution model or the output, so both wait for the owner
   (docs/BUGS_AND_IMPROVEMENTS.md §2.5; `docs/Finished/optimization-pass-2026-10-20/profile.md`).
11. **The card pickle saves are not atomic.** `StatsDatabase.save` (card_models.py:138-143) and
   `save_aliases` (query_card.py:359-362) write in place, unlike reminder.pkl's tmp + fsync + replace, and
   nothing locks them against a concurrent `/push` or `/restart_bot`; a kill mid-write leaves a truncated
   pickle (network-robustness audit item 2; open).

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
- **Bounded parallelism**: pass `executor=<ThreadPoolExecutor>` through `queue_command` (main.py:751-752)
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

Line numbers were verified 2026-07-11 at HEAD `284d13c`, re-verified 2026-10-20 at HEAD `9fb3093` and re-mapped after that
day's optimization pass (psd_analyzer split, `refs6.py`)
(expected values below are the 2026-10-20 readings). They WILL drift.
Re-find before citing:

| Fact | Re-verify with (cwd = repo root) |
|---|---|
| Three 1s loops + drain functions | `grep -n -e "tasks.loop" -e "async def process_" main.py` (838-839 / 852-853 / 864-865) |
| check_reminder re-queued every tick | `grep -n "queue_command(check_reminder)" main.py` (840) |
| Loops started in on_ready / stopped in restart/stop/update | `grep -n "process_.*_queue.st" main.py` (123-125 / 152-154 / 164-166 / 201-203) |
| Five queued inits incl. refresh_history and type emojis; reminder client binding | `grep -n "queue_command(init\|queue_command(refresh_history\|bind_reminder_client(client)" main.py` (117-121, 211) |
| The three deques | `grep -n "= deque()" commands/utils.py` (25-27) |
| queue_* helpers | `grep -n "^def queue_" commands/utils.py` (36/41/47/53/58/63) |
| to_thread coroutine passthrough | `grep -n -e "def to_thread" -e "iscoroutinefunction" commands/utils.py` (80, passthrough 82-84) |
| run_command + reserved `executor` kwarg | `grep -n "async def run_command" main.py` (748); `grep -rn "executor=executor" commands/music_player.py` |
| Registration rebind count | `grep -c "= to_thread(" main.py` (48) |
| slash_registry population | `grep -n "slash_registry\[cmd.name\]" main.py` (111) |
| Reminder resolve/bind/queue | `grep -n "slash_registry.get\|queue_command(_run_reminder_command" commands/analytics.py` |
| Reminders fire as creator; single-flight | `grep -n "_ReminderUser(\|_check_lock\|^LEGACY_RUNS_AS_BOT" commands/analytics.py` |
| FakeInteraction + default bot user id | `grep -n "def make_fake_interaction\|1011264680683110433" commands/utils.py` (124, 138) |
| lacks_admin polarity | `grep -n -A2 "def lacks_admin" commands/utils.py` (returns `not in ADMINS`) |
| Unique handler names (was duplicate defs) | `grep -n -e "async def slash_set_re" -e "async def slash_delete_" main.py` (4 distinct names) |
| Deque shuffle | `grep -n "shuffle(music_queue)" commands/music_player.py` (338) |
| Ephemeral dropped in fallback; get_channel None guarded | `grep -n 'pop("ephemeral"\|def _fallback_send\|not found")' main.py` |
| Temp files removed after parse (weak point 4 fixed) | `grep -n "mkstemp(\|remove(local_file)" commands/repo_traverser.py` |
| Local mode archived 2026-10-20 (weak point 5) | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS`, analytics.py:41-44) |
| Module-level cycles + lazy breaks | `grep -n "from commands.query_card import card_repo" commands/repo_traverser.py commands/psd_analyzer.py commands/type_emojis.py` (module level in the first two; inside a function in type_emojis); `grep -n "from commands.psd_analyzer import" commands/query_card.py` (all inside functions) |
| Singletons | `grep -n "stats_db = StatsDatabase()" commands/card_models.py` (203); `grep -n "card_repo = CardRepository()" commands/query_card.py` (746); `grep -n "state = MusicState()" commands/music_player.py`; `grep -n "^reminders = \|^type_emojis:" commands/analytics.py commands/type_emojis.py` |
| config import chain | `grep -ln "^import config" commands/*.py` (music_player, utils); `grep -n "import config" commands/github_api.py` (try/except) |
| type_emojis uncompiled | `grep -c "type_emojis" setup.py` (0) |
| update_bot bypasses queues | `grep -n "followup.send" commands/update_bot.py`; `grep -n "from commands" commands/update_bot.py` (empty) |

Maintenance rule: if any grep above stops matching, the architecture moved — update this contract in
the same change, and check whether the corresponding invariant row still holds before assuming it does.

2026-10-20 (later): profile.md, cited in weak point 10, moved from `~/Documents/ac-optimization/` to `docs/Finished/optimization-pass-2026-10-20/`.
