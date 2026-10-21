# AutoCroissant profile (2026-10-20, read-only)

Code: `main` at `7f6bd05`, run from a sandbox copy (`git archive HEAD` plus copies of config.py and the four pickles, in
`/tmp/acprof/sb`). The live repo and its pickles were not touched, and `git status` is still clean.
Machine: Mac, Apple M2 Ultra, Python 3.14.8, psd-tools 1.25.0, no Cython build (pure .py). The jetson is slower. Re-run
the same scripts there off-hours before quoting absolute numbers for production.
Scripts: `/Users/michaelsrouji/Documents/ac-optimization/bench/` (they expect the sandbox at `/tmp/acprof/sb`).
Network numbers come from 9 unauthenticated read-only GitHub requests and 1 website request.

## Ranked opportunities (by expected user-visible gain)

| # | Opportunity | Measured now | Expected after | Behaviour change? |
|---|---|---|---|---|
| 1 | **Queue tick latency.** Every reply waits for the next 1 s drain of `dispatch_queue`/`edit_queue`. | queue_message → send call: mean **502 ms**, p95 966, max 982 (60 msgs, `hops.py`). /query's own work is 3.9 ms, so the tick is ~99% of the bot-side latency. Reminder commands go through 2 hops (~1 s mean). | Wake the drain on enqueue (`loop.call_soon_threadsafe(event.set)` in `queue_*`, loop waits on the event with a 1 s timeout), or a 0.1-0.2 s interval with `check_reminder` kept at 1 Hz: ~0-100 ms per hop. Idle CPU stays well under 2 ms/s. | Timing only, but it touches the execution model (architecture contract "WHY" 3): needs owner sign-off. discord.py still handles 429s and there is still one consumer. |
| 2 | **One Discord message per search result** (/query_ability, /query_rulebook), all sent one after another by the single dispatch loop, which also blocks every other user's replies behind them. | 100 sampled ability queries queued **85.1 messages each** (median 15 results, p90 275, max 763). Rulebook: 15.4 each. Send time per message not measured (needs Discord). With Discord's per-channel limits, a big result set takes tens of seconds to minutes. | Up to 10 image embeds per message: about 10x fewer sends. Or a default `limit` plus the website link that is already sent. | Yes (how results look): owner decision. |
| 3 | **/update_stats serial commit lookup per changed PSD.** `_get_remote_timestamp` runs on the parse thread, after the 4-way download pool. | `totalCount` 562 ms + `commits[0]` 303 ms = **865 ms, 2 requests** (3 for a new card), against a parse of **95 ms** and a 20.8 MB download of 1.50 s (13.9 MB/s per stream) that overlaps with them. A forced 813-card run spends ~12 min in these lookups alone (Mac numbers). | Do the lookup in the download worker: the serial path becomes the parse only. Or use 1 request (page 1 with `per_page=1` + the `Link` header for the last page, same data). Or read the history clone (`repo_history._paths` already has the last commit per path): 0 requests. Saves ~0.8 s per changed PSD. | None for the first two. Using the history clone needs a check that it gives the same author and timestamp. |
| 4 | **psd-tools `PSDImage.open` is 99% of a parse.** | 50 PSDs: open 4.84 s, the parser's own layer work **0.04 s**, raw file read 0.12 s. Inside open (`blocks.py`): document `TEXT_ENGINE_DATA` (Txt2) **25%**, `TYPE_TOOL_OBJECT_SETTING` (TySh, needed) 21.5%, layer effects (lfx2 + v1) **16.7%**, channel image data 7.7%, smart objects 2.3%. | The psd-tools fork could parse Txt2 and effects blocks lazily: ~40% off open, ~95 → ~55 ms per PSD. Cythonising psd_analyzer gains nothing (its code is 0.8%). | None if lazy; gate = parse_all/compare, 0 differences on 813. |
| 5 | **Fuzzy step of `ability_search_engine`** (`SequenceMatcher` against all 765 abilities on every plain-text query; 96% of the engine's time, 54% of that is `__chain_b` rebuilding each ability's index). | Plain-text query **17.5 ms** mean (engine on all 100 queries: 12.7 ms mean, p95 25 ms). | Skip a card when the length bound `2*min(la,lb)/(la+lb) <= match_ratio` (an upper bound of `ratio()`, so the results are the same). Fuzzy step 16.5 → **0.97 ms**. Adding cached per-card matchers (`set_seq1`) gives 0.86 ms. Engine on 60 plain queries: 1023 → **78 ms**. Results are identical on all 60 (asserted). Fuzzy hits are rare anyway (0.45 per query). | None. |
| 6 | **Duplicate tree fetch at startup.** `init_query` (`populate_files`) and `init_type_emojis` (`_list_type_icons`, unconditional) each GET the recursive tree. | One tree GET: 210 ms warm, **6.2 s cold** (first request of the session); a 304: 120 ms. That is 1 extra request per start, and it doubles the B7 5xx exposure. | Reuse `card_repo._tree` (or pass the shas): -1 request, -0.2 to 6 s from when emojis are ready. | None. |
| 7 | **`_find_closest_match`**: `get_close_matches` over 1077 registry names. | 3.9 ms mean (exact names 4.4 ms, typos up to 8 ms). | Return early when the normalized key is already in `git_files` (an exact match is the only way to score 1.0, so the result is the same): 100 queries 386 → 157 ms, exact names 219 → 0 ms. | None. |
| 8 | **Import time** (`import main` 0.85 s, 1.01 s wall). | Total self-time by package: pandas 220 ms, psd_tools 89, numpy 60, cv2 49, aiohttp 36, yt_dlp 35, git 30, discord 30, github 25. Avoidable at import: **cv2** (used only by /frankenstein through `utils.url_to_cv2image`/`cv2discordfile`) +52 ms, +25 MB RSS. **Module-level `dl_pre = YoutubeDL(...)`** (music_player.py:69) 56-59 ms, +6.5 MB. **github** 51 ms, 13 MB (only /update_stats). **git** 65 ms (only update_bot). | Lazy imports / creating these on first use: ~-230 ms per restart, ~-50 MB of the 183 MB baseline. pandas is needed by `init_psd` right after login, so deferring it gains nothing. | None (restart time and RAM only). |
| 9 | **`resp.json()` re-parsed 7x** on the 0.6 MB tree per changed /update_stats (psd_analyzer.py:1470/1493/1515/1527/1552/1566/1637). | 2.1 ms each, 15 ms per run. | Use `self.tree`. Code health, not speed. | None. |

No action needed: the idle loops, the reminder pass, `prep_dataframes`, `rulebook_search_engine`, stat queries and memory
(details below).

## 1. Import/startup (`python -X importtime`, best of 3, warm .pyc)

`import main` = 851 ms (`importtime_main.txt`). Interpreter alone: 20 ms. Cumulative by commands module inside main
(the first importer pays for shared deps): query_card 299 ms (pandas 277), psd_analyzer 145 (psd_tools 94,
github 47), analytics 122 (through utils: cv2 73 with numpy 23, requests 48), discord 117 (aiohttp 74), music_player 98
(yt_dlp 44, plus ~40 ms self from building `dl_pre` at import), update_bot 34 (GitPython).
Imported alone in a fresh process: utils 303, help 264, management 265, analytics 284, music_player 334,
repo_history 15, type_emojis 268, query_card 450, tts_save 76, psd_analyzer 573, frankenstein 621, update_bot 180 ms.
Marginal RSS after discord+pandas+psd_tools+requests: cv2 +25.1 MB, github +13.3, yt_dlp +9.9, git +3.4.

Startup after login (from on_ready, 5 queued inits that run in parallel threads): `init_psd` = `stats_db.load` 3 ms +
`prep_dataframes` 9 ms. `init_query` = the tree GET (0.2-6.2 s measured), plus the Card Edits blob when its sha changed,
plus the website's cards.json (222 ms, 0.43 MB). `init_type_emojis` does a second tree GET (item 6).
`refresh_history` (git fetch on the jetson) was not measured.
Side note (robustness, not speed, unverified live): `on_ready` re-syncs every guild's commands and calls `.start()` on
the three loops. If Discord ever fires on_ready again, `Loop.start` raises "already launched".

## 2. /update_stats path

- Parser on 50 PSDs, evenly sampled from the 813 sorted paths (`parse50.py`, parse_one style, warm): `parse()` mean
  **95 ms**, median 105, max 273. Files average 20.2 MB (max 84.5) with 52 layers. Validation takes 6 us. Slowest:
  Z!!!!! 202 ms, The_Watcher 175, Zigg 173, 07_Stats3 168.
- Split (`split50.py`): open 4.84 s / parser logic 0.04 s / file read 0.12 s. `_core_box` (star numpy) ran 24 times
  for 4 ms total. `_pixel_coverage` never ran in the sample.
- One creature card, cProfile (`prof1.py`, Z!!!!!, warm 133 ms): 99% under `PSDImage.open` → `layer_and_mask` →
  `tagged_blocks.read` (88%), engine_data parsing 41%, descriptors 33%. The rest of the hot spots are
  `bin_utils.read_fmt`/`read_exact` and parse_limits bookkeeping (~800k `check` calls for 5 opens). The descendants
  walk is 0.2 ms. Text extraction, regexes and validation are all under 1%.
- Network per changed PSD: 2 serial API calls (865 ms), plus a streamed blob download (1.5 s for 20.8 MB) that
  overlaps with them. The forced-sweep floor is the download: **16.4 GB** of PSDs in total.
- A no-change run: one conditional tree GET (304, 120 ms), then the queued `populate_files` does its own 304 and the
  site check, and `prep_dataframes` (9 ms).

## 3. Search engines (real stats.pkl via StatsDatabase, sandbox copy; `search_bench.py`)

765 cards and 48 rulebook entries in the dataframes. 1077 registry names (local clone's PNG list plus aliases plus
Card Edits names, offline).
100 ability queries (60 plain phrases, 10 quoted, 5 NOT, 10 stat, 15 `|`/`&`), seed 7:

| Path | Mean | Median | p95 | Max |
|---|---|---|---|---|
| `ability_search_engine` | 12.66 ms | 13.84 | 25.10 | 30.81 |
| : plain text / quoted / NOT / stat / combos | 17.49 / 3.26 / 0.36 / 0.18 / 12.32 ms | | | |
| `rulebook_search_engine` | 0.30 ms | 0.33 | 0.54 | 0.59 |
| `_find_closest_match` | 3.89 ms | 3.81 | 6.59 | 8.33 |
| Full workers (`worker.py`): query_ability / query_name / query_rulebook | 14.1 / 3.9 / 1.4 ms | | | |

- **pandas copies and rebuilds:** `prep_dataframes` runs only in `init_psd` and after /update_stats, never per query.
  Per query there are only boolean-mask slices, `concat` and `index.duplicated`, costing ~0.2-0.4 ms (the NOT and stat
  paths show the floor). There is nothing to cache on the pandas side. All 17 columns are `object` dtype, which is
  fine at this size.
- **O(n) per-call work that can be cached:** (a) each card's `SequenceMatcher` b-index, rebuilt per card per query, and
  a redundant `x.lower()` on already-lowered text (item 5); (b) `git_filenames` rebuilds the list on every call, but it
  only costs 4 us; the cost is difflib itself (item 7); (c) `_parse_stat_query` recompiles 2 regexes per call, which
  the `re` cache makes free.

## 4. Memory (RSS from `ps`; `mem.py`, `mem2.py`)

| Stage | RSS |
|---|---|
| Bare interpreter | 16.5 MB |
| After `import main` | **183.4 MB** |
| After reminders + stats.pkl + old_stats.pkl + dataframes | 188.2 MB (Python heap: pickles 1.5 MB, 819 cards + 291 old entries; dataframes 0.4 MB) |
| After the registry + 200 queries | 192.6 MB |
| After parsing the largest PSD (84 MB) | 281.7 MB |
| During 50-300 parses | 310-425 MB, fluctuating |

No leak: the live Python heap is 0.5 MB after 300 parses (300 CardInfo kept), and the heap peak is 91 MB, on the
largest file. The RSS high-water is the allocator keeping freed PSD buffers (macOS; glibc on the jetson may behave
differently, so check there). The pickles and dataframes are negligible. Imports are 90% of the steady RSS.

## 5. Idle per-tick work (`idle.py`: main's three real loops on asyncio, no Discord connection, 30 s)

- Per second: `process_command_queue` queues `check_reminder` and drains it **in the same tick** →
  `create_task(run_command)` → `asyncio.to_thread`. The dispatch and edit loops find empty deques and return.
- Cost: **0.33 ms CPU per second** (0.033% of a core) with 0 reminders; 0.40 ms/s with 200 not-yet-due reminders.
  `check_reminder` takes 12 us (0 reminders) and 49 us (200). No pickle I/O when idle: it loads once and saves only on
  change. The local reminder.pkl holds no reminders, so a jetson-sized schedule was simulated.
- Threads: 2 alive. `to_thread` reuses the default executor's threads; it does not start a fresh thread each second
  (the architecture skill's "fresh thread" wording is inaccurate, a doc fix only).
- Conclusion: idle cost is not worth optimizing. The tick's cost is latency (item 1), not CPU.

## Gates for whoever implements

- Items 4 (fork) and any psd_analyzer change: `/tmp/psdreg/parse_all.py` + `compare.py`, 0 differences on 813
  (validation-and-qa §4.1 method A). The last full run took 37 s with 16 processes.
- Items 5 and 7: assert identical result indexes on the `search_bench.py` query set (as `search_variants.py` does)
  and re-measure.
- Item 3: count requests per changed PSD before and after, and compare the stored `timestamp`/`author` on a
  sandboxed `cards:` run.
- Items 1 and 2 change behaviour or the execution model: owner sign-off first. Measure live send times by timestamping
  the drain loop.
