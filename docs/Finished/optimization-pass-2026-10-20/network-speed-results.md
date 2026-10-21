# Network and speed pass: results (2026-10-20)

Code: the working tree on `main` (7f6bd05 plus the uncommitted robustness/docs work), Mac M2 Ultra, Python 3.14.8,
psd-tools 1.25.0. "before" = a copy of the tree taken before this pass (`/tmp/acns/before`), "after" = the tree with
this pass (`/tmp/acns/after`), `_c` = the same trees built with `setup.py build_ext` (Cython 3.3.0, the 7 INCLUDE_FILES).
Every sandbox has copies of config.py and the four pickles; the live pickles were never written.
Scripts: `/Users/michaelsrouji/Documents/ac-optimization/scripts/` (each takes the sandbox root as its first argument).
The machine was busy with other jobs during some runs, so absolute times vary between runs; the A/B timings that matter
were taken interleaved in one process, or back to back.

## What changed (no output change; the only timing change is the emoji listing at startup, see below)

| # | Change | Where |
|---|---|---|
| 1 | Fuzzy ability step: skip a card when `2*min(la,lb)/(la+lb) <= match_ratio` (that is `real_quick_ratio()`, an upper bound of `ratio()`, so the same cards pass). Lowered abilities are cached once for each `cards_dff` that prep_dataframes builds | query_card.py `_fuzzy_labels` |
| 2 | `_find_closest_match`: return the normalized key straight away when it is in `git_files` (an exact key is the only possible 1.0 score; not used for keys of 200+ characters, where difflib's autojunk starts) | query_card.py |
| 3 | Commit lookups: one `get_page(0)` with `per_page=100` gives the newest and the first commit (2 requests become 1 per changed card, 1+N become 1 per new card). For files with 100 or more commits it falls back to `.reversed[0]` | psd_analyzer.py `_commit_page`, `_get_remote_timestamp` |
| 4 | The lookups run in a 2-thread pool next to the 4 download threads, so the serial parse loop no longer waits ~0.9 s per PSD on GitHub. Results are read at the same point as before (exceptions surface in the same place, author choice uses the same db state) | psd_analyzer.py `_process_files_from_response`, `_repo` (one PyGithub client per thread) |
| 5 | The tree JSON is parsed once per /update_stats (`self.tree`) instead of 7 times | psd_analyzer.py `_tree_items` |
| 6 | /update_stats hands its tree's ETag to `card_repo.populate_files`, so the card list's next fetch is a 304. A tree passed without its ETag clears the ETag, so the code never sends an ETag that belongs to a different tree | psd_analyzer.py, query_card.py `populate_files(tree, repository, etag)` |
| 7 | Type-emoji tree listing: it shares `card_repo._tree_lock` with the card list, sends the card list's ETag and reuses its tree on a 304. When it runs first at startup, it seeds the card list's tree, so the card list's own fetch gets the 304. Fallbacks and messages are unchanged | type_emojis.py `_list_type_icons`, query_card.py `_tree_lock` |
| 8 | Lazy imports: cv2 (utils, frankenstein), yt_dlp plus the module-level `dl_pre` YoutubeDL (music_player `_preprocessor()`/`_ydl()`), PyGithub (psd_analyzer) | utils.py, frankenstein.py, music_player.py, psd_analyzer.py |
| 9 | TTS save: the git trees and blobs it reads are cached by sha (a sha's content never changes; only the last run's objects are kept, about 6 MB). The website's cards.json is fetched with its ETag. After a 403 on `POST git/blobs`, the bot remembers that the token can't write and goes straight to SSH until a restart (a rate-limited 403, with `Retry-After`, `X-RateLimit-Remaining: 0` or "rate limit" in its message, only for that run) | tts_save.py `_object`, `_blob`, `_site`, `_cant_write` |

New tests: `tests/test_network_speed.py` (10 tests: fuzzy step against brute force at 4 ratios, the exact-key fast path
against difflib including a 230-character key, commit pages including the more-than-one-page fallback, the tree ETag
handoff, emoji listing seeding and 304s, the TTS sha cache, the 403 memory). The whole suite has 211 tests and all pass
on .py. Compiled, the 10 new tests pass; the one failure, `test_reminders_time.py::test_interval_invalid[None]`, fails
the same way on the compiled "before" build, so it predates this pass (analytics.py is compiled and untouched here).

## Numbers

### Searches (`search_ab.py`: interleaved in one process, best of 5 per query, results asserted equal)

| Path | before | after | Gain |
|---|---|---|---|
| Fuzzy step, 60 short plain phrases (1-3 words, the usual /query_ability) | 1138 ms total, median 16.8 ms | 63 ms total, median 0.68 ms | x18 (.py), x34 compiled |
| Fuzzy step, all 95 plain queries (adds 20 long phrases, 10 whole-ability prefixes, `a`*300) | 6616 ms, median 29.3 | 2674 ms, median 1.29 | x2.5 (long queries keep most cards in the length window) |
| `_find_closest_match`, 234 names (exact, typos, first words, aliases) | 935 ms, median 3.93 | 332 ms, median 0.000 | x2.8 |
| `_find_closest_match`, 100 exact names | 424 ms | 0.0 ms | removes the lookup entirely |

`search_ns.py` (the whole engine, before vs after, back to back at a quiet moment): ability plain queries mean
36.4 → 14.6 ms (median 21.1 → 1.7); quoted/NOT/stat/combos 5.4 → 1.2 ms; rulebook 0.31 → 0.29 ms (unchanged, nothing to
do); `get_card_path` 4.0 → 1.3 ms mean.
**Identity:** every result (index order, index dtype, column dtypes, every cell through `to_csv`) is identical for
95 + 40 ability queries, 100 rulebook queries and 234 names, in all four builds (before, after, before_c, after_c).

Not done, because measuring showed nothing to gain: precomputed lowercase columns for `type==`/`type~` (stat queries take
0.2-0.4 ms, mostly the boolean mask), cached per-card SequenceMatchers (slower than fresh ones with the length bound: 2161
vs 1675 ms on the 95 queries, and they are not thread-safe), and a character-count bound (prunes only about 25% of the
long queries' candidates).

### /update_stats

Live GitHub reads, `e2e_update_ns.py`: `update_stats(cards=<6 cards>)` on a throwaway copy, with 2 of the cards deleted
first so they take the new-card path (author from the first commit).

| | before | after |
|---|---|---|
| Wall time, 6 PSDs (2 runs) | 16.3 s, 18.1 s | 6.5 s, 8.7 s |
| Commit requests | 25 | 6 |
| Other requests | tree 1, blobs 9, get_repo 1 | the same (get_repo is now lazy: 0) |
| Saved card entries (to_dict, timestamp and author included) | - | identical to before |

`commits_ns.py` (`_get_remote_timestamp` alone, 5 paths with 0, 1, 2, 5 and 9 commits): existing card 2 → 1 request
(~1.0 s → ~0.5 s); new card 2-10 → 1 request (Mini Doomer 6 → 1, 3.0 s → 0.49 s; The Freezer 10 → 1, 5.2 s → 0.49 s).
Same (timestamp, author) for all 9 lookups, and the same again with `COMMITS_PAGE=2`, which exercises the
more-than-one-page path (3 requests).
Forced sweep, by the audit's arithmetic: 2,714 → 1,810 GitHub requests, and the ~0.9 s per PSD of serial lookup time
now overlaps the downloads (the audit put the lookups at ~12 min of a forced 813-card run).
Thread safety (review fix): PyGithub's connection keeps the request on the instance (`request()` stores the URL,
`getresponse()` sends it), so one client shared by the lookup threads and the parse thread could hand a PSD another
PSD's commits. Each thread now has its own client (`threading.local`, `get_repo(lazy=True)`: no request).
`scripts/race_ns.py` (local echo server, real `_commit_page` in the 2-thread pool plus 100 lookups on the main
thread, a GIL-busy thread, switch interval 1e-5 s): shared client 6, 6, 5 of 500 wrong; per-thread 0 of 500 in 5 runs.
Tree JSON parses per changed /update_stats: 7 → 1 (`etag_handoff_ns.py`), at 2.1-3.9 ms each (597,648-byte tree).

### Tree requests (startup, /sync_emojis, after an update)

`startup_ns.py` (live): init_query and the emoji listing run in parallel threads, the way on_ready queues them.

| | before | after |
|---|---|---|
| Startup tree GETs | 200 + 200 (2 x 597,648 B) | 200 + 304 (4 of 4 runs, in either order) |
| `/sync_emojis` tree GET | 200 (597,648 B, 450 ms) | 304 (0 B, 350-380 ms) |
| Card list's next fetch after a changed /update_stats (offline fake, `etag_handoff_ns.py`) | 200 (its ETag belonged to an older tree) | 304 |

Timing change: at startup the emoji listing now waits for the card list's fetch and then sends its 304, so emojis are
ready about 0.35 s later (~0.52 s → ~0.85 s after start). Overall startup time is unchanged (~0.9-1.0 s here).

### TTS save (`tts_save.py` reads, `tts_ns.py`, live and read-only)

| Back-to-back no-change runs | Requests | Downloaded |
|---|---|---|
| before, each run | 9 (ref, commit, 3 trees, 3 blobs, site) | 6.40 MB |
| after, first run | 9 | 6.40 MB |
| after, next run | 3 (ref, commit, site 304) | 0.00 MB |

When the token can't write, every changed run after the first no longer uploads 7.8 MB of base64 before switching to
SSH (`test_tts_token_that_cant_write_is_remembered`). The script reproduces `_Names`' ETag logic rather than building a
`_Names`.

### Import (`import_ns.py`, 3 interleaved rounds x 5 fresh processes)

| Build | `import main` median | Peak RSS | Heavy modules loaded |
|---|---|---|---|
| before (.py) | 1059-1278 ms | 185 MB | cv2, yt_dlp, github, git, numpy, pandas, psd_tools |
| after (.py) | 646-720 ms | 122.5 MB | git, numpy, pandas, psd_tools |
| before_c | 1129-1210 ms | 186 MB | as before |
| after_c | 693-755 ms | 123.5 MB | as after |

About -450 ms and -62 MB per restart on this (busy) Mac; the audit predicted -230 ms / -50 MB on a quiet one. Smoke
checks: /frankenstein's merge and PNG bytes are identical (slice and average modes, .py and compiled), and the lazily
made YoutubeDL has the same 14 params as the old `dl_pre`.

### Parser gate (validation-and-qa §4.1 method A)

`parse_all_ns.py` + `compare.py` over all 813 PSDs: before vs after 0 differences, and before vs after_c (compiled)
0 differences; 0 errors in each run.

## Not done (left for the owner or a later pass)

- Queue tick latency (profile #1) and one message per search result (#2): these change the execution model or the
  output, so they need the owner's sign-off.
- The psd-tools fork (#4): a separate effort (docs/NEXT_SESSION_PLAN.md).
- Lazy GitPython in update_bot (~30-65 ms): `tests/test_robustness.py::test_push_reports_the_pull_error_not_the_abort`
  patches `update_bot.Git` and `update_bot.GitCommandError` as module attributes, so I reverted it rather than edit
  another phase's test.
- A `requests.Session` per thread for PSD blob downloads (audit: 85 → 17 ms per request, about 61 s over a forced
  sweep): `tests/test_update_stats.py` fakes GitHub by patching `psd_analyzer.get`, so this needs a test change first.
- Skipping populate_files and prep_dataframes after a 304 /update_stats: not done, because that path also refreshes
  the website ids (site cards.json can change without a tree change) and rebuilds the dataframes from a pickle that
  /pull may have replaced.
