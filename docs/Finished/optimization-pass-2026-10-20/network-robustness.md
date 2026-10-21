# AutoCroissant: network and robustness audit (read-only, 2026-10-20, main @ 7f6bd05)

I changed no code. Every figure under "measured" came from a command run today: live GitHub/Pages requests from the Mac (a few unauthenticated plus 11 PyGithub calls), a local aiohttp server, a timing run on a /tmp copy of the pickles (copy deleted afterwards), and git on the history clone. Line numbers refer to the current tree.

## 0. Measured facts used below

| Fact | Value |
|---|---|
| Card-repo tree (`git/trees/main?recursive=1`) | 2,379 entries, not truncated; 112,213 B gzip on the wire, 597,873 B JSON; `json.loads` 2.1 ms (Mac) |
| Tree with `If-None-Match` | 304 (the tree ETag works) |
| Website `data/cards.json` | 99,617 B gzip / 427,681 B; weak ETag, `If-None-Match` -> 304; `max-age=600` |
| PSDs in tree | 904 files, 18.6 GB total, largest 84.5 MB (average 20.6 MB) |
| Card PNGs | 1.97 MB average (what `/frankenstein` downloads per card) |
| TTS save | 5,881,494 B (7,841,992 B as base64 in `POST git/blobs`) |
| Icons | Types/ 35 PNG, Types/Stars/ 56 PNG -> 126 emojis (35 x 2 + 56) |
| Commits per PSD (sample of 30, history clone) | 1-8 (4x1, 5x2, 9x3, 4x4, 2x5, 2x6, 2x7, 2x8) |
| PyGithub `_get_remote_timestamp` (live, 8-commit PSD, `per_page=1`) | `get_repo` 1; existing card 2 requests; new card / no author **1 + N commits = 9** (it walks every page) |
| Cards without `blob_sha` (pickle copy) | 0 of 819, so `_committed_since_update` (psd_analyzer.py:1395) can't run today |
| `stats_db.load()` / `prep_dataframes()` | 3 ms / 8 ms (Mac, 350 KB + 107 KB pickles) |
| New TLS connection vs reused session (raw.githubusercontent, median of 6) | 85 ms vs 17 ms per GET |
| `/query_ability` result counts | damage 133, attack 287, type==creature 297, hp>5 200, draw 21 |
| A discord.py `File` re-sent after a failed send (aiohttp test, same flow as `reset(seek=0)`) | 1st upload 1,000 B, **2nd upload 0 B** |
| `/queue` chunks over 2,000 chars (simulated with main's code, 100 songs) | paths 46-86 chars: 22% of chunks, 64% of calls; paths 66-126: 40% / 94% |
| `git diff --stat` of 2026-10-20's 30 commits | 2,950 chars (46 files) |
| `tasks.loop` retry set (discord.py 2.7.1 source) | `OSError, GatewayNotFound, ConnectionClosed, aiohttp.ClientError, asyncio.TimeoutError`; anything else stops the loop for good. The popped item is lost either way |
| Library defaults | requests: no timeout unless given; PyGithub timeout 15 s, `GithubRetry(total=10)` (it may sleep until the rate-limit reset); yt-dlp socket timeout 20 s; GitPython `kill_after_timeout=None` |

## 1. Every outbound call

T = timeout, R = retry, C = cache/conditional. "Loop" = runs on the asyncio event loop (blocks every sender and the heartbeat); "thread" = `to_thread`/pool.

| # | file:line | Call / URL | T | R | C | Runs in |
|---|---|---|---|---|---|---|
| 1 | query_card.py:76 | GET api.github.com `/git/trees/main?recursive=1` (populate_files) | 30 s | no | ETag (after the first fetch) | thread; **loop** when called from `set_repository` (query_card.py:707) |
| 2 | query_card.py:344 | GET `/git/blobs/{sha}` Card Edits.json | 30 s | no | by blob sha | same as 1 |
| 3 | query_card.py:363 | GET `{SITE_URL}/data/cards.json` | 30 s | no | ETag | same as 1 |
| 4 | psd_analyzer.py:1464 | GET tree (update_stats) | 30 s | no | ETag, in memory only (`_tree_etags`, :38) | thread |
| 5 | psd_analyzer.py:1414 (`_blob`) | GET `/git/blobs/{sha}` raw: PSDs (stream, 4-worker pool :1648), manifest/authors/edits (:1426) | 120 s per read | no | small files by sha (`_file_cache`) | pool / thread |
| 6 | psd_analyzer.py:1408-1409 | PyGithub `get_repo` | 15 s | GithubRetry 10 | per run | thread |
| 7 | psd_analyzer.py:1710-1717 | PyGithub `get_commits(path)`: totalCount, [0], [totalCount-1] | 15 s | GithubRetry 10 | no | thread |
| 8 | psd_analyzer.py:1401-1403 | PyGithub `get_commits(since)` + `commit.files` (legacy cards only) | 15 s | GithubRetry 10 | per run | thread (can't run today) |
| 9 | type_emojis.py:101 | GET tree (`_list_type_icons`) | 30 s | no | **none** (second tree download at startup and on every /sync_emojis) | thread |
| 10 | type_emojis.py:111 | GET `/git/blobs/{sha}` icon | 30 s | no | per run by sha | thread |
| 11 | type_emojis.py:236/240/255/297/357 | Discord: delete / create / fetch application emojis | aiohttp 300 s | discord.py 429 handling | n/a | loop (cheap awaits) |
| 12 | repo_history.py:33 | `git fetch` (history clone) | 300 s | no | incremental | thread |
| 13 | repo_history.py:125 | `git clone --bare --filter=blob:none` | 300 s | next call | clone `.tmp` then rename | thread |
| 14 | tts_save.py:904/905/766/911/912 | API GETs: ref, commit, 3 trees, 3 blobs (5.9 MB save) | 120 s | 3 tries only for lag/conflict | **none** | thread |
| 15 | tts_save.py:926/942/944/947/949 | API POST blob (7.8 MB), tree, commit, PATCH ref, GET ref | 120 s | same | none | thread |
| 16 | tts_save.py:303 | GET site cards.json | 30 s | no | **none** | thread |
| 17 | tts_save.py:611 | GET raw `TTS/decks.json` (standalone only) | 30 s | no | no | CLI |
| 18 | tts_save.py:821 | `git push` over SSH, `BatchMode=yes` | 300 s | 3 tries | n/a | thread |
| 19 | update_bot.py:28, 35, 49, 65, 93, 105, 119 | GitPython pull --rebase, push, fetch, pull, pull, pull --rebase, fetch | **none** | no | n/a | **loop** |
| 20 | utils.py:96 | GET raw card PNG (`/frankenstein`, frankenstein.py:37) | 30 s | no | **no status check, no cache** | thread |
| 21 | music_player.py:232, 245, 260 | yt-dlp extract/download | 20 s socket | yt-dlp 10 | files on disk | thread / 2-worker pool |
| 22 | music_player.py:181 | voice `connect()` | 30 s | reconnect | n/a | loop |
| 23 | main.py:97, management.py:87/93 | `tree.sync` per guild | aiohttp | 429 | n/a | loop |

No `subprocess` call is left without a timeout except GitPython (#19). Every `requests` call has a timeout. None of them retry.

### Requests per command (static count, checked against the measurements above)

| Command | GitHub API | Other HTTP / git | Discord | Bytes |
|---|---|---|---|---|
| startup (`on_ready`) | 3: tree (#1), Card Edits blob (#2), tree again (#9) | site 1 (#3), `git fetch` 1 (#12; a clone if missing) | 1 `tree.sync` per guild + 1 emoji list (+ uploads for missing emojis) | 2 x 112 KB tree |
| `/query` | 0 | 0 (Discord's proxy fetches the image) | 1 (+1 when the name is ambiguous) | - |
| `/query_ability` | 0 | 0 | **results + 2**: 135 for "damage", 299 for "type==creature" (`limit` defaults to -1); 1 with howmany | - |
| `/update_stats`, nothing changed | **2 conditional** (tree 304 at :1464, then populate_files' tree 304 at query_card.py:76). The analysis-toolkit skill's "1 request" leaves out the second | site 1 (304) | defer + 3 messages | 3 ms load + 8 ms dataframes rebuilt for nothing |
| `/update_stats`, changed tree (k PSDs changed, n new) | 1 tree + 0-3 small files (only if their blob changed) + 1 get_repo (when k+n > 0) + **k x 3** (blob + 2 commit pages) + **n x (2 + N commits)** | `git fetch` 1 (`_refresh_links`); populate_files: edits blob if changed + site 1 (conditional); TTS run (when images/rules/edits/save changed, or owed, which is always the case on the first changed tree after a restart): see the next row | emoji sync: 0 calls if no icon changed, else 1 list + per icon (delete + create) + 1 blob GET each | about 20.6 MB per PSD |
| example: 1 card's PSD+PNG edited | 5 + TTS 8-12 = 13-17 | site 2, git fetch 2 | about 6 | about 26 MB |
| `/update_stats force_update:True` | 1 + 1 + 904 x 3 = **2,714** | as above | progress edits every 25 cards | **18.6 GB** |
| `/sync_emojis` | 1 tree (unconditional) + 1 blob per missing/edited icon | - | 1 list + per icon delete + create | 112 KB |
| `/sync_emojis refresh:True` | 1 + 91 blobs | - | 1 + 126 deletes + 126 creates | - |
| `/update_tts_save`, no change | **8** (ref, commit, 3 trees, 3 blobs) | site 1 (unconditional), `git fetch` 1 | 3-4 messages + report file | 5.9 MB down |
| `/update_tts_save`, changed, token can write | 12 (+ POST blob, tree, commit, PATCH) | same | same | 5.9 MB down + 7.8 MB up |
| `/update_tts_save`, changed, token can't write | 9: the 403 probe **uploads 7.8 MB every run** before it switches to SSH | + `git push` + `git fetch` | same | 5.9 + 7.8 MB |
| lagging history clone | x up to 3 attempts, 10 s apart (tts_save.py:915-920) | | | 3 x 5.9 MB |

## 2. Background loops and tasks: what can stop them for good

The three `tasks.loop`s are in main.py:762-816. discord.py retries only the exceptions in its retry set (section 0). Any other exception stops the loop until a restart, and the popped item is lost in both cases.

### 2.1 `process_dispatch_queue` (main.py:771-799), B5: CONFIRMED, plus new vectors

1. **Channel `None`** (main.py:799): `client.get_channel(id)` returns None (deleted channel, uncached thread/DM, user-installed context) -> `AttributeError` inside the `except` -> loop dead.
2. **Content over 2,000 characters** (new concrete triggers):
   - `/queue` and `/prev_queue`, music_player.py:366-371 and 377-382: the flush check `len(queue_msg) > BREAK_LEN` (1950) runs *after* a path is appended, so a chunk reaches up to 1950 + path + 4. Measured: 64-94% of 100-song `/queue` calls send one, and `/play <playlist>` builds such queues routinely.
   - **Any user can kill output for everyone** with a long slash option: `/query_ability` echoes `query` in "Found N results for '...'" (query_card.py:536). The same goes for `/play` ("Downloading: {song}", music_player.py:266/270), `/delete_song` (:436) and `/query_rulebook` (query_card.py:587). Discord accepts up to 6,000 characters in a string option. The first send gets 400, the fallback gets 400 again, and the loop dies.
3. **`Forbidden` in the fallback** (new): every reminder and FakeInteraction send goes through `channel.send`. A reminder in a channel the bot can see but can't post in (lost Send Messages, or a read-only channel) raises `Forbidden`, a subclass of `HTTPException`, inside the `except`, and the loop dies. `_channel_missing` (analytics.py:627) checks only visibility.
4. **Empty attachments in the fallback** (new, measured): the first attempt reads the `File`, and discord.py's `reset(seek=0)` doesn't rewind on the first try, so the fallback `channel.send` uploads **0 bytes**. That hits `/export_cards`, `/frankenstein` and the TTS report after a token expiry or a 4xx. A 413 "too large" fails the fallback too, and the loop dies.

**Fix** (one helper, used by both loops; keeps `ephemeral` dropping and the FakeInteraction AttributeError path):
```python
async def _fallback_send(interaction, kwargs):
    for f in [kwargs.get("file"), *kwargs.get("files", [])]:
        if f:
            f.reset()                                   # rewind: the failed try read it
    channel = client.get_channel(interaction.channel_id)
    if channel is None:
        print(f"Dropped a message: channel {interaction.channel_id} not found")
        return
    try:
        await channel.send(**kwargs)
    except HTTPException as e:
        print(f"Dropped a message for channel {interaction.channel_id}: {e}")
```
Then wrap each drain iteration in `try: ... except Exception as e: print(...)`. Before the send, split over-long content: `parts = split_long_message(content)`, send `parts[0]`, and `dispatch_queue.appendleft` the rest in reverse order. That splitting is the one behaviour change, and B5's charter covers it.

### 2.2 `process_edit_queue` (main.py:803-816)
It has the same unguarded fallback at :816, so vectors 1, 3 and 4 apply. Reminder-run `/update_stats` progress edits always take that path. **Fix:** use the same `_fallback_send` and the same per-item `try`.

### 2.3 `process_command_queue` (main.py:762-767) and `run_command` (:753-758)
The body can't raise, so the loop survives. But the `create_task` results are dropped, so a failure in queued work gets nothing better than a "Task exception was never retrieved" line, if that. Silent today: `populate_files` (B7), `card_repo.prep_dataframes`, `_download_song`/`_download_playlist` (music_player.py:244-265; a failed yt-dlp download never tells the user), and `sync_guild_commands` (main.py:111). **Fix:** keep a module-level `set` of tasks with `task.add_done_callback(_reap)`, where `_reap` discards the task and prints `func.__name__` plus the traceback when `task.exception()` is set. For the music downloads, also wrap the two download closures in a `try` that calls `send_to_last_channel(f"Download failed: {e}")`.

### 2.4 `check_reminder` (analytics.py:835-850)
It's robust: a non-blocking `_check_lock` released in `finally`, a `try` per reminder (:720), and a guarded save (:730). It does take a default-executor thread every second, and that executor (`min(32, cpu+4)` workers, 10 on a 6-core jetson) is shared with every `to_thread` slash worker, so 10 long workers delay reminders. Low risk; no change needed.

### 2.5 `on_ready` firing again (new)
discord.py dispatches `on_ready` again after any failed RESUME, for example after the loop was blocked by `/pull` past the heartbeat. main.py:102-122 then re-syncs every guild, re-queues all five inits, and `process_command_queue.start()` raises `RuntimeError` ("already launched"), so a dead dispatch loop is *not* revived. `init_psd` re-runs `stats_db.load()` (psd_analyzer.py:2298), which rebinds `stats_db.stats` under a running `/update_stats`: cards parsed before the reload drop out of the save. **Fix:** set a `_started` flag. On a repeat call, return after `for l in (loops): if l.failed(): l.start()`; that skips the inits and revives only failed loops.

## 3. Temp files and partial writes

| file:line | Write | Cleaned / atomic? | Fix |
|---|---|---|---|
| psd_analyzer.py:549-554 `StatsDatabase.save` | `open('wb')` + `pickle.dump` of stats.pkl and old_stats.pkl | **Not atomic.** A truncated pickle comes from (a) `/restart_bot` (ungated) or `/update` exec-ing during a save, (b) a concurrent `/update_stats` (user + reminder) or `/mass_replace`/`/update_metadata` mutating the dict mid-dump ("dictionary changed size", file already truncated), (c) `/push` `git add` reading a half-written file (update_bot.py:19-23). The load only catches `EOFError`/`FileNotFoundError` (:568, :579): an `UnpicklingError` kills `init_psd`, an `EOFError` silently starts from `{}` and the next run re-downloads 18.6 GB, and a later `/push` publishes it to the website | Copy `save_reminders` (analytics.py:530-547): tmp + `fsync` + `os.replace` per file, plus a `threading.Lock` held for `update_stats`' whole run and every `save()`. `git_push` should refuse while it is held ("/update_stats is running") |
| query_card.py:331-334 `save_aliases` | same pattern for aliases.pkl | not atomic | same tmp + replace |
| psd_analyzer.py:1433-1442 `_download` | `mkstemp(.psd)` then streaming | **Leaks on a mid-stream error**: `ReadTimeout`/`ChunkedEncodingError` after `mkstemp` returns None without removing the partial file (up to 84.5 MB each) | `except ...: if name: remove(name)`, and use `mkstemp(prefix="ac_psd_")` so a sweep at `init_psd` can delete leftovers from a killed process (up to 4 files, which exec does leave) |
| psd_analyzer.py:1658-1659, 1686-1692 | per-file `finally` + pool cleanup | yes (fixed 2026-10-19) | - |
| psd_analyzer.py:2214-2219, 2231-2266, 2279-2289 exports | fixed names in the bot's cwd (`*.csv`/`*.txt`), reopened for the `File` | overwritten non-atomically; two concurrent exports race inside the 1 s before the dispatcher reads them; the reopened handle is never closed by discord.py (`File(fp)` doesn't own it) | Build in `BytesIO`/`StringIO` and send `File(BytesIO(...), filename)`: no disk, no race |
| analytics.py:530-547 | reminder.pkl | atomic + fsync, tmp removed on error | - |
| type_emojis.py:130-135, repo_history.py:102-107 | json tmp + replace | atomic (no fsync; derived data, fine) | - |
| repo_history.py:123-128 | clone into `.tmp`, rename | a timed-out clone leaves `.tmp`, deleted on the next try | - |
| tts_save.py:899 | `TemporaryDirectory` | removed even on exceptions | - |
| tts_save.py:683-687, 696 | standalone backup / tmp + replace / report | atomic | - |
| tts_save.py:791-819 | `hash-object -w`, `mktree`, `commit-tree` in the history clone | each rejected or failed push leaves about 1-2 MB of unreachable objects until `git gc` (2 weeks) | acceptable; optionally `git gc --prune=now` after a failure |
| music (yt-dlp) | `.part` files | left in `music/` after a failed download and listed by `/list` | delete `*.part` in the download-failure handler (2.3) |

## 4. Specific bugs

### B4: `init_vc` (music_player.py:178), CONFIRMED with a correction
`await queue_message(...)` appends the message first, then raises `TypeError` on `await None`. So the user **does** get "You are not in a voice channel." What remains is a logged traceback. The exception is also the only thing stopping `slash_play` (main.py:506-508) and `slash_play_all` (main.py:514-516) from going on to `play_music`. Just dropping the `await` would queue and download songs with no voice client. **Fix:** have `init_vc` return `False` there (`True` at the end), and change the two wrappers to `if not await init_vc(interaction): return`.

### B7: startup 5xx, CONFIRMED and wider than described
- query_card.py:80: any non-304 status runs `response.json()`. An HTML 5xx raises `JSONDecodeError`, and with an empty startup registry `/query` says "No card found!". A JSON error body (401 bad token, 403 rate limit, some 5xx) is worse: `tree = {"message": ...}`, `git_files` is cleared (:82), **all cards vanish silently even mid-session** (populate_files runs after every `/update_stats`, :1895), and `_tree`/`_tree_etag` are overwritten with the error.
- query_card.py:706-712 `set_repository`: it sets `repository` *before* fetching, so a typo wipes the card list and keeps the bad repo. It is also `async def`, so up to 3 x 30 s of blocking requests run **on the event loop**.
- type_emojis.py:296-299: if the GitHub listing fails, `fetch_application_emojis` is never reached. `type_emojis` stays `{}` and every `[fire]` renders as text until `/sync_emojis` or a changed-tree `/update_stats`. A 304 run doesn't help.

**Fix:** in `populate_files`, return early on any status other than 200/304 without touching state. Try 3 times (sleep 1 s, 2 s) on 5xx or `RequestException`. In `set_repository`, make it a plain `def` (so `to_thread` threads it), use `queue_message`, and restore the old repo plus a re-fetch when the status isn't 200. In `sync_type_emojis`, fetch application emojis first and `_publish(existing)` even when the listing fails. Also have `_list_type_icons` send `If-None-Match` with `card_repo._tree_etag` and reuse `card_repo._tree` on 304. That cuts one 112 KB tree download at startup and makes `/sync_emojis`' tree free (startup goes from 3 to 2 API requests).

### C6: `/list_guild_members` (management.py:102-136; main.py:88-90), CONFIRMED
discord.py 2.7.1's `Guild.chunk()` and `fetch_members()` both raise `ClientException` without `intents.members`, so the command always answers "Intents.members must be enabled to use this." **Fix, one of two:**
- (a) Remove the command; it's not in help.py (C2).
- (b) First turn on "Server Members Intent" in the Developer Portal. **Until that is done, a bot with `intents.members = True` gets `PrivilegedIntentsRequired` at login and won't start**, so do it before deploying. Then set `intents.members = True` and `Client(intents=intents, chunk_guilds_at_startup=False)`, so startup doesn't chunk every guild and the existing on-demand `chunk()` works.

### Also found: a failed PSD download leaves the card stale with no warning (psd_analyzer.py:1440-1442, 1485)
`_download` prints and skips. `traverse_remote` still records the new tree ETag, so the next `/update_stats` gets a 304 and that card keeps its old data until something else changes in the repo. The user is never told. **Fix:** count the failures. When there are any, skip `_tree_etags[repository] = ...` (or call `_forget_tree`) and add "N downloads failed (next /update_stats retries)" to `_send_summary`.

### Also found: `/update_stats` fails silently (psd_analyzer.py:1469, 1710)
A 5xx on the tree (`raise_for_status`) or a `GithubException` from the commits lookup escapes `update_stats`. The deferred reply stays at "thinking", and only stdout shows the error. Reminder runs do report it (analytics.py:651-659). **Fix:** wrap `traverse_remote` in `except (RequestException, GithubException) as e: _notify(f"Card statistics update failed: {e}. Nothing was saved; run /update_stats again."); return []`.

## 5. Self-update (/push, /pull, /update): risks not in run-and-operate §5

1. **No timeout on any git network call, and they run on the event loop** (update_bot.py:28, 35, 49, 65, 93, 105, 119). Weak point 7 says "freezes briefly". In fact a stalled HTTPS fetch or SSH push freezes the bot **indefinitely**: no heartbeat, no queues. **Fix:** add `kill_after_timeout=120` to each network call (GitPython accepts it as a call kwarg; verified it is in `git.cmd.execute_kwargs`).
2. **An SSH host-key or passphrase prompt hangs the bot on the Mac.** The push remote is `ssh://` (`git remote -v`), and there's no `core.sshCommand` and no `BatchMode`. `ssh` reads `/dev/tty`, so a bot started from a terminal waits on it forever, for example on the first push from a new machine. **Fix:** `Git('.').update_environment(GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15")`. tts_save.py:775-780 already does this.
3. **The interaction token expires during a long hang.** All the update_bot messages are `interaction.followup.send`, which works for 15 minutes. After that the error message inside `except` (update_bot.py:38, 98, 154-155) raises too, the real git error is lost, and with `/update` the queue loops stay stopped. Fix 1 bounds this. Also route the messages through a helper that falls back to `print` on `HTTPException`.
4. **Messages over 2,000 characters break `/pull`'s reporting.** "Pull complete:\n```{result}```" (update_bot.py:94) carries git's diffstat, and one day of commits (2026-10-20) is 2,950 characters, so the send gets 400 `HTTPException`. That isn't a `GitCommandError`, so it escapes: the pull succeeded, but the user only ever sees "Fetching from remote...". `/update` normally reaches this line with "Already up to date." because `git_push`'s rebase already pulled. The same thing can happen with "Push failed: {e}" / "Git pull failed: {e}" (:38, :98) when stderr is long. **Fix:** `text[-1800:]` in all three.
5. **`/push`, `/update` or the ungated `/restart_bot` during a running `/update_stats` can stage or leave a truncated pickle** (section 3, first row). The atomic save plus lock fixes all three.

## 6. Cheap optimizations (measured gain)

| Change | Where | Gain |
|---|---|---|
| Commit lookups via `get_page(0)` with `per_page=100` (1 request: newest = [0], first = [-1] for ≤100 commits) | psd_analyzer.py:1408, 1710-1717 | per changed card 2 -> 1; per new card 1+N -> 1 (9 -> 1 measured); forced sweep 2,714 -> 1,810 requests |
| Remember "token can't write" after the first 403 | tts_save.py:925-930 | 7.8 MB upload saved on every later run |
| Tree reuse/ETag for the emoji listing (B7 fix) | type_emojis.py:101 | startup 3 -> 2 API requests, -112 KB; /sync_emojis tree becomes a 304 |
| Skip populate_files + prep_dataframes after a 304 | psd_analyzer.py:1895-1896 | no-change /update_stats 2 -> 1 API request; -8 ms of CPU |
| One `requests.Session` per thread (`threading.local`) for api.github.com | psd_analyzer.py:1414, query_card.py, type_emojis.py | 85 -> 17 ms per request; about 61 s of handshakes saved in a 904-PSD sweep |
| Parse the tree once (`self.tree`) instead of 7 `response.json()` calls | psd_analyzer.py:1470-1637 | 6 x 2.1 ms saved per changed run (low) |
| `/frankenstein`: `raise_for_status()` (a 404 now shows as "No card found") | utils.py:96 | correctness; optionally cache by pinned URL (immutable), 1.97 MB per card |
| `/query_ability` sends one message per result (135-299 for common queries) through the single serial dispatch loop, which delays every other user's replies by minutes at Discord's per-webhook pace | query_card.py:538-542 | **behaviour change, owner's call:** a default `limit` or several links per message |
