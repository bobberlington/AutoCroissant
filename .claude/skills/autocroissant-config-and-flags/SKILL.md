---
name: autocroissant-config-and-flags
description: Catalog of EVERY configuration axis in the AutoCroissant Discord bot. Load for any question containing "config", "flag", "setting", "default", "where is X configured", "how do I change/set X", or "why did my setting reset" — and for any mention of TOKEN, ADMINS, GIT_TOKEN, ACTIONS_TOKEN, CARD_SYNC (removed 2026-10-25), CARD_GUARDS, card-preview.yml and the previews branch (KEEP, DAYS: no secret needed), browser_cookies, KEEPVIDEO, POSTPROCESS, SITE_URL, match ratio, /set_ratio, /set_repo, admin-only commands, rate limits, BREAK_LEN, UPDATE_RATE, DOWNLOAD_WORKERS, TYPE_REGION_RATIO, EXCLUDE_FOLDERS, MUSIC_BASE_DIR, LOCAL_DIR_LOC, HISTORY_CLONE, TTS_SAVE_PATH, AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE, TIMEZONE/PST, the pickle filenames (removed 2026-10-25), ICON_FONT_PREFIX, ICON_MANIFEST_PATH, ICON_VARIANTS, ICON_SCAN_PATTERN, EFFECT_LAYER_TAG, EFFECT_COVERAGE_MIN, AUTHORS_PATH(S), EDITS_PATH(S), STAR_CORE_ALPHA, the type_emojis constants (EMOJI_SIZE, MAX_NAME_LEN, ICON_FOLDERS, NO_SUFFIX, NO_RED, TOKEN_PATTERN), the reminder constants (REMINDER_VERSION, MAX_INTERVAL, CHANNEL_RETRY, TIME_FORMATS, WEEKDAYS), or the AI fields and setters that live only on branch archive/ai-generation (HF_TOKEN, model, lora, device_no, scheduler_name, vram_usage, /set_scheduler, /set_model, /set_lora, /set_device). Contains the config.py field table (names only), global_config.py and per-module constants (incl. card_sync, the push guards, psd_edit parity, card_bake), the TTSCardMaker repo settings (names only), prod-vs-experimental status, runtime-settable vs restart-required (/set_* do NOT persist), update_stats defaults, an add-a-config-field checklist and re-verification greps.
---

# AutoCroissant: Configuration and Flags

All facts verified against the working tree on 2026-07-11; re-verified 2026-10-19 against HEAD a7a4bb3 plus
the uncommitted working tree (new: icon-font constants in psd_analyzer, commands/type_emojis.py, the
reminder constants in analytics.py, `force_update` default); re-verified 2026-10-20 against HEAD 9fb3093
(AI code only on branch `archive/ai-generation`, `browser_cookies`, `SITE_URL`); re-verified 2026-10-23 against main 2ada357
(ACTIONS_TOKEN, the global_config additions, the push guards, parity and bake knobs, the admin gates). Line numbers and defaults DRIFT —
before relying on any table here after the repo has moved, rerun the one-line commands in
"Provenance and maintenance" at the bottom. The master drift check is:

```
grep -rn "getattr(config" --include="*.py" .
```

(cwd = repo root `/Users/michaelsrouji/Desktop/AutoCroissant` for every command in this file.)

## The four configuration layers

| Layer | Where | Committed to git? | Takes effect |
|---|---|---|---|
| Secrets + per-machine knobs | `config.py` (repo root) | NO — gitignored, mode 0700 | on restart only (read once at import) |
| Shared constants | `global_config.py` | yes | on restart (code edit) |
| Module constants | the `# Configuration` block at the top of each `commands/*.py` | yes | on restart (code edit) |
| Runtime setters | `/set_ratio`, `/set_repo` (admin only since 2026-10-22; the four AI setters are on branch `archive/ai-generation`) | n/a | immediately — and LOST on restart |

Mental model: `config.py` is read ONCE, at import time, into module-level constants via
`getattr(config, "field", default)`. Nothing in the codebase ever re-reads it, and nothing ever
writes it. Consequences: (a) editing `config.py` does nothing until restart; (b) `/set_*` changes
do nothing after restart. Both directions of that asymmetry confuse people — see the
"Runtime-settable vs restart-required" section.

## config.py — the secret per-machine file

Ground rules:

- SECRET. It holds tokens. Never print, `cat`, quote, or commit its VALUES. Field names are
  public (listed below). File mode is `-rwx------` (0700) — keep it that way (verified
  2026-07-11 and 2026-10-19: `ls -l config.py`).
- Gitignored (`.gitignore`, "Discord Bot Specific" section: `config.py`). Confirm with
  `git check-ignore -v config.py`.
- PER-MACHINE. Production is the jetson (systemd `autocroissant.service`); the owner develops on
  a Mac. Each has its own `config.py` with different values. A field you add on one machine does
  not exist on the other until someone hand-edits that machine's file. Every consumer therefore
  uses a `getattr` default — except TOKEN.
- The file must EXIST for the bot (and for any script importing `commands.*` through utils) to even import:
  the bot imports it at startup (utils and music_player do `import config`); github_api and card_sync take it
  optionally (try/except), and everything else reaches it through utils (autocroissant-architecture-contract
  invariant 6). Minimum viable `config.py` = a single `TOKEN = ...` line; every other field falls back to its default.
- Side effect worth knowing: main.py prints one token-PRESENCE line at import
  ("Git token found, API limited to 5000 requests/hour." or the 60/hour variant,
  `print_token_status` in commands/github_api.py). Presence only — values are never printed.

### The 8 fields main reads (as of 2026-10-23)

| Field | Required? | Default (getattr) | Read at | Status | Effect |
|---|---|---|---|---|---|
| `TOKEN` | YES — bot won't boot without it | none — the ONLY hard import: `from config import TOKEN` (main.py:13); used by `client.run(TOKEN)` (main.py:987) | main.py | prod | Discord bot token. It decides the bot IDENTITY, and since 2026-10-19 that matters beyond login: application emojis belong to the identity (autocroissant-run-and-operate §11). On the dev Mac (2026-10-19) the active line is the "Waxwell" identity and the AutoCroissant token line is commented out |
| `ADMINS` | no | `[]` (commands/utils.py:19) | utils.py | prod | list of Discord user IDs (ints) allowed past `lacks_admin` — see gate table below |
| `GIT_TOKEN` | no (strongly recommended) | `""` (commands/github_api.py — the one read since 2026-10-20; query_card, card_data (the traverser's code since 2026-10-22) and type_emojis import it from there; git_commit.token() reads it, else the Actions' GITHUB_TOKEN, for card_bake and card_guards) | github_api | prod | GitHub API auth: raises the rate limit 60 → 5000 req/hr. Used for the Authorization headers (the PyGithub client for commit timestamps until 2026-10-25) and the emoji sync's tree listing (1 call per startup or /sync_emojis). Since 2026-10-20 it also WRITES: `/update_tts_save` commits the TTS save to the card repo with it, so it needs write access there (when the token can't write — 401/403/404 on the blob upload, or no token — it commits through the history clone and pushes over SSH with the service's agent instead) |
| `ACTIONS_TOKEN` (2026-10-22) | no | `""` (commands/card_sync.py) | card_sync | prod | A fine-grained token for TTSCardMaker only, Actions read and write. `/update_stats force_update` or `cards:` dispatches card-sync.yml with it and follows the run. (Until 2026-10-25 `/push` also dispatched the website build with it after a push of the pickles (card_sync.dispatch_site). In the jetson's config.py since 2026-10-22 ~12:25 |
| `browser_cookies` (2026-10-20) | no | `False` (music_player.py:37) | music_player | prod | True = yt-dlp uses Safari cookies — see the browser_cookies section below |
| `KEEPVIDEO` | no | `True` (commands/music_player.py:54) | music_player | prod | yt-dlp `keepvideo`: keep the downloaded media file after any post-processing |
| `POSTPROCESS` | no | `False` (music_player.py:61) | music_player | prod | if True, appends a yt-dlp `FFmpegExtractAudio` postprocessor: mp3 at 320k (music_player.py:61-67) |

### AI fields (branch `archive/ai-generation` only)

Main never reads these; a config.py that still has them is harmless. On the branch they are read in
commands/diffusion.py:26-31:

| Field | Default | Effect |
|---|---|---|
| `HF_TOKEN` | `""` | HuggingFace token, used ONLY by `_init_flux_pipeline` to download the gated `black-forest-labs/FLUX.1-dev`. Irrelevant unless `model` contains "flux" |
| `model` | `""` | `""` DISABLES pipeline init (`if not MODEL: return`). Otherwise by substring: "flux" → Flux from HF; "xl" → SDXL from `./models/<model>`; else SD1.5. A missing file is caught by init's try/except — the bot keeps running |
| `lora` | `""` | LoRA filename in `./models/loras/`; loaded only if the file exists |
| `device_no` | `0` | CUDA GPU index (`gpu_id` of the CPU-offload calls); matters only for `vram_usage` low/medium |
| `scheduler_name` | `""` | `""` or unrecognized → the model's default scheduler. Recognized: "dpm++ sde" (+ Karras if it ends in "karras", not flux), "euler a". `POSSIBLE_SCHEDULERS` is enforced only by `/set_scheduler`, so a config.py typo silently falls back |
| `vram_usage` | `"low"` | memory strategy — mode table below |

Prod-vs-experimental doctrine (user-stated 2026-07-11): the AI/diffusion capability is an
EXPERIMENT. Since 2026-10-20 it lives only on branch `archive/ai-generation`; main has no AI code
and boots without any AI field or torch. Details: **autocroissant-ai-boundary** (branch `archive/ai-generation`).

### browser_cookies — the music downloader's Safari cookies

`commands/music_player.py:37` reads
`cookies_from_browser: bool = bool(getattr(config, "browser_cookies", False))`;
when true, yt-dlp gets `"cookiesfrombrowser": ("safari", None, None, None)` (music_player.py:55). Set
`browser_cookies = True` only on a machine with Safari (the Mac; the jetson has False). This replaced the
vram_usage-only hack on 2026-10-20 (docs/BUGS_AND_IMPROVEMENTS.md S1); the `vram_usage` fallback went 2026-10-26.

### ADMINS — which commands it gates

`lacks_admin(interaction)` (commands/utils.py; named `perms_check` until 2026-10-20) returns **True when the
user is NOT an admin** (`interaction.user.id not in ADMINS`). Every call site reads
`if lacks_admin(...): send "You do not have permission"; return`. The invariant is owned by
**autocroissant-architecture-contract**.

Commands gated by `lacks_admin` in main.py (all 15 call sites, verified 2026-10-25 after /push went; line numbers
drift — re-derive with `grep -n "lacks_admin" main.py`):

| Slash command | main.py line |
|---|---|
| `/stop_bot` | 216 |
| `/pull` | 232 |
| `/update` | 258 |
| `/set_reminder` | 288 |
| `/set_repo` (added 2026-10-22) | 392 |
| `/update_stats`, only with `force_update` or `cards` given (2026-10-22: a Card sync run costs API budget; a reminder's run is checked as its creator) | 434 |
| `/delete_song` (added 2026-10-22) | 604 |
| `/delete_all_music` (added 2026-10-22) | 613 |
| `/purge` (added 2026-10-22) | 633 |
| `/list_guilds` | 643 |
| `/leave_guild` | 658 |
| `/sync_global` | 667 |
| `/sync_emojis` (added 2026-10-19) | 682 |
| `/list_guild_channels` | 698 |
| `/get_channel_messages` | 718 |

The 2026-10-22 gates are whole, view modes included (`/set_repo` with no arguments). `/update_metadata`, `/mass_replace`,
`/alias`, `/bake_cards` and `/update_tts_save` (all admin only) were retired 2026-10-25 (branch `archive/discord-edits`).
`/purge` cannot be scheduled by a reminder: its `user` option has no default. Tests: `tests/test_admin_gates.py`.

`/list_guild_members` was removed 2026-10-20 (C6: it needed the privileged members intent, which the bot
does not enable).

Two more ADMINS uses outside main.py since 2026-10-19 (commands/analytics.py):
- `/remove_reminder` and the `/reminders` panel's Reschedule, Turn off/on and Delete: admins, or the reminder's
  creator (`can_edit`; no creator: admins only). The panel's New is admin only, like `/set_reminder`.
- A firing reminder runs as its creator, so every gate above is re-checked against the creator at fire time.
  A reminder with no creator (saved before 2026-10-19) runs as an unknown non-admin user since 2026-10-26.
  Mechanism: autocroissant-architecture-contract invariant 9.

Quirk, stated plainly: `/restart_bot` (main.py:167-175) is NOT gated — anyone in the guild can
restart the bot, while `/stop_bot` is admin-only. Observed code fact as of 2026-10-20; whether
that is intentional is unknown (candidate cleanup — route through autocroissant-change-control).

### vram_usage — the five modes

> **AI left main on 2026-10-20:** this section applies to branch `archive/ai-generation` (main has no AI code).


Accepted values are never validated anywhere; they are plain string comparisons in
`_apply_memory_optimizations` (diffusion.py:377-392) and `init_pipeline` (diffusion.py:212-214).
A typo (e.g. "lo") silently lands in a fallthrough: float16, VAE slicing+tiling, no offload, no
device placement. One line per mode — **the canonical full mode table (dtype, offload, VAE opts,
quantization, per-mode caveats) lives in autocroissant-ai-boundary** (branch `archive/ai-generation`):

| Value | One-line behavior |
|---|---|
| `low` (default) | float16; VAE slicing+tiling; sequential CPU offload (`gpu_id=device_no`) — least VRAM, slowest |
| `medium` | float16; VAE slicing+tiling; model CPU offload (`gpu_id=device_no`) |
| `high` | float16; `_apply_memory_optimizations` applies NOTHING (no VAE opts, no offload). No code path explicitly moves the pipe to CUDA in this mode (the only pipe `.to(...)` is the mps one, line 380; line 699's `device` only seeds the `torch.Generator`) — UNVERIFIED on CUDA (detail: autocroissant-ai-boundary (branch `archive/ai-generation`)) |
| `mps` | **float32**; `pipe.to(device='mps')`, nothing else; Flux unquantized (bitsandbytes is CUDA-only). On main, `mps` only matters as the browser_cookies fallback (above) |
| `distributed` | float16; VAE slicing+tiling; no CPU offload; `device_map='balanced'` when >1 CUDA GPU — wired only into the Flux init path (diffusion.py:214, 218) |

## global_config.py — committed shared constants

Whole file, 45 lines on 2026-10-23 (verify: `cat global_config.py`). Three constants read the environment first
(`environ.get(name, default)`): `AUTOCROISSANT_LOCAL_DIR` and `AUTOCROISSANT_HISTORY_CLONE` (2026-10-20, so TTSCardMaker's
Actions can run `python -m commands.tts_save` and the build on their checkout without a symlink); `AUTOCROISSANT_BOT_CLONE`
went with the pickle seed on 2026-10-25. Modules also read a few environment variables themselves:
`PSD_TOOLS_FONT_DIRS` (`FONT_DIRS_ENV`: card_bake, font_check), `RUNNER_TEMP` (card_bake's work dir in the Action),
`GITHUB_TOKEN` (the Actions' token when config.py gives none: github_api.token_from_env, git_commit.token), `BUILDER_SHA`
(the card_data build's stamp), `DISCORD_ARCHIVE` / `DISCORD_MATCH_REPO` (the Discord archive CLIs) and `SSH_AUTH_SOCK`
(tts_save words its SSH-push error with it). repo_history passes the environment to git with `batch_overrides`
(GIT_TERMINAL_PROMPT=0, ssh BatchMode) and card_history with `GIT_NO_LAZY_FETCH=1`. Verify:
`grep -rn environ --include="*.py" commands/ main.py global_config.py`:

| Constant | Value | Meaning |
|---|---|---|
| `LOCAL_DIR_LOC` | `"~/Desktop/TTSCardMaker"`, env `AUTOCROISSANT_LOCAL_DIR` | the card repo checkout. `/update_stats` no longer reads it (local mode was archived 2026-10-20); commands/tts_save.py reads `TTS/decks.json` and `site/Card Edits.json` under it in standalone runs (the Action sets it to `$GITHUB_WORKSPACE`); the diagnostics scripts (parse_one.py, gap_trace.py) and the archived traversal on branch `archive/local-update-stats` expand it with `expanduser`. This path convention is load-bearing across the toolchain |
| `SITE_URL` (2026-10-20) | `"https://michaeljsr.github.io/TTSCardMaker"` | the card browser: `/website`, `/query` Card page buttons, search links, the site's `data/cards.json` (main.py, query_card.py) |
| `REMIND_JSON` (2026-10-24) | `"reminder.json"` | the reminder store (GITIGNORED, per machine; schema_version 2) |
| `HISTORY_JSON` (2026-10-20) | `"repo_history.json"` | repo path -> (blob, commit that last changed it), written by commands/repo_history.py; gitignored, rebuilt from the clone |
| `FONT_DIRS_ENV` (2026-10-21) | `"PSD_TOOLS_FONT_DIRS"` | the name of the environment variable whose first entry overrides the fonts dir (card_bake, font_check) |
| `CREATED_PATH` (2026-10-22, 6fe7a90) | `"site/Card Creation Dates.json"` | when each card was made (written by `python -m commands.card_dates created`); read on every /update_stats (`CardDates`) |
| `RATINGS_PATH` (2026-10-24) | `"site/Card Ratings.json"` | dated card ratings, whole numbers 1-10 (`python -m commands.tts_ratings`; the website adds entries; `CardRatings`); never read by the bot, the card sync or the checks |
| `DATA_BRANCH`, `CARD_DATA_DIR` (2026-10-22) | `"data"`, `"card_data"` | the card repo's branch holding `db/` (written by Card sync only) and the bot's gitignored copy of it |
| `CARD_TREE_JSON` (2026-10-25) | `"card_tree.json"` | the card repo's last tree, its ETag and Card Edits.json's text (query_card `load_saved`/`_save_tree`): the card list at startup before any request; gitignored, per machine; a corrupt one is ignored and rewritten |
| `EMOJI_JSON` (2026-10-20) | `"type_emojis.json"` | bot application id -> {type emoji name -> blob sha of the icon it was made from} (emojis belong to the bot identity), written by commands/type_emojis.py; gitignored, per machine. Missing, unreadable or no entry for this bot = its existing emojis count as current (no re-upload) |
| `HISTORY_CLONE` (2026-10-20) | `"~/Documents/TTSCardMaker-history.git"`, env `AUTOCROISSANT_HISTORY_CLONE` | bare blobless clone of the card repo, fetched on startup and on every /update_stats; cloned on first use (the Action sets `~/TTSCardMaker-history.git` on its runner). No content diffs or renames run there (they would download blobs): repo_history runs rev-list/ls-tree/merge-base/fetch, and since 2026-10-22/23 `card_history.GitIndex.from_git` runs `git log --raw --no-abbrev --no-renames` with `GIT_NO_LAZY_FETCH=1` (the archive dates, the history recovery); `mailmap_args` fetches head's `.mailmap` blob, the one lazy fetch. It must be a full (non-shallow) clone, else the index raises ValueError and the dates wait for the next run |
| `CARD_REPO`, `CARD_BRANCH` (2026-10-20) | `"MichaelJSr/TTSCardMaker"`, `"main"` | the card repo and its branch, moved here from query_card (`DEFAULT_REPOSITORY`, kept as an alias), repo_history (`REPOSITORY`, alias) and tts_save; /set_repo still overrides `card_repo.repository` at runtime |
| `API_BASE`, `RAW_BASE` (2026-10-20) | `"https://api.github.com/repos/"`, `"https://raw.githubusercontent.com/"` | URL prefixes used by commands/github_api.py (API) and repo_history/tts_save (raw links) |
| `TYPES_DIR`, `STARS_DIR`, `RULEBOOK_DIR` (2026-10-20) | `"Types"`, `"Types/Stars"`, `"Rulebook"` | card-repo folders: type icons, star icons, and the rule pages (a substring test on card paths); used by psd_parser, card_data, query_card, type_emojis |
| `ICON_MANIFEST_PATH`, `AUTHORS_PATH`, `EDITS_PATH`, `RULES_PATH` (2026-10-20) | `"MDW/Fonts/stoogedings-src/stoogedings.json"`, `"site/Card Authors.txt"`, `"site/Card Edits.json"`, `"TTS/decks.json"` | the hand-kept files in the card repo (font manifest, card authors, website edits, TTS deck rules). The website files moved from `MDW/` to `site/` on 2026-10-20 (TTSCardMaker 13ac680); the one-path `*_PATHS` tuples were collapsed to these on 2026-10-25. Moved here from psd_analyzer and tts_save (psd_analyzer re-exports them) |
| `MAILMAP_PATH` (2026-10-23) | `".mailmap"` | the card repo's git mailmap (one person's git names as one): `card_history.mailmap_args` passes head's blob to `git log` for `%aN` (the API path's `card_edits.Mailmap` until 2026-10-25) |
| `TTS_SAVE_PATH` (2026-10-20) | `"TTS/Impossibility Simulator.json"` | the Tabletop Simulator save in the card repo, updated by TTSCardMaker's card-sync.yml TTS save step since the cutover 2026-10-24 (tts-save.yml before; /update_tts_save until 2026-10-25); it passes the path on the command line; players download it from its raw link (commands/tts_save.py) |
| `TIMEZONE` | `ZoneInfo("America/Los_Angeles")` | reminder times are PST/PDT — `/set_reminder when:` is interpreted in this zone (ops details: autocroissant-run-and-operate) |

Removed 2026-10-25 (DATA_BUILD_ACTION.md step 10): `ALIAS_PKL`, `STATS_PKL`, `OLD_STATS_PKL`, `DETAILS_JSON` and the
cutover switch `CARD_SYNC` (True since 2026-10-24); the card data is the data branch's (`DATA_BRANCH`, `CARD_DATA_DIR`).
Card data schemas and the data gate are owned by **impossibility-cards-reference** and **autocroissant-change-control**.

## Module-constant catalog

Every `commands/*.py` opens with a `# Configuration` (or `# Configurable Parameters`) block.
Catalog per module, verified 2026-07-11; re-verified 2026-10-20:

### card modules: commands/card_data.py, psd_parser.py, psd_analyzer.py (psd_analyzer split 2026-10-20; the build in card_data.py since 2026-10-22)

The first rows and `UPDATE_RATE`/`EXCLUDE_FOLDERS` live in commands/card_data.py (uncompiled; run by the Card sync
Action). repo_traverser.py (the legacy /update_stats, with `_tree_etags`/`_file_cache`, `LOOKUP_WORKERS` and
`COMMITS_PAGE`) was removed 2026-10-25.

| Constant | Value | Meaning |
|---|---|---|
| `DOWNLOAD_WORKERS` (2026-10-20) | `4` | PSDs the build fetches ahead of parsing |
| `AHEAD` (2026-10-23) | `8` | PSDs fetched and not yet parsed, at most (open PSDs, temp files) |
| `RANGE_READS` (2026-10-23, github_api.py) | `True` | read a PSD by range requests at the tree's commit when psd-tools can (lazy open, fork feat/lazy-records), in the build and `card_check --after`; whole download otherwise and on any failure |
| `RANGE_REFUSALS` (2026-10-23) | `3` | PSDs whose range read gets a 4xx, a 5xx or no connection before the run downloads the rest whole (one odd PSD leaves range reads on) |
| `TITLE_REGION_RATIO` (2026-10-20) | `0.12` | the title region: text layers above this share of the card height are title candidates (`title_layers`); hidden ones name nothing since 2026-10-25 (`alt_names` no longer set) |
| `UPDATE_RATE` | `25` | progress cadence of the build: a line every 25 processed cards (`num_updated % UPDATE_RATE == 0`) |
| `TYPE_REGION_RATIO` | `0.5` | fraction of card height splitting "creature type icons" (above midline) from "inject-into-ability icons" (below). Semantics: impossibility-cards-reference |
| `EXCLUDE_FOLDERS` | `["Markers", "MDW"]` | top-level TTSCardMaker folders skipped during stats traversal |
| `EXPORTED_STATS_NAME` | `"stats"` | `/export_cards` writes `stats.csv` / `stats.txt` in the REPO ROOT (psd_analyzer.py:246, 253) — gitignored via `*.csv` / `*.txt` |
| `EXPORTED_RULES_NAME` | `"rules"` | `/export_rulebook` writes `rules.txt` in the repo root (psd_analyzer.py:307) |
| `COLUMN_ORDER` | field names, `'aliases'`…`'subtype'` (`'subseries'` after `'series'` since 2026-10-20) | display/export field ordering for card text output and the CSV columns (used psd_analyzer.py:115-116, 237) |
| `MISSPELT_CARD_TYPES` | `['undread', 'tornado', 'error']` | layer names that trigger the MISSPELT TYPE validation problem (from 2026-07-11 one live card tripped it, the "20 Creature Types" rulebook page; none since TTSCardMaker 100511d3 renamed its layer to wind, 2026-10-22) |
| `ICON_FONT_PREFIX` (2026-10-19) | `"stoogedings"` | a text-layer style run whose font PostScript name starts with this (case-insensitive) is read as icon tokens; the font's full name is `stoogedings-Regular`. Changing it without renaming the font in TTSCardMaker silently turns token parsing off |
| `ICON_MANIFEST_PATH` (2026-10-19) | `"MDW/Fonts/stoogedings-src/stoogedings.json"` | repo-relative path of the font manifest (aliases + codepoints), read by `_repo_text`: fetched by blob sha through the GitHub API and cached in `_file_cache` (only the offline diagnostics helper `_icons_from_local` reads a local clone). Missing file = plain names only, no aliases or Glyphs-panel characters. A cross-repo contract with TTSCardMaker's build script (ttscardmaker-companion §1.3) |
| `ICON_VARIANTS` (2026-10-19; glow and darkglow 2026-10-23) | `("", "tint", "sticker", "no", "glow", "darkglow")` (psd_parser.py:32) | the `:variant` suffixes the parser accepts (all but `no` read as the plain icon); any other suffix is an `UNKNOWN ICON TOKEN` problem |
| `ICON_SCAN_PATTERN` (2026-10-19) | `\[[^\[\]\r\n\x03]{1,52}\]` or one character in U+E000–U+F8FF or ★ (U+2605) | what the parser treats as a candidate token: a bracketed run of 1-52 characters (the longest icon name with `:darkglow`) on one line, a Private Use Area character, or the star. Semantics: impossibility-cards-reference §4.1 |
| `EDITS_PATH` (2026-10-20) | `"site/Card Edits.json"` | website fixes (`site/` since 2026-10-20, `MDW/` before). Read on every `/update_stats` (`CardEdits`, `_apply_edits`; PSD values kept in `card.parsed_values`) and by `populate_files` for website names/aliases. Fields = the website's EDIT_FIELDS (`CardEdits.TYPES`; `subseries` since 2026-10-20, database-only like note/aliases/hidden: impossibility-cards-reference §6). Missing or bad JSON: warning, cards unchanged |
| `AUTHORS_PATH` (2026-10-20) | `"site/Card Authors.txt"` | card-repo path of the authors file. It is read on every `/update_stats` (`_repo_text`, cached by blob sha) and applied to every card (`CardAuthors`, `_apply_authors`). If missing or unreadable, a warning is printed and authors stay as they are. Semantics: impossibility-cards-reference (Author) |
| `STAR_CORE_ALPHA` (2026-10-20) | `0.5` | opacity at or above which a star layer's pixels count as its core when overlapping layers are merged into one star (`_count_stars`) |
| `EFFECT_LAYER_TAG` (2026-10-20) | `" FX: "` | marks an effect layer (`Ability FX: Blazing`, made by TTSCardMaker `MDW/Scripts/Effect Layers.jsx`). Such layers are skipped by the walk and checked afterwards. Semantics: impossibility-cards-reference §4.2 |
| `EFFECT_COVERAGE_MIN` (2026-10-20) | `0.95` | minimum share of an effect layer's opaque pixels that must sit on its text layer's pixels; below it the card gets `EFFECT LAYER OUT OF DATE` (fresh layers score 1.0, a stale test scored 0.268) |
| `PARSER_VERSION` (2026-10-21) | `4` (psd_parser.py:24; 3 and 4 on 2026-10-25: a PSD's pictures, then the census fixes) | stamped on every parsed card (`CardInfo.parser_version`); the Card sync build re-parses a card whose stamp differs, once (card_data.py:144), and retries a PSD that failed under another version (:909). Bump it when a parser change must reach the stored cards: the next run reads every PSD (~812) |
| `CardValidator.ABILITY_EXCLUSIONS` | 21 card names (psd_parser.py:1056-1078) | cards allowed to have no ability layer |

### the card data build: commands/card_data.py (2026-10-22; docs/Campaigns/DATA_BUILD_ACTION.md)

Build mode only (`python -m commands.card_data`); legacy /update_stats ignores them. CLI flags, not config.py.

| Constant | Value | Meaning |
|---|---|---|
| `BUDGET` | `400` | REST PSD downloads per run (`--budget`); the rest go to `pending.stale` and the next run reads them. Two runs an hour stay at 800 of the token's 1,000 requests |
| `RATE_FLOOR` | `150` | no more REST downloads once `X-RateLimit-Remaining` is below this |
| `MIN_CARDS`, `FLOOR_SHARE` | `100`, `0.95` | gate G1 |
| `ARCHIVE_MIN`, `ARCHIVE_SHARE` | `5`, `0.01` | gate G2: at most max(5, 1%) archived a run (`--allow-archive` lifts it) |
| `REPARSE_MAX` | `25` | gate G4: same-blob re-parses without approval (`--full`, kept in `pending.full` while a budgeted re-parse goes on) |
| `QUALITY_SLACK`, `DETAILS_SHARE` | `3`, `0.05` | gate G5 |
| `SPARSE_FILES` | 18 paths | what card-sync.yml checks out of AutoCroissant: the build's, the bake's and the save's import closure (tests/test_card_data.py runs `--help` on all three CLIs with exactly this list) |
| `EXIT_OK`, `EXIT_ERROR`, `EXIT_GATE` | `0`, `1`, `2` | written or nothing to do; an error (a file not fetched included); a gate refused |
| changelog `MESSAGES_MAX` | `20` | Discord posts per run (one per push); the rest wait for the next run (commands/changelog.py) |

`EXCESSIVE_STAT_EXCLUSIONS` went on 2026-10-21: a stat above 10 is allowed per card by `allow_high_stats` in Card Edits.json.
The exclusion set is a per-card whitelist the owner wants to eventually eliminate ("perfect
extraction" goal). Their semantics and the validation rules live in
**impossibility-cards-reference**; the elimination campaign in
**autocroissant-psd-extraction-campaign**.

### commands/query_card.py (lines 31-32)

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_REPOSITORY` | `"MichaelJSr/TTSCardMaker"` | GitHub repo the bot queries for card images; seeds `card_repo.repository` |
| `DEFAULT_MATCH_RATIO` | `0.6` | difflib fuzzy-match cutoff; seeds `card_repo.match_ratio` |

Both seed fields of the `CardRepository` dataclass whose singleton is `card_repo`
(query_card.py:990) — which is exactly why `/set_ratio` and `/set_repo` don't persist (below).
`card_repo.repository` is also the repo `/update_stats` traverses and, since 2026-10-19, the repo
`/sync_emojis` lists icons from.

### commands/card_grid.py (2026-10-25; /query_ability's grid images)

| Constant | Value | Meaning |
|---|---|---|
| `COLUMNS`, `ROWS` (`PER_GRID`) | `4`, `5` (20) | cards per grid image, so per message; a row is as tall as its tallest card (2:3 or 1:2) |
| `CARD_W` | `300` | px per card; with `GAP` 8 a grid is 1240 px wide (about 1240 x 2500-3100, 1.2-1.6 MB JPEG) |
| `LABEL_H`, `FONT_SIZE` | `34`, `20` | the "n. name" line under each card (cut with an ellipsis) |
| `QUALITY` | `88` | grid JPEG quality, 4:4:4 so card text stays sharp |
| `CACHE_DIR` | `"card_images"` | gitignored: `<PNG blob>.jpg` per card at `CARD_W`; a picture is downloaded once per blob (cached only from a commit-pinned link; a branch link can lag on the CDN), stale blobs pruned after each grid answer |
| `POOL` | `4` | parallel picture downloads |

### commands/utils.py (lines 18-19)

| Constant | Value | Meaning |
|---|---|---|
| `BREAK_LEN` | `1950` | max chunk length for `split_long_message` (Discord hard limit is 2000; 50 chars of headroom preserves ``` code fences). Queue/splitting mechanics: autocroissant-architecture-contract |
| `ADMINS` | getattr, default `[]` | see config.py table above |

### commands/music_player.py (lines 26-67)

| Constant | Value | Meaning |
|---|---|---|
| `MUSIC_BASE_DIR` | `"music/"` | download/playback directory (gitignored; the dir is kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite .gitignore's `!music/.gitkeep` exception) |
| `SKIP_FILES` | `{".gitignore", ".DS_Store"}` | filenames ignored when traversing `music/` (defined music_player.py:30, used :171, :470) |
| `cookies_from_browser` | derived (line 37): `browser_cookies`, default False | see the browser_cookies section |
| `ydl_opts` | dict, music_player.py:50-58 | yt-dlp options: bestaudio, restricted filenames, `music/%(title)s-%(id)s.%(ext)s` template, `keepvideo` from KEEPVIDEO, Safari `cookiesfrombrowser` iff `cookies_from_browser`, quiet. `POSTPROCESS=True` appends mp3-320k extraction (61-67). Built ONCE at import — all its config fields are restart-required (the YoutubeDL objects themselves are made on first use since 2026-10-20) |

### commands/diffusion.py (lines 21-119)

> **AI left main on 2026-10-20:** this section applies to branch `archive/ai-generation` (main has no AI code).


| Constant | Value | Meaning |
|---|---|---|
| `MODELS_FOLDER` | `"./models/"` | where `.safetensors` checkpoints live (gitignored; the dirs are kept by the tracked `models/README` and `models/loras/README` stubs — no `.gitkeep` exists despite .gitignore's `!models/.gitkeep` exception) |
| `LORAS_FOLDER` | `"./models/loras/"` | where LoRA `.safetensors` live |
| `POSSIBLE_SCHEDULERS` | `["dpm++ sde", "dpm++ sde karras", "euler a"]` | whitelist enforced by `/set_scheduler` only (diffusion.py:427) |
| `SD15_RGB_FACTORS`, `SDXL_RGB_FACTORS(+_BIAS)`, `FLUX_RGB_FACTORS(+_BIAS)` | matrices, diffusion.py:36-119 | latent→RGB preview math cribbed from ComfyUI's latent_formats.py — not tunables, leave alone |

### commands/github_api.py (lines 12-18, added 2026-10-20; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `GIT_TOKEN` | getattr, default `""` | the one place the token is read (it used to be read in 4 modules); `config` is optional here, so `python -m commands.tts_save` runs without config.py |
| `HEADERS` | `{"Authorization": "token ..."}` or `{}` | auth for every GitHub REST call the bot makes |
| `RAW_MEDIA` | `"application/vnd.github.raw"` | Accept header for a blob's bytes |
| `RETRY_WAITS` | `(1, 2)` | seconds between the 3 tries of `get_with_retry` (connection error, 429 or 5xx) |

Also new 2026-10-20: `NET_TIMEOUT = 120` (commands/update_bot.py, seconds per git fetch or pull) and
`MAX_CONTENT = 2000` (main.py:831, Discord's message limit; longer content is split before sending).

### commands/type_emojis.py (lines 18-29, added 2026-10-19; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `EMOJI_SIZE` | `128` | emoji canvas in pixels (Discord shows emojis at 48 px or less; 128 stays sharp enlarged; the 256 KB upload limit is far away) |
| `MAX_NAME_LEN` | `32` | Discord's emoji-name limit; longer names are cut at a word boundary |
| `ICON_FOLDERS` | `("Types", "Types/Stars")` | the repo folders whose PNGs become emojis (direct children only) |
| `NO_SUFFIX` | `"_no"` | name suffix of the crossed-out copy made for every `Types/` icon (not stars): `[fire:no]` → `:fire_no:` |
| `NO_RED` | `(224, 27, 34, 255)` | the prohibition-sign red, the same as the font's `:no` (`#e01b22`) |
| `TOKEN_PATTERN` | `\[([^\[\]\n]{1,52})\]` | what `render_type_tokens` tries to replace; brackets that aren't a known emoji stay text |
| `type_emojis` | `{}` at import | module-level map emoji name → `<:name:id>`, filled by each sync. Empty until the startup sync finishes |

Name rules (lowercase; dots and apostrophes dropped; other non-alphanumeric runs → `_`; 32-char cut) are
in `emoji_name`. They must stay compatible with the parser's canonical token names — see
ttscardmaker-companion §1.6.

### commands/tts_save.py (Configuration block, added 2026-10-20; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `RULES_PATH` | `"TTS/decks.json"` (global_config.py since 2026-10-20) | deck rules (which repo PNGs each complete deck holds): the checkout's, else raw main |
| `HTTP_TIMEOUT` | `30` s | the website's cards.json and raw rules reads |

The bot's own run (`update_repo_save`, /update_tts_save, committing through git_commit) was retired 2026-10-25; tts_save is
the card-sync.yml CLI only.

### commands/git_commit.py (Configuration block, added 2026-10-21; not compiled; shared by card_bake and card_guards)

| Constant | Value | Meaning |
|---|---|---|
| `AUTHOR`, `GIT_IDENTITY` | AutoCroissant / autocroissant@gmail.com | the commit author and committer (API and SSH paths alike; as the PICKLE commits) |
| `COMMIT_TRIES` | `3` | how many times a caller redoes its step when the branch moved (409/422, or a rejected push) |
| `CONFLICT` | `(409, 422)` | ref update refused: not a fast forward |
| `NO_WRITE` | `(401, 403, 404)` | POST git/blobs refused: the token can't write (classic PATs get 404); `cant_write` remembers it until a restart |
| `GIT_TIMEOUT`, `API_TIMEOUT` | `300` s each | per git call in the history clone / per API call (a PSD blob is up to 85 MB, 113 MB as base64) |
| `cant_write` | in-memory, starts `False` | set on a NO_WRITE reply that is not a rate limit; every later commit goes over SSH |
| `token()` | `github_api.GIT_TOKEN`, else env `GITHUB_TOKEN` | the Actions have no config.py; `github_api.token_from_env()` patches the fetch headers the same way |
| `commit_files(...)` | | a rate-limited 403 raises `Transient` (the Action reports it; `ssh_on_limit` and `limited`, the bot's TTS save's, removed 2026-10-25) |

Removed 2026-10-20 with the post-/update_stats run (TTSCardMaker's Action owns the automatic path: tts-save.yml, card-sync.yml since 2026-10-24):
`IMAGES`, `_retry`, `_watched`, `save_affected`, `update_after_stats`.

### commands/font_check.py (Configuration block, added 2026-10-21; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `FONTS_DIR` | `"MDW/Fonts"` | the card repo's fonts folder; top level only (`stoogedings-src/` is the icon font's source) |
| `EXCLUDE_FOLDERS` | `("MDW", "Markers")` | PSDs under these path parts are not cards (card_data's list) |
| `ROLE_ID` | `"546427268751294475"` | the card makers' role the Discord message pings (`<@&id>`; `allowed_mentions` names that role only); the only role ping (card_check, card_guards and changelog post with `{"parse": []}`, owner 2026-10-23) |
| `BITE` | `"{name} has pushed a card without adding the {font} font to MDW/Fonts, bite this user."` | the user's wording, one line per (author, font) |
| `DISCORD_LIMIT` | `2000` | the webhook payload is cut to one message; the rest is in the commit comment |
| `NOBODY` | `"Someone"` | the author when no login is known (a checkout scan, the bot) |
| `API_TIMEOUT` | `60` s | per GitHub API call |
| `FILES_PAGE` | `300` | the compare and commit APIs list at most this many files; a bigger push, or a commit listing that many, is noted in the report |
| `COMMITS_PAGE` | `250` | the compare API lists at most this many commits; a bigger push is noted (later files may carry the wrong author) |
| `POOL_MIN`, `POOL_CHUNK` | `8`, `4` | `--all` parses this many PSDs or more in a process pool of half the cores, this many per task (813 PSDs: 98 s -> 9.8 s on the Mac, 2026-10-21); `--repo` reads blobs one by one (`/fonts` reads the card data since 2026-10-25: no download, no `CACHE_JSON`/`BATCH`) |

The Actions' token: `GITHUB_TOKEN` from the environment when config.py supplied none (`github_api.token_from_env()`, called by font_check's and card_bake's entry points since 2026-10-21).

### commands/card_check.py (Configuration block, added 2026-10-21; not compiled; CLI only)

| Constant | Value | Meaning |
|---|---|---|
| `LINE` | `"{name} has pushed {card} with: {notes}."` | the Discord line per card (card = `card_name(path)`, notes joined by "; ") |
| `CLOSING` | `"The bot's next /update_stats prints these too; the parser's own notes (no stars found, misspelt type, several creatures) also show on the website as Parser notes."` | the Discord message's last line (the validator's notes never reach the website) |
| `UNREADABLE` | `"could not be read"` | the error line's wording for a PSD psd-tools can't open or an edits file that isn't JSON |

Shared with font_check (imported): `DISCORD_LIMIT`, `NOBODY`, `API_TIMEOUT`, `EXCLUDE_FOLDERS`, `POOL_MIN`, `POOL_CHUNK`; the token
through `github_api.token_from_env()`. Repo paths from global_config: `TYPES_DIR`, `STARS_DIR`, `ICON_MANIFEST_PATH`, `EDITS_PATH`.
It runs from card-guards.yml's check job since 2026-10-25 (card-check.yml before; its `SPARSE_FILES` went: the guards'
`card_guards.SPARSE_FILES` holds its closure).

### commands/psd_details.py (Configuration block, added 2026-10-21; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `VERSION` | `1` | the details file's `"version"` |
| `TEXT_CHARS` | `200` | characters of text kept per type layer |
| `TREE_CHILDREN`, `TREE_DEPTH` | `80`, `8` | the compact layer tree's caps; past them one `MORE` entry (`["…", "more", true]`) |
| `DRAWN` | `("pixel", "smartobject", "shape")` | digit-named layers of these kinds are drawn bar cells (stat encoding `pixel`); exposure-style adjustments are `template` / `two-row` |

No CLI since 2026-10-25 (the one-time backfill is done). In the build the `supplied` flags come from
`psd_details.font_index()` (`PSD_TOOLS_FONT_DIRS`, else the checkout's MDW/Fonts).

### commands/card_runner.py (removed 2026-10-25)

Its `bake_lock` and the startup sweep of temp PSDs (`TEMP_PSD`, `TEMP_AGE`) went with the legacy /update_stats: the bot
parses no PSD.

The bake (card-sync.yml) takes the Action's token: `card_bake.run` calls `github_api.token_from_env()`, and
`git_commit.token()` prefers `GIT_TOKEN`, else `GITHUB_TOKEN`. The work dir is card_bake's (`RUNNER_TEMP` else a fresh `mkdtemp`). The fonts dir
(`card_bake._fonts_dir`): `PSD_TOOLS_FONT_DIRS`' first entry, else the checkout's `MDW/Fonts` (the `fonts_cache/` fallback
for the jetson went on 2026-10-25: nothing there bakes).

### commands/analytics.py — reminders (lines 28-81, added 2026-10-19; adapter block 2026-10-24)

The "Bot adapter" block at the top holds what differs from OJWikibot's copy of the module: `STORE_PATH`
(`REMIND_JSON`), `STORE_SCHEMA`, `TIME_SEMANTICS`, `ZONE_LABEL` ("Pacific time"),
`RUN_AS_CREATOR = True`, `ALLOW_POSITIONAL = True`, `ON_LOAD_ERROR = "off"` (an unreadable
store turns reminders off; OJWikibot exits), `CHANNEL_RETRY` and `is_admin`.

| Constant | Value | Meaning |
|---|---|---|
| `SCHEMA_VERSION`, `KEYS` (2026-10-24) | `2`; the 11 stored fields | reminder.json's format; a newer schema_version is refused |
| `NOT_SCHEDULABLE` | `{"reminders"}` | commands a reminder can't run (the panel needs a real user) |
| `PANEL_TIMEOUT`, `SELECT_SIZE` | `600` s; `25` | an idle panel closes; options per jump list (more: 23 per page plus earlier/later) |
| `REMINDER_VERSION` | `3` (2 until 2026-10-20) | stored as `v` on each reminder once migrated; reminders without `v` are read with the old command semantics, and any reminder below the current version is migrated on load (3 = retired options stripped) |
| `MAX_INTERVAL` | `timedelta(days=366)` | upper limit for `offset` and `frequency` (longer = typo, and huge values overflow datetime) |
| `CHANNEL_RETRY` | `timedelta(hours=1)` | how long a one-shot reminder whose channel is gone waits before trying again |
| `INTERVAL_UNITS` | s/m/h/d/w in seconds | the units `offset`/`frequency` accept; combinations like `1d12h` are allowed |
| `WEEKDAYS`, `TIME_FORMATS` | Monday-first names; `("%H:%M", "%I%p", "%I:%M%p")` | what `when` accepts (weekday prefixes need 3+ letters) |
| `TRUE_WORDS`, `FALSE_WORDS` | true/yes/y/on/1 and false/no/n/off/0 | how boolean command options are read (any case) |
| `DATE_PREFIX`, `WORD_PREFIX`, `INTERVAL_FULL`, `INTERVAL_PART`, `OPTION_NAME` | compiled regexes | the `when`, interval and `name:value` grammars |

None of these is a config.py field; changing them is a code change. Reminder behavior in practice:
autocroissant-run-and-operate §9.

### commands/card_sync.py (2026-10-22; not compiled; the bot's card data)

| Constant | Value | Meaning |
|---|---|---|
| `ACTIONS_TOKEN` | getattr, default `""` (card_sync.py:38) | the config.py field above |
| `WORKFLOW` | `"card-sync.yml"` | the card repo's Card sync Action that `force_update` / `cards:` dispatch |
| `POLL_SECONDS` | `15` | between two looks at a dispatched run's Card data job (the state is edited into the reply) |
| `WAIT_MINUTES` | `30` | a dispatched run's Card data job is followed this long (not its Bake and save) |
| `DATA_JOB`, `MAIN_JOB` (2026-10-25) | `"Card data"`, `"Bake and save"` | card-sync.yml's lane job names: the followed job, and the lanes the notes judge |
| `LANE_RUNS` (2026-10-25) | `5` | notes: the runs whose jobs are read for each lane's last outcome (one request a run) |
| `FIND_SECONDS` | `60` | a dispatched run must show up in the run list within this |
| `EDIT_MINUTES` | `14` | progress edits stop here (Discord's interaction lasts 15 minutes) |
| `REFRESH_SECONDS` (2026-10-25) | `120` | on use: a card command or autocomplete looks at the data branch at most this often (any sync counts) |
| `READERS` (2026-10-25) | 12 slash command names | the commands that read card data, so look on use (autocomplete always does) |

No clock in the bot since 2026-10-24: card-sync.yml starts itself on a card push (its `push` trigger). Since 2026-10-25
(owner: only main's writes wait; branch work/split) its data lane (job `data`, the data branch, posts, site) runs at once
and only its main lane (job `main`: the bake and the TTS save on main) waits 30 minutes after the first card push of a
burst (the environment `card-sync-quiet`'s wait timer, a TTSCardMaker repo setting). /update_stats only syncs, and since
2026-10-25 card commands sync on use (run-and-operate section 12), so no reminder is needed.

### the push guards: commands/card_guards.py, card_standards.py, png_export.py (2026-10-22/23; not compiled; card-guards.yml only)

| Constant | Value | Meaning |
|---|---|---|
| `card_guards.MODES` | `("dry", "report", "on")` | what the Action does; the repo variable `CARD_GUARDS` picks it (unset or off = the guards' steps skip; the card and font checks still run) |
| `card_guards.LINE` | `"{login} pushed {card}{changes} with these issues: {issues}. [{short}]({url})"` | the standards-channel line (no ping); `{changes}` is the card's semantic PSD diff (psd_diff) |
| `CHANGES_DISCORD`, `CHANGES_COMMENT` | `120`, `600` | characters of a card's changes on Discord and in the commit comment |
| `QUIET_MARGIN` | 60 s | `land` waits card_bake's `RECENT` plus this after the push |
| `ANCHOR_WINDOW` | 2 h | in a burst, PSD commits this close to the push may stand in for its PNG's PSD |
| `POOL_MIN`, `POOL_MAX` | `8`, `4` | from this many cards the checks run in a process pool: every core on a GitHub runner (`GITHUB_ACTIONS`, K-2 experiment 2026-10-25; revert: drop that branch of `pool_size`), else half the cores, at most POOL_MAX (each runs git); `--workers N` overrides |
| `TELL_PNG_MISMATCH` | `False` | R10 and "current but outside the bar" stay off Discord |
| `card_standards.AUTO_FIX` | `("R9",)` | what the bot fixes itself (the Txt2 story); everything else is told |
| `card_standards.LINE_IOU` | `0.9` | R8 (the Single-line Composer, always set and never told since 2026-10-25): a box text line inked under this share of the stored one rewraps, so the layer is redrawn and the guards re-export its PNG; no longer a bar to the switch |
| `png_export.TOL` | `2` | the difference a pixel may show and still be unchanged (ICC rounding) |
| `png_export.BESIDE_CUT` | `8` | a new PSD pushed beside its PNG counts as current within this many levels (owner, 2026-10-23) |
| `png_export.UNCHECKED` | "no stored image to cross-check the bot's PNG against: save the PSD with Maximize Compatibility on" | told for a PNG the guards made from a PSD with no stored composite (G2 rendered anyway, owner 2026-10-24, branch work/py314) |
| `png_export.KEPT` | "PNG kept as it is: the PSD has no stored image, and the bot's drawing differs from the PNG in only {px:,} px, ..." | told for a PNG kept beside a new PSD with no stored composite, at most `png_export.KEPT_PX` (300) px past BESIDE_CUT of the render, else redrawn (default 2026-10-24, owner to confirm; branch work/py314) |

### commands/card_preview.py (2026-10-23; not compiled; TTSCardMaker's card-preview.yml only; design section 16)

| Constant | Value | Meaning |
|---|---|---|
| `SPARSE_FILES` | 13 paths | the import closure card-preview.yml's preview job checks out of this repo (a test compares the two) |
| `MAX_EDIT` | 16384 bytes | the edit's JSON; past it: an input error |
| `WEBP_WIDTH`, `WEBP_QUALITY` | 720, 75 | preview.webp: the card view's width (the site's own thumbs are AVIF since 2026-10-24) |
| `RETRY` | card_bake's `TRANSIENT` + `blob` | refusal codes reported as `error`, `transient` (try again) |

### commands/card_preview_branch.py (2026-10-23; not compiled; standard library only; the publish job of card-preview.yml)

| Constant | Value | Meaning |
|---|---|---|
| `SPARSE_FILES` | 3 paths | its import closure (itself, repo_history, global_config): the publish job's checkout |
| `REQUEST` | `[A-Za-z0-9-]{8,64}` | the client's id: the folder `previews/<request>/` (the workflow's Validate step checks it too) |
| `MAX_CARD` | 200 characters | the card input (longer: an input error); result.json, the index row and the commit message keep it clipped |
| `BRANCH`, `DIR`, `INDEX` | `previews`, `previews`, `index.json` | where it writes on the card repo |
| `KEEP`, `DAYS` | 200, 7 days | more previews or an older one: the next publish is a fresh orphan commit... |
| `KEEP_AFTER`, `DAYS_AFTER` | 150, 6 days | ... keeping the newest this many under this age (so a rewrite comes about daily) |
| `TRIES` | 5 | publish pushes, each from the branch head then |
| `IDENTITY` | AutoCroissant | git_commit's `GIT_IDENTITY`, copied (that module imports requests; a test compares them) |

The workflow needs **no secret and no variable**: `GITHUB_TOKEN` reads the card's blobs by sha in the preview job
(`permissions: contents: read`, the workflow's) and pushes `previews` in the publish job (`contents: write`, that job
only); its pip install is the other card workflows' (card-sync.yml's, card-edits.yml's until 2026-10-24), fork pin and all. Nothing in config.py.

### psd_edit parity and card_bake limits (the bake and the guards)

| Constant | Value | Meaning |
|---|---|---|
| `PARITY_THRESH` | `24` | a pixel is bad past this difference |
| `PARITY_BAD_MAX` | `0.5` % | bad pixels a box may show against the official PNG; past it the miss is told (report, commit note), not refused |
| `PARITY_FLOOR` | `25.0` % | the only pixel refusal: a box past this is a broken render (owner, 2026-10-23) |
| `PARITY_MEAN_MAX` | `2.0` | mean difference over a box (owner, 2026-10-22); a reported measure since 2026-10-23 |
| `ADD_ABILITY_LIVE` | `False` | `+ability` (a first ability layer from the card kind's template) bakes only when True; off, the edit waits (f92fb5e; on once the owner approves the test PSDs) |
| env `PSD_DIRTY_VERIFY` | unset | `"1"`: the dirty-box gate also renders whole and raises on any difference (K-1; on in tests/conftest.py) |
| `card_bake.MAX_BAKES` | `40` | PSD edits per run; the rest wait for the next |
| `card_bake.RECENT` | 30 min | a PSD or PNG pushed more recently is its author's: the bake and the guards don't race them |
| `card_bake.BUDGET` | 40 min | no card starts later, so the commit lands within the job's 60 minutes |

## TTSCardMaker repo settings (names only; never write a value)

| Setting | Kind | Meaning |
|---|---|---|
| `CARD_GUARDS` | repository variable | the push guards' mode (`dry`, `report`, `on`); unset = off (2026-10-23) |
| `DISCORD_CHANGELOG_WEBHOOK` | secret | the changelog channel (changelog, bake/sync failure alerts; never pings) |
| `DISCORD_STANDARDS_WEBHOOK` | secret | the standards channel: card-check, the push guards (no ping) and font-check (pings the card makers' role) |
| `card-guards-quiet` | environment (planned 2026-10-24, branch work/py314; the owner creates it after both branches land) | its 30-minute wait timer is the push guards' quiet window: card-guards.yml's post job waits in it on a push's first attempt (no runner held); without the timer, Land sleeps the window out. Command: docs/Campaigns/PUSH_GUARDS.md section 20 |
| `card-sync-quiet` | environment (2026-10-24, branch work/clock; created by the session at landing: `wait_timer=30`, no branch policy) | its 30-minute wait timer is Card sync's push timer for the main lane (bake and save; since 2026-10-25 the data lane does not wait): card-sync.yml's `Quiet window` job waits in it on a push's first attempt (no runner held); a push while one waits skips its own main lane (its data lane runs). Without the timer, Quiet sleeps the window out |

`DISCORD_FONT_WEBHOOK` was deleted on 2026-10-23 (owner).

`card-preview.yml` (2026-10-23, website previews) reads none of these: no secret beyond `GITHUB_TOKEN`, no variable, no
Discord. Its branch `previews` is data, not a setting: it is rewritten by the workflow and never merged.

### The psd-tools fork pin: `requirements-cards.txt` (2026-10-23; every Action on 3.14 since 2026-10-24, branch work/py314)

The fork's tag is in one place: AutoCroissant `requirements-cards.txt`, the card code's packages
(`psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@<tag>`: read the tag there, no doc copies it;
fonttools, requests; pyoxipng left it on 2026-10-25: the Actions run the oxipng CLI, 10.2.1 by sha256, from the shared setup). `requirements.txt` (the bot)
starts with `-r requirements-cards.txt`, `requirements-dev.txt` with `-r requirements.txt`. The five card workflows on
Python 3.14 (six until card-edits.yml was deleted at the cutover, 2026-10-24), card-guards.yml (its check and post jobs), card-sync.yml (one venv for every stage),
card-preview.yml (without the oxipng CLI: its PNG is never published), and until 2026-10-25 card-check.yml and font-check.yml
(now card-guards.yml's check job), check the file out of AutoCroissant and install it offline from wheels in the Actions cache (path
`~/.cache/card-wheels`, key `cards-<hashFiles('bot/requirements-cards.txt')>-<OS>-<arch>-py<Python>`: one entry for all
of them; until 2026-10-24 card-edits, card-guards and card-preview on 3.12 and card-sync's own `-py<3.14.x>-py<3.12.x>`). The
first run after a change to the file or a new runner Python builds them (`pip wheel -r`: the fork's C extensions,
pyoxipng's Rust) and saves them; later runs skip the git clone and the builds (a cold miss on the Mac: 16.4 s for the
wheels, 19.5 s for pyoxipng's sdist, 10.7 s to install; a runner is slower, PUSH_GUARDS.md section 20). The
install goes by name (a copy of the file without the URL, `sed 's/ @ [^;]*//'`): pip builds a URL line from git again,
offline or not. The wheels freeze numpy, scipy, Pillow and the rest at that first build; any edit of the file refreshes
them (for a bad dependency release, delete the entry: step 1's `gh cache delete`).

Re-pin:
1. Push the new fork tag (never move an existing one, never pin a branch: the cache keeps a file's first wheels;
   delete the entry with `gh cache delete <key> -R MichaelJSr/TTSCardMaker` if one ever moved).
2. Change the tag in `requirements-cards.txt` (one line) and commit it to AutoCroissant main. The workflows need nothing:
   they check out main, and the new file is a new key (GitHub drops an entry unused for 7 days).
3. Run the tests: `tests/test_card_data.py -k requirements` (one pin, a `cards-YYYYMMDD[letter]` tag, the layers) and,
   against a TTSCardMaker checkout, `CARD_GUARDS_YML=<checkout>/.github/workflows/card-guards.yml AC_SITE_SRC=<checkout>`
   with `tests/test_card_guards.py tests/test_card_preview.py -k workflow` and `tests/test_card_bake.py -k "fork or 314"`
   (`test_card_bake.fork_install` checks each of the six workflows' install: no pin of its own, Python 3.14, the file's
   wheels, one cache key; `test_every_action_runs_python_314` every workflow's Python).
4. The jetson's venv and the dev Mac's: `pip install --force-reinstall --no-deps "$(grep '^psd-tools' requirements-cards.txt)"`
   (a plain `pip install -r` keeps the old commit while the fork's version is unchanged: 1.25.0 up to `cards-20261023d`,
   1.26.0+cards since `cards-20261024`, also at `cards-20261024b`, `cards-20261025`, `cards-20261025b` and `cards-20261026`; autocroissant-run-and-operate, "When the psd-tools fork tag moves"; autocroissant-build-and-env).

A card-sync run whose builder (`builder_ref`, `CARD_DATA_BUILDER_REF`) is an AutoCroissant commit from before
2026-10-23's requirements-cards.txt fails at its "Python environment" step: that commit has no such file.

### setup.py (lines 8-49) — Cython build configuration

| Constant | Meaning |
|---|---|
| `EXCLUDE_FILES` | 8 entries never compiled, each with an in-file reason comment (config files, main.py entry point, update_bot.py's execv/execl, utils.py dynamic behavior, `__init__`s) |
| `INCLUDE_FILES` | the 11 command modules that DO get compiled (diffusion.py left with the AI code on 2026-10-20; type_emojis.py and the other modules in neither list are never compiled) |
| `COMPILER_DIRECTIVES` | `boundscheck: True`, `wraparound: True` (since 9919a14, 2026-10-23, owner: a bad or negative index behaves as in Python; the `x[len(x)-1]` idiom stays), `cdivision: True`, etc. |

Why these are set this way, the build/clean commands, and the stale-`.so`-shadows-your-`.py`
trap: **autocroissant-build-and-env**.

## Runtime-settable vs restart-required

THE classic confusion in this bot. Two facts, verified by reading every setter:

1. The two `/set_*` commands on main assign to `card_repo` attributes in memory. **Neither
   writes config.py** — no file write exists in either. Changes evaporate on restart. And
   restarts are routine: every `/update` restarts the bot,
   silently resetting anything set at runtime.
2. `config.py` is the ONLY persistence, and only for the fields it has (tokens/ADMINS/music
   flags). `match_ratio` and `repository` are NOT config.py fields at all — they reset to their
   hardcoded defaults on every restart, and there is no way to persist a change short of editing
   `commands/query_card.py`.

| Slash command | Mutates (verified location) | After restart becomes |
|---|---|---|
| `/set_ratio` | `card_repo.match_ratio` (`set_match_ratio`, query_card.py:945-963; validates 0.0-1.0) | `0.6` (`DEFAULT_MATCH_RATIO`) — always |
| `/set_repo` | `card_repo.repository` + re-runs `populate_files` (`set_repository`, query_card.py:965-988) | `"MichaelJSr/TTSCardMaker"` — always |

Restart-required only (no runtime setter exists): `TOKEN`, `ADMINS`, `GIT_TOKEN`, `ACTIONS_TOKEN`,
`browser_cookies`, `KEEPVIDEO`, `POSTPROCESS` — edit config.py on that machine, then restart.

> Branch `archive/ai-generation` only: `/set_scheduler`, `/set_model`, `/set_lora` and `/set_device`
> set the diffusion globals `SCHEDULER_NAME`, `MODEL`, `LORA`, `DEVICE_NO` (reset to the config.py
> values on restart). Each calls `_clear_pipelines()`, so the next generation re-runs `init_pipeline()`.
> Details: autocroissant-ai-boundary.

Each `/set_*` command called with no argument is the read path: it replies with the current
value without changing anything — the cheapest way to inspect live
state.

The `/set_repo` handler is `slash_set_repo` (it shared the name `slash_set_ratio` until 2026-10-20, C5).
Since 2026-10-20 `/set_repo` switches only once the new tree loads; a failed fetch keeps the old repository and card list and says so.

## Defaults-mismatch traps

1. **`update_stats` force_update — resolved 2026-10-19.** The Python default used to be `True` while
   the slash wrapper passed `False`, so a bare programmatic call force-reparsed every non-excluded
   PSD (story: autocroissant-failure-archaeology Entry 10.4). Both are now `False`
   (card_sync.update, main.py). Code that wants a forced re-parse must pass
   `force_update=True` explicitly.
2. **`use_local_repo` / `use_local_timestamp` — removed 2026-10-20.** `use_local_repo` defaulted to
   True (local mode) until 2026-10-19 and to False for one day; `use_local_timestamp` defaulted to True,
   which made local mode compare file mtimes with stored GitHub commit dates and re-archive every card
   (docs/BUGS_AND_IMPROVEMENTS.md B6). On 2026-10-20 local mode was archived: both options are gone from the
   slash command and from `update_stats()`, the traversal lives on branch `archive/local-update-stats`
   (not in main, not compiled), and stored reminders naming them fail binding (the dropping shim was removed 2026-10-25).
   (`force_update` now dispatches a full Card sync re-parse: an admin's decision, not a config probe.)
3. **`vram_usage` is branch-only:** main reads it nowhere since 2026-10-26 (the browser_cookies fallback went);
   on branch `archive/ai-generation` it is the diffusion memory mode (default `"low"`).
4. **Unvalidated config.py values (branch `archive/ai-generation`):** neither `vram_usage` nor
   `scheduler_name` is checked against a whitelist at init — typos silently degrade (fallthrough
   memory behavior; default scheduler). Only the `/set_*` paths validate.

## How to add a new config field (checklist)

Story behind the rules: every field except TOKEN is getattr-defaulted precisely because there
are two machines with two different hand-maintained config.py files — a field that hard-fails
when missing takes the bot down on the OTHER machine at its next startup.

1. Choose the consuming module and add one line to its `# Configuration` block:
   `MY_FIELD = getattr(config, "my_field", <safe default>)`. Never `config.my_field` bare, never
   `from config import my_field` — only TOKEN does that, deliberately fatal.
2. Pick the default so that a machine WITHOUT the field behaves exactly as before the change.
   Test mentally against both machines: the dev Mac (Safari) and the jetson (production).
3. Add the field by hand to `config.py` on the machine(s) that need a non-default value. Keep
   mode 0700. NEVER commit it — confirm with `git check-ignore -v config.py` and never weaken
   .gitignore.
4. If it changes user-visible behavior, document it in `commands/help.py`'s dicts and the
   `@tree.command(description=...)` / `@app_commands.describe` text (templates:
   **autocroissant-docs-and-style**).
5. If it's AI-related: the AI code lives only on branch `archive/ai-generation`. There, consume it
   ONLY in `commands/diffusion.py` behind getattr, keep torch imports lazy, and verify core startup
   is untouched (rules: **autocroissant-ai-boundary**).
6. If it needs live toggling, add a `/set_*` command — and accept that the runtime change won't
   persist; config.py stays the only persistence. Say so in the command's help text.
7. Update this skill: add the field to the "fields main reads" table (making it 9), and extend
   the provenance greps. Route the code change itself through **autocroissant-change-control**.

## When NOT to use this skill

- Setting up an environment from scratch, installing requirements, Cython build/clean mechanics
  → **autocroissant-build-and-env**.
- What a flag's SUBSYSTEM actually does beyond the flag (diffusion pipeline internals →
  **autocroissant-ai-boundary** (branch `archive/ai-generation`); card parsing/validation semantics →
  **impossibility-cards-reference**; queue/dispatch machinery →
  **autocroissant-architecture-contract**).
- Starting/operating the bot, reminders in practice, the /pull//update flow →
  **autocroissant-run-and-operate**.
- Diagnosing a misbehaving flag at runtime → **autocroissant-debugging-playbook**.
- The history of why a default changed → **autocroissant-failure-archaeology**.

## Provenance and maintenance

Everything above was verified 2026-07-11 by the commands below; the rows touched on 2026-10-19
and 2026-10-20 were re-run those days. Flags drift: regenerate each
table with its command before quoting it in the future.

| Fact / table | Re-verification command (cwd = repo root) |
|---|---|
| Master getattr map (all config.py consumers + defaults) | `grep -rn "getattr(config" --include="*.py" .` |
| TOKEN is the only hard import; run site | `grep -rn "from config import" --include="*.py" .` and `grep -n "client.run" main.py` |
| config.py field names (NAMES ONLY — never print values) | `python3 -c "import config; print(sorted(k for k in dir(config) if not k.startswith('_')))"` |
| config.py mode 0700 + gitignored | `ls -l config.py` and `git check-ignore -v config.py` |
| lacks_admin gate list + the ungated /restart_bot | `grep -n "lacks_admin" main.py` then read each hit's enclosing `@tree.command`; `grep -n -A2 'name="restart_bot"' main.py` |
| lacks_admin polarity | `grep -n -A2 "def lacks_admin" commands/utils.py` |
| card-module constants (psd_analyzer split 2026-10-20; moved 2026-10-22) | `grep -n "^UPDATE_RATE\|^EXCLUDE_FOLDERS\|_WORKERS =" commands/card_data.py`; build constants: `grep -n "^BUDGET\|^RATE_FLOOR\|^MIN_CARDS\|^ARCHIVE_\|^REPARSE_MAX\|^QUALITY_\|^DETAILS_SHARE" commands/card_data.py`; `grep -n "TYPE_REGION_RATIO\|MISSPELT_CARD_TYPES\|^ICON_\|^EFFECT_\|STAR_CORE" commands/psd_parser.py`; `grep -n "EXPORTED_\|COLUMN_ORDER" commands/psd_analyzer.py`; repo paths: `grep -n "CARD_\|_BASE\|_DIR\|_PATH" global_config.py` |
| Exclusion set size/location; PARSER_VERSION | `grep -n "ABILITY_EXCLUSIONS\|^PARSER_VERSION" commands/psd_parser.py` |
| UPDATE_RATE cadence usage | `grep -n "% UPDATE_RATE" commands/card_data.py` |
| query_card defaults | `grep -n "DEFAULT_REPOSITORY\|DEFAULT_MATCH_RATIO" commands/query_card.py` |
| BREAK_LEN | `grep -n "BREAK_LEN = " commands/utils.py` |
| music constants + browser_cookies + ydl_opts | `grep -n "MUSIC_BASE_DIR\|SKIP_FILES\|browser_cookies\|cookiesfrombrowser\|keepvideo\|postprocessors\|preferredquality" commands/music_player.py` |
| No AI code on main | `ls commands/diffusion.py` fails; `grep -n "set_model\|set_lora\|set_scheduler\|set_device" main.py` is empty |
| Branch only: diffusion folders/schedulers | `git grep -n "MODELS_FOLDER\|LORAS_FOLDER\|POSSIBLE_SCHEDULERS" archive/ai-generation -- commands/diffusion.py` |
| Branch only: vram modes, model='' disables init, HF_TOKEN Flux-only, setters → lazy re-init | `git grep -n "device_map\|float32\|if not MODEL\|HF_TOKEN\|_clear_pipelines\|if not txt2img_pipe" archive/ai-generation -- commands/diffusion.py` |
| GIT_TOKEN consumers/rate-limit prints | `grep -rn "GIT_TOKEN" commands/ main.py` (defined once in github_api.py) |
| tts_save constants | `grep -n "^[A-Z_]* = " commands/tts_save.py` |
| git_commit constants and the two callers | `grep -n "^[A-Z_]* = \|^cant_write" commands/git_commit.py`; `grep -n "commit_files(" commands/card_bake.py commands/card_guards.py` |
| font_check constants (no alias table since 2026-10-23) | `grep -n "^[A-Z_]* = " commands/font_check.py`; `grep -rn "FONT_ALIASES" --include="*.py" .` (no output) |
| card_check constants and its sparse list | `grep -n "^[A-Z_]* = " commands/card_check.py`; `grep -n -A1 "^SPARSE_FILES" commands/card_check.py` |
| card_preview constants and its sparse list (2026-10-23) | `grep -n "^[A-Z_]* = \|^[A-Z_]*, [A-Z_]* = " commands/card_preview.py commands/card_preview_branch.py`; `grep -n "secrets\.\|vars\." <TTSCardMaker>/.github/workflows/card-preview.yml` (only `secrets.GITHUB_TOKEN`) |
| No card_runner, repo_traverser or CARD_SYNC (2026-10-25) | `ls commands/card_runner.py commands/repo_traverser.py` (absent); `grep -rn "CARD_SYNC" global_config.py main.py commands/ \| grep -v changelog` (nothing) |
| global_config env overrides (two: LOCAL_DIR_LOC, HISTORY_CLONE) | `grep -n "environ.get" global_config.py`; `grep -rn "AUTOCROISSANT_" --include="*.py" commands/ global_config.py` (global_config.py only) |
| Module environment reads | `grep -rn "environ.get" --include="*.py" commands/` (FONT_DIRS_ENV, WORK_ENV = RUNNER_TEMP, GITHUB_TOKEN, BUILDER_SHA, DISCORD_ARCHIVE, DISCORD_MATCH_REPO, SSH_AUTH_SOCK) |
| ACTIONS_TOKEN and the card_sync knobs | `sed -n '38,56p' commands/card_sync.py` |
| Push-guard, parity and bake knobs | `grep -n "^MODES\|^LINE\|^CHANGES_\|^QUIET_MARGIN\|^ANCHOR_WINDOW\|^POOL_\|^TELL_PNG" commands/card_guards.py`; `grep -n "^AUTO_FIX" commands/card_standards.py`; `grep -n "^TOL\|^BESIDE_CUT" commands/png_export.py`; `grep -n "^PARITY_\|^ADD_ABILITY_LIVE\|^DIRTY_VERIFY" commands/psd_edit.py`; `grep -n "^MAX_BAKES\|^RECENT\|^BUDGET" commands/card_bake.py` |
| Setters never write config.py | `sed -n '954,997p' commands/query_card.py`; confirm zero `open(`/`write` there |
| update_stats defaults (force_update False both places; use_local_repo / use_local_timestamp removed 2026-10-20) | `grep -n "force_update: \|use_local" main.py commands/card_sync.py commands/card_data.py` (no use_local hits; every force_update default is False) |
| Icon-font constants | `grep -n "^ICON_" commands/psd_parser.py global_config.py` |
| type_emojis constants | `sed -n '14,26p' commands/type_emojis.py` |
| Reminder constants | `sed -n '28,81p' commands/analytics.py` |
| Bot identity (names only) | `grep -n "^#" config.py` shows which identity's TOKEN line is active (comments only — never print values) |
| global_config constants | `cat global_config.py` (45 lines) |
| setup.py build config | `sed -n '8,49p' setup.py` |
| Gitignore coverage (config.py, reminder.json, music/, *.csv, *.txt) | `grep -n "config.py\|reminder\|music/\|csv\|txt" .gitignore` |

When any command's output disagrees with this file, the repo wins: update the table, re-date the
"as of" stamps, and note the change here.

2026-10-21: query_card.py and main.py line numbers re-pointed after the atomic-save and 0.1 s tick changes (no flag or default changed; review: client.run, MAX_CONTENT, /set_ratio).

2026-10-21 (later): `FONT_ALIASES` moved from commands/psd_edit.py:47 to global_config.py:26 (32 lines now) and commands/font_check.py's
constants catalogued; no flag or default changed. Review the same day: main.py citations re-pointed after the /fonts block (:278-285, lazy
font_check import; `fonttools` added to requirements.txt).

2026-10-23: `FONT_ALIASES` deleted (owner: "change the fonts to the exact ones and delete the aliases"): `Papyrus-Regular` is
MDW/Fonts/Papyrus.ttf's own PostScript name and Kung Fu Panda Regular.ttf was renamed `Kung-Fu-Panda` (TTSCardMaker work/kfp-name
be8c96ff); font_check matches PostScript names exactly, the fork's FontMap gets no aliases.

2026-10-21 (audit fixes): the commit path of tts_save.py and card_bake.py folded into commands/git_commit.py (its table above; `GIT_TOKEN`
row and the tts_save table re-pointed); font_check gained `COMMITS_PAGE`, `POOL_MIN`, `POOL_CHUNK` and takes the Action token through
`github_api.token_from_env()`. No flag or default changed.

2026-10-21 (review): git_commit gained `limited` (a rate limit keeps one tts_save run on SSH, as the old `via_git` did); its table row and grep.

2026-10-21 (titles): `PARSER_VERSION` row added (2); `EXCESSIVE_STAT_EXCLUSIONS` row removed (deleted; `allow_high_stats` in Card Edits.json replaces it); psd_parser.py and repo_traverser.py citations re-pointed.

2026-10-21 (review): psd_analyzer.py citations re-pointed (+2 for the PARSER_VERSION/title_text imports; `EXPORTED_*`, `COLUMN_ORDER`), repo_traverser.py `UPDATE_RATE` check :391, psd_parser.py `ABILITY_EXCLUSIONS` :950-969 after the WX-78 title fix. Gate re-run: `EXPORTED_STATS_NAME` uses are :398, :405.

2026-10-21 (S7): commands/card_runner.py catalogued (bake_lock, _last_edits_sha, REPORT_FILE, the dry-run hold line, /bake_cards defaults); the lacks_admin table re-derived (13 gates); main.py, repo_traverser.py and help.py citations re-pointed. No flag added: the hold is a literal in the code.

2026-10-21 (hook removed): `_last_edits_sha` and the dry-run hold line are gone from card_runner.py (the user: the Action is the one live runner; /bake_cards is the manual fallback and nothing in the bot queues a bake); the card_runner table and its re-verification row updated. No cited line moved.

2026-10-21 (details): `DETAILS_JSON` row; commands/psd_details.py's constants catalogued; no flag or default changed. repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-21 (card check): commands/card_check.py's constants catalogued (`SPARSE_FILES`, `LINE`, `CLOSING`, `UNREADABLE`; the rest imported from font_check); no flag or default changed. Same day: `CLOSING` reworded (the validator's notes never reach the website's Parser notes).

2026-10-22 (data build): the traverser's constants (`UPDATE_RATE`, `DOWNLOAD_WORKERS`, `LOOKUP_WORKERS`, `COMMITS_PAGE`,
`EXCLUDE_FOLDERS`) and body moved to the uncompiled commands/card_data.py; citations and the re-verification greps
re-pointed there (repo_traverser.py keeps `update_stats`, :59, and the ETag/file caches). The build's own constants
catalogued (`BUDGET` 400, `RATE_FLOOR`, the gate thresholds, `SPARSE_FILES`, the exit codes, changelog `MESSAGES_MAX`).
2026-10-22 (admin gates): the ADMINS gate table re-derived (20 call sites): /purge, /update_metadata, /mass_replace, /alias, /set_repo, /delete_song and /delete_all_music are admin only (owner decision).

2026-10-23 (docs drift): the 8th config field `ACTIONS_TOKEN` (card_sync.py:34) and the trigger words ACTIONS_TOKEN,
CARD_SYNC, CARD_GUARDS; config is imported by utils and music_player, optionally by github_api and card_sync;
global_config is 45 lines with three env overrides (BOT_CLONE) plus the modules' own env reads; rows for BOT_CLONE,
FONTS_CACHE, FONT_DIRS_ENV, CREATED_PATHS, CARD_SYNC, DATA_BRANCH, CARD_DATA_DIR; the HISTORY_CLONE row (log --raw, no
lazy fetch, one .mailmap blob, full clone); ICON_VARIANTS has six values on main; setup.py: 11 modules, bounds checks and
wraparound on; card_runner's TEMP_PSD/TEMP_AGE and the fonts fallback (fonts_cache); new tables for card_sync, the push
guards, psd_edit parity and card_bake, and the TTSCardMaker repo settings; 21 lacks_admin sites (the conditional
/update_stats gate); every main.py, query_card.py, card_data.py, psd_parser.py and psd_analyzer.py citation re-pointed.

2026-10-23 (unbatch): query_card.py citations re-pointed (card_repo, set_match_ratio, set_repository) after /query_ability went back to one message per card.

2026-10-24 (py314, branch work/py314): the fork pin section (six workflows on 3.14, one cache entry, the pyoxipng
marker), the `card-guards-quiet` environment among the repo settings, `png_export.UNCHECKED`. The guards take no fonts
(R7 dropped; no `--fonts`).
2026-10-25 (refresh on use, work/refresh): card_sync's REFRESH_SECONDS and READERS; card_sync.py citations re-pointed (ACTIONS_TOKEN at :38).

2026-10-25 (retire): five Discord edit commands retired (branch archive/discord-edits); the lacks_admin table re-derived (16 gates); card_runner's REPORT_FILE and /bake_cards defaults gone; analytics gained `RETIRED_COMMANDS`; git_commit's `limited`/`ssh_on_limit` (the bot's TTS save only) removed.

2026-10-25 (step 10, work/pickles): `ALIAS_PKL`, `STATS_PKL`, `OLD_STATS_PKL`, `DETAILS_JSON`, `CARD_SYNC`, card_data's `LOOKUP_WORKERS`/`COMMITS_PAGE`, font_check's `CACHE_JSON`/`BATCH` and the card_runner and repo_traverser tables removed; `BOT_CLONE` now serves the build's seed only; /push's gate row gone (15 gates); psd_details' CLI needs `--clone` and `--out`.

2026-10-25 (clean, work/clean): BOT_CLONE, REMIND_PKL, FONTS_CACHE and card_check.SPARSE_FILES removed; the *_PATHS tuples collapsed to the single paths; the card and font checks run in card-guards.yml's check job.

2026-10-25 (grid): commands/card_grid.py's constants.
2026-10-25 (cardlist, work/cardlist): `CARD_TREE_JSON`.
2026-10-25 (wrap-up): `PARSER_VERSION` is 4 (3: a PSD's pictures, d6d2f5e; 4: the census fixes, 1b17aba); card_data.py citations re-pointed.
2026-10-25 (wrap-up review): `ADD_ABILITY_LIVE` and env `PSD_DIRTY_VERIFY` in the psd_edit table.
