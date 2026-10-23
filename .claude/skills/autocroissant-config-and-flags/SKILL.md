---
name: autocroissant-config-and-flags
description: Catalog of EVERY configuration axis in the AutoCroissant Discord bot. Load for any question containing "config", "flag", "setting", "default", "where is X configured", "how do I change/set X", or "why did my setting reset" — and for any mention of TOKEN, ADMINS, GIT_TOKEN, browser_cookies, KEEPVIDEO, POSTPROCESS, SITE_URL, match ratio, /set_ratio, /set_repo, admin-only commands, rate limits, BREAK_LEN, UPDATE_RATE, DOWNLOAD_WORKERS, TYPE_REGION_RATIO, EXCLUDE_FOLDERS, MUSIC_BASE_DIR, LOCAL_DIR_LOC, HISTORY_CLONE, TTS_SAVE_PATH, AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE, TIMEZONE/PST, pickle filenames, ICON_FONT_PREFIX, ICON_MANIFEST_PATH, FONT_ALIASES, ICON_VARIANTS, ICON_SCAN_PATTERN, EFFECT_LAYER_TAG, EFFECT_COVERAGE_MIN, AUTHORS_PATH(S), EDITS_PATH(S), STAR_CORE_ALPHA, the type_emojis constants (EMOJI_SIZE, MAX_NAME_LEN, ICON_FOLDERS, NO_SUFFIX, NO_RED, TOKEN_PATTERN), the reminder constants (REMINDER_VERSION, MAX_INTERVAL, CHANNEL_RETRY, LEGACY_RUNS_AS_BOT, RETIRED_OPTIONS, TIME_FORMATS, WEEKDAYS), or the AI fields and setters that live only on branch archive/ai-generation (HF_TOKEN, model, lora, device_no, scheduler_name, vram_usage, /set_scheduler, /set_model, /set_lora, /set_device). Contains the config.py field table (names only), global_config.py and per-module constants, prod-vs-experimental status, runtime-settable vs restart-required (/set_* do NOT persist), update_stats defaults, an add-a-config-field checklist and re-verification greps.
---

# AutoCroissant: Configuration and Flags

All facts verified against the working tree on 2026-07-11; re-verified 2026-10-19 against HEAD a7a4bb3 plus
the uncommitted working tree (new: icon-font constants in psd_analyzer, commands/type_emojis.py, the
reminder constants in analytics.py, `force_update` default); re-verified 2026-10-20 against HEAD 9fb3093
(AI code only on branch `archive/ai-generation`, `browser_cookies`, `SITE_URL`). Line numbers and defaults DRIFT —
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
- The file must EXIST for the bot (and for any script importing `commands.*`) to even import:
  every command module does `import config`. Minimum viable `config.py` = a single `TOKEN = ...`
  line; every other field falls back to its default.
- Side effect worth knowing: main.py prints one token-PRESENCE line at import
  ("Git token found, API limited to 5000 requests/hour." or the 60/hour variant,
  `print_token_status` in commands/github_api.py). Presence only — values are never printed.

### The 7 fields main reads (as of 2026-10-20)

| Field | Required? | Default (getattr) | Read at | Status | Effect |
|---|---|---|---|---|---|
| `TOKEN` | YES — bot won't boot without it | none — the ONLY hard import: `from config import TOKEN` (main.py:12); used by `client.run(TOKEN)` (main.py:920) | main.py | prod | Discord bot token. It decides the bot IDENTITY, and since 2026-10-19 that matters beyond login: application emojis belong to the identity (autocroissant-run-and-operate §11). On the dev Mac (2026-10-19) the active line is the "Waxwell" identity and the AutoCroissant token line is commented out |
| `ADMINS` | no | `[]` (commands/utils.py:19) | utils.py | prod | list of Discord user IDs (ints) allowed past `lacks_admin` — see gate table below |
| `GIT_TOKEN` | no (strongly recommended) | `""` (commands/github_api.py — the one read since 2026-10-20; query_card, card_data (the traverser's code since 2026-10-22) and type_emojis import it from there; git_commit.token() reads it, else the Actions' GITHUB_TOKEN, for tts_save and card_bake since 2026-10-21) | github_api | prod | GitHub API auth: raises the rate limit 60 → 5000 req/hr. Used for the Authorization headers, the PyGithub client for commit timestamps, and the emoji sync's tree listing (1 call per startup or /sync_emojis). Since 2026-10-20 it also WRITES: `/update_tts_save` commits the TTS save to the card repo with it, so it needs write access there (when the token can't write — 401/403/404 on the blob upload, or no token — it commits through the history clone and pushes over SSH with the service's agent instead) |
| `browser_cookies` (2026-10-20) | no | absent → falls back to `vram_usage == "mps"` (music_player.py:37) | music_player | prod | True = yt-dlp uses Safari cookies — see the browser_cookies section below |
| `vram_usage` | no | `""` in the fallback (music_player.py:37) | music_player | legacy fallback | on main it matters ONLY when `browser_cookies` is absent (old configs). Its AI meaning is branch-only, below |
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
`cookies_from_browser: bool = bool(getattr(config, "browser_cookies", getattr(config, "vram_usage", "") == "mps"))`;
when true, yt-dlp gets `"cookiesfrombrowser": ("safari", None, None, None)` (music_player.py:55). Set
`browser_cookies = True` only on a machine with Safari (the Mac). A config.py without `browser_cookies`
falls back to the old side channel `vram_usage == "mps"` (default `""`), so old Mac configs keep their
cookies. This replaced the vram_usage-only hack on 2026-10-20 (docs/BUGS_AND_IMPROVEMENTS.md S1).

### ADMINS — which commands it gates

`lacks_admin(interaction)` (commands/utils.py; named `perms_check` until 2026-10-20) returns **True when the
user is NOT an admin** (`interaction.user.id not in ADMINS`). Every call site reads
`if lacks_admin(...): send "You do not have permission"; return`. The invariant is owned by
**autocroissant-architecture-contract**.

Commands gated by `lacks_admin` in main.py (all 20 call sites, verified 2026-10-22 after the owner's admin-only batch; line numbers
drift — re-derive with `grep -n "lacks_admin" main.py`):

| Slash command | main.py line |
|---|---|
| `/stop_bot` | 180 |
| `/pull` | 195 |
| `/push` | 205 |
| `/update` | 218 |
| `/set_reminder` | 248 |
| `/set_repo` (added 2026-10-22) | 354 |
| `/alias` (added 2026-10-22) | 369 |
| `/update_tts_save` (added 2026-10-20) | 414 |
| `/bake_cards` (added 2026-10-21) | 430 |
| `/update_metadata` (added 2026-10-22) | 491 |
| `/mass_replace` (added 2026-10-22) | 546 |
| `/delete_song` (added 2026-10-22) | 681 |
| `/delete_all_music` (added 2026-10-22) | 690 |
| `/purge` (added 2026-10-22) | 710 |
| `/list_guilds` | 720 |
| `/leave_guild` | 735 |
| `/sync_global` | 744 |
| `/sync_emojis` (added 2026-10-19) | 759 |
| `/list_guild_channels` | 775 |
| `/get_channel_messages` | 795 |

The 2026-10-22 seven are gated whole, view modes included (`/set_repo` and `/alias` with no arguments).
`/purge` cannot be scheduled by a reminder: its `user` option has no default. Tests: `tests/test_admin_gates.py`.

`/list_guild_members` was removed 2026-10-20 (C6: it needed the privileged members intent, which the bot
does not enable).

Two more ADMINS uses outside main.py since 2026-10-19 (commands/analytics.py):
- `/remove_reminder` lets admins remove any reminder and everyone else only their own.
- A firing reminder runs as its creator, so every gate above is re-checked against the creator at fire time.
  Reminders saved before 2026-10-19 have no creator and still run as the bot (`LEGACY_RUNS_AS_BOT`); the bot's
  own id is in the dev Mac's ADMINS. Mechanism: autocroissant-architecture-contract invariant 9.

Quirk, stated plainly: `/restart_bot` (main.py:152-159) is NOT gated — anyone in the guild can
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

Whole file, 32 lines (verify: `cat global_config.py`; `FONT_ALIASES` added 2026-10-21). Two constants read the environment first (2026-10-20,
`environ.get(name, default)`), so TTSCardMaker's `tts-save.yml` Action can run `python -m commands.tts_save` on
its checkout without a symlink. No other configuration comes from the environment (tts_save.py only inspects
`SSH_AUTH_SOCK` to word its SSH-push error; tts_save.py and repo_history.py pass the environment through to git with
`GIT_TERMINAL_PROMPT=0`; verify: `grep -rn environ --include="*.py" commands/ main.py global_config.py`):

| Constant | Value | Meaning |
|---|---|---|
| `LOCAL_DIR_LOC` | `"~/Desktop/TTSCardMaker"`, env `AUTOCROISSANT_LOCAL_DIR` | the card repo checkout. `/update_stats` no longer reads it (local mode was archived 2026-10-20); commands/tts_save.py reads `TTS/decks.json` and `site/Card Edits.json` under it in standalone runs (the Action sets it to `$GITHUB_WORKSPACE`); the diagnostics scripts (parse_one.py, gap_trace.py) and the archived traversal on branch `archive/local-update-stats` expand it with `expanduser`. This path convention is load-bearing across the toolchain |
| `ALIAS_PKL` | `"aliases.pkl"` | card-alias pickle (committed to git) |
| `STATS_PKL` | `"stats.pkl"` | the card stats database pickle (committed) |
| `SITE_URL` (2026-10-20) | `"https://michaeljsr.github.io/TTSCardMaker"` | the card browser: `/website`, `/query` Card page buttons, search links, the site's `data/cards.json` (main.py, query_card.py) |
| `OLD_STATS_PKL` | `"old_stats.pkl"` | archived old card versions (committed) |
| `REMIND_PKL` | `"reminder.pkl"` | reminders pickle (GITIGNORED, unlike the other three) |
| `DETAILS_JSON` (2026-10-21) | `"psd_details.json"` | the website's per-card PSD facts by blob sha (commands/psd_details.py); committed and pushed with the pickles by /push, kept local on a /pull conflict; its `.tmp` gitignored |
| `HISTORY_JSON` (2026-10-20) | `"repo_history.json"` | repo path -> (blob, commit that last changed it), written by commands/repo_history.py; gitignored, rebuilt from the clone |
| `EMOJI_JSON` (2026-10-20) | `"type_emojis.json"` | bot application id -> {type emoji name -> blob sha of the icon it was made from} (emojis belong to the bot identity), written by commands/type_emojis.py; gitignored, per machine. Missing, unreadable or no entry for this bot = its existing emojis count as current (no re-upload) |
| `HISTORY_CLONE` (2026-10-20) | `"~/Documents/TTSCardMaker-history.git"`, env `AUTOCROISSANT_HISTORY_CLONE` | bare blobless clone of the card repo, fetched on startup and on every /update_stats; cloned on first use (the Action sets `~/TTSCardMaker-history.git` on its runner). Only rev-list/ls-tree/merge-base/fetch run there (diff/log would download blobs) |
| `CARD_REPO`, `CARD_BRANCH` (2026-10-20) | `"MichaelJSr/TTSCardMaker"`, `"main"` | the card repo and its branch, moved here from query_card (`DEFAULT_REPOSITORY`, kept as an alias), repo_history (`REPOSITORY`, alias) and tts_save; /set_repo still overrides `card_repo.repository` at runtime |
| `API_BASE`, `RAW_BASE` (2026-10-20) | `"https://api.github.com/repos/"`, `"https://raw.githubusercontent.com/"` | URL prefixes used by commands/github_api.py (API) and repo_history/tts_save (raw links) |
| `TYPES_DIR`, `STARS_DIR`, `RULEBOOK_DIR` (2026-10-20) | `"Types"`, `"Types/Stars"`, `"Rulebook"` | card-repo folders: type icons, star icons, and the rule pages (a substring test on card paths); used by psd_parser, card_data, query_card, type_emojis |
| `ICON_MANIFEST_PATH`, `AUTHORS_PATHS` / `AUTHORS_PATH`, `EDITS_PATHS` / `EDITS_PATH`, `RULES_PATH` (2026-10-20) | `"MDW/Fonts/stoogedings-src/stoogedings.json"`, `("site/Card Authors.txt",)`, `("site/Card Edits.json",)`, `"TTS/decks.json"` | the hand-kept files in the card repo (font manifest, card authors, website edits, TTS deck rules). The website files moved from `MDW/` to `site/` on 2026-10-20 (TTSCardMaker 13ac680); every reader (card_data, query_card, tts_save) takes the first path present, so the tuples stay; `AUTHORS_PATH` / `EDITS_PATH` are the first element, for messages and the facade. The `MDW/` fallback was dropped the same day. Moved here from psd_analyzer and tts_save (psd_analyzer re-exports them) |
| `FONT_ALIASES` (2026-10-21) | `{"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}` | the Windows PostScript names of two faces MDW/Fonts holds under other names (the targets are the faces' FreeType-style names, `Kung Fu Panda` -> `KungFuPanda`), so the fork's FontMap (commands/psd_edit.py, where the dict lived until 2026-10-21) and the font check (commands/font_check.py) count them as present; here because both importers load from a sparse Action checkout without the rest of the bot (stdlib-only file) |
| `TTS_SAVE_PATH` (2026-10-20) | `"TTS/Impossibility Simulator.json"` | the Tabletop Simulator save in the card repo that `/update_tts_save` updates and commits (the automatic path is TTSCardMaker's tts-save.yml Action, which passes the path on the command line); players download it from its raw link (commands/tts_save.py) |
| `TIMEZONE` | `ZoneInfo("America/Los_Angeles")` | reminder times are PST/PDT — `/set_reminder when:` is interpreted in this zone (ops details: autocroissant-run-and-operate) |

Pickle schemas and commit discipline are owned by **impossibility-cards-reference** and
**autocroissant-change-control** respectively.

## Module-constant catalog

Every `commands/*.py` opens with a `# Configuration` (or `# Configurable Parameters`) block.
Catalog per module, verified 2026-07-11; re-verified 2026-10-20:

### card modules: commands/card_data.py, repo_traverser.py, psd_parser.py, psd_analyzer.py (psd_analyzer split 2026-10-20; the traverser's data path moved to card_data.py 2026-10-22)

The first five rows and `UPDATE_RATE`/`EXCLUDE_FOLDERS` live in commands/card_data.py:38-42 since 2026-10-22 (the
traverser body moved there, uncompiled; repo_traverser.py re-imports them); `_tree_etags`/`_file_cache` stay in
repo_traverser.py.

| Constant | Value | Meaning |
|---|---|---|
| `DOWNLOAD_WORKERS` (2026-10-20) | `4` | PSDs `/update_stats` downloads ahead of parsing |
| `LOOKUP_WORKERS` (2026-10-20) | `2` | commit lookups `/update_stats` runs ahead of parsing (one PyGithub client per thread) |
| `COMMITS_PAGE` (2026-10-20) | `100` | commits per lookup request: the newest and, for a new or authorless card, the first come in one page; a file with this many or more falls back to `.reversed[0]` |
| `_tree_etags`, `_file_cache` (2026-10-20) | in-memory | tree ETag per repo (304 = skip the run) and small repo files by blob sha; reset on restart |
| `TITLE_REGION_RATIO` (2026-10-20) | `0.12` | hidden text layers above this share of the card height are duplicates' titles (`card.alt_names`) |
| `UPDATE_RATE` | `25` | progress-message cadence during `update_stats`: an edit is queued every 25 processed cards (`num_updated % UPDATE_RATE == 0`, card_data.py:515) |
| `TYPE_REGION_RATIO` | `0.5` | fraction of card height splitting "creature type icons" (above midline) from "inject-into-ability icons" (below). Semantics: impossibility-cards-reference |
| `EXCLUDE_FOLDERS` | `["Markers", "MDW"]` | top-level TTSCardMaker folders skipped during stats traversal |
| `EXPORTED_STATS_NAME` | `"stats"` | `/export_cards` writes `stats.csv` / `stats.txt` in the REPO ROOT (psd_analyzer.py:398, 405) — gitignored via `*.csv` / `*.txt` |
| `EXPORTED_RULES_NAME` | `"rules"` | `/export_rulebook` writes `rules.txt` in the repo root (psd_analyzer.py:459) |
| `COLUMN_ORDER` | 16 field names, `'aliases'`…`'subtype'` (`'subseries'` after `'series'` since 2026-10-20) | display/export field ordering for card text output and the CSV columns (used psd_analyzer.py:110-111, 389) |
| `MISSPELT_CARD_TYPES` | `['undread', 'tornado', 'error']` | layer names that trigger the MISSPELT TYPE validation problem (as of 2026-07-11 exactly one live card trips it: the "20 Creature Types" rulebook page, per inspect_pickle.py) |
| `ICON_FONT_PREFIX` (2026-10-19) | `"stoogedings"` | a text-layer style run whose font PostScript name starts with this (case-insensitive) is read as icon tokens; the font's full name is `stoogedings-Regular`. Changing it without renaming the font in TTSCardMaker silently turns token parsing off |
| `ICON_MANIFEST_PATH` (2026-10-19) | `"MDW/Fonts/stoogedings-src/stoogedings.json"` | repo-relative path of the font manifest (aliases + codepoints), read by `_repo_text`: fetched by blob sha through the GitHub API and cached in `_file_cache` (only the offline diagnostics helper `_icons_from_local` reads a local clone). Missing file = plain names only, no aliases or Glyphs-panel characters. A cross-repo contract with TTSCardMaker's build script (ttscardmaker-companion §1.3) |
| `ICON_VARIANTS` (2026-10-19) | `("", "tint", "sticker", "no")`; `+ "glow", "darkglow"` on branch work/glow-tokens (2026-10-23) | the `:variant` suffixes the parser accepts; any other suffix is an `UNKNOWN ICON TOKEN` problem |
| `ICON_SCAN_PATTERN` (2026-10-19) | `\[[^\[\]\r\n\x03]{1,52}\]` or one character in U+E000–U+F8FF or ★ (U+2605) | what the parser treats as a candidate token: a bracketed run of 1-52 characters (the longest icon name with `:darkglow`) on one line, a Private Use Area character, or the star. Semantics: impossibility-cards-reference §4.1 |
| `EDITS_PATHS` / `EDITS_PATH` (2026-10-20) | `("site/Card Edits.json",)` | website fixes; the first path in the card repo's tree is read (`site/` since 2026-10-20, `MDW/` before). Read on every `/update_stats` (`CardEdits`, `_apply_edits`; PSD values kept in `card.parsed_values`) and by `populate_files` for website names/aliases. Fields = the website's EDIT_FIELDS (`CardEdits.TYPES`; `subseries` since 2026-10-20, database-only like note/aliases/hidden: impossibility-cards-reference §6). Missing or bad JSON: warning, cards unchanged |
| `AUTHORS_PATHS` / `AUTHORS_PATH` (2026-10-20) | `("site/Card Authors.txt",)` | card-repo paths of the authors file; the first one present is read. It is read on every `/update_stats` (`_repo_text`, cached by blob sha) and applied to every card (`CardAuthors`, `_apply_authors`). If missing or unreadable, a warning is printed and authors stay as they are. Semantics: impossibility-cards-reference (Author) |
| `STAR_CORE_ALPHA` (2026-10-20) | `0.5` | opacity at or above which a star layer's pixels count as its core when overlapping layers are merged into one star (`_count_stars`) |
| `EFFECT_LAYER_TAG` (2026-10-20) | `" FX: "` | marks an effect layer (`Ability FX: Blazing`, made by TTSCardMaker `MDW/Scripts/Effect Layers.jsx`). Such layers are skipped by the walk and checked afterwards. Semantics: impossibility-cards-reference §4.2 |
| `EFFECT_COVERAGE_MIN` (2026-10-20) | `0.95` | minimum share of an effect layer's opaque pixels that must sit on its text layer's pixels; below it the card gets `EFFECT LAYER OUT OF DATE` (fresh layers score 1.0, a stale test scored 0.268) |
| `PARSER_VERSION` (2026-10-21) | `2` (psd_parser.py:22) | stamped on every parsed card (`CardInfo.parser_version`); `/update_stats` re-parses a card whose stamp differs, once, and skips the tree ETag while any is stale (card_data.py:220, :247). Bump it when a parser change must reach the stored cards: the next run downloads every PSD (~813) |
| `CardValidator.ABILITY_EXCLUSIONS` | 21 card names (psd_parser.py:950-972) | cards allowed to have no ability layer |

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

### commands/query_card.py (lines 24-29)

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_REPOSITORY` | `"MichaelJSr/TTSCardMaker"` | GitHub repo the bot queries for card images; seeds `card_repo.repository` |
| `DEFAULT_MATCH_RATIO` | `0.6` | difflib fuzzy-match cutoff; seeds `card_repo.match_ratio` |

Both seed fields of the `CardRepository` dataclass whose singleton is `card_repo`
(query_card.py:796) — which is exactly why `/set_ratio` and `/set_repo` don't persist (below).
`card_repo.repository` is also the repo `/update_stats` traverses and, since 2026-10-19, the repo
`/sync_emojis` lists icons from.

### commands/utils.py (lines 15-19)

| Constant | Value | Meaning |
|---|---|---|
| `BREAK_LEN` | `1950` | max chunk length for `split_long_message` (Discord hard limit is 2000; 50 chars of headroom preserves ``` code fences). Queue/splitting mechanics: autocroissant-architecture-contract |
| `ADMINS` | getattr, default `[]` | see config.py table above |

### commands/music_player.py (lines 26-67)

| Constant | Value | Meaning |
|---|---|---|
| `MUSIC_BASE_DIR` | `"music/"` | download/playback directory (gitignored; the dir is kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite .gitignore's `!music/.gitkeep` exception) |
| `SKIP_FILES` | `{".gitignore", ".DS_Store"}` | filenames ignored when traversing `music/` (defined music_player.py:30, used :171, :470) |
| `cookies_from_browser` | derived (line 37): `browser_cookies`, else `vram_usage == "mps"` | see the browser_cookies section |
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

Also new 2026-10-20: `NET_TIMEOUT = 120` (commands/update_bot.py:9, seconds per git fetch/pull/push) and
`MAX_CONTENT = 2000` (main.py:779, Discord's message limit; longer content is split before sending).

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
| `RULES_PATH` | `"TTS/decks.json"` (global_config.py since 2026-10-20) | deck rules (which repo PNGs each complete deck holds). The bot reads main's copy; standalone runs read the checkout's, else raw main |
| `LAG_PAUSE` | `10` s | wait before redoing a run whose history clone lagged main |
| `HTTP_TIMEOUT` | `30` s | the website's cards.json and raw rules reads |
| `REPORT_FILE` | `"tts_save_report.md"` | name of the report attached in Discord |

The commit path (token, `COMMIT_TRIES`, `API_TIMEOUT`, the SSH fallback and its `_cant_write` memory) moved to commands/git_commit.py on
2026-10-21 (below); `update_repo_save` passes `ssh_on_limit=True` (a rate-limited API falls to SSH for that run, as before).

### commands/git_commit.py (Configuration block, added 2026-10-21; not compiled; shared by tts_save and card_bake)

| Constant | Value | Meaning |
|---|---|---|
| `AUTHOR`, `GIT_IDENTITY` | AutoCroissant / autocroissant@gmail.com | the commit author and committer (API and SSH paths alike; as the PICKLE commits) |
| `COMMIT_TRIES` | `3` | how many times a caller redoes its step when the branch moved (409/422, or a rejected push) |
| `CONFLICT` | `(409, 422)` | ref update refused: not a fast forward |
| `NO_WRITE` | `(401, 403, 404)` | POST git/blobs refused: the token can't write (classic PATs get 404); `cant_write` remembers it until a restart |
| `GIT_TIMEOUT`, `API_TIMEOUT` | `300` s each | per git call in the history clone / per API call (a PSD blob is up to 85 MB, 113 MB as base64) |
| `cant_write` | in-memory, starts `False` | set on a NO_WRITE reply that is not a rate limit; every later commit goes over SSH |
| `limited` | in-memory, starts `False` | set on a rate-limited 403 with `ssh_on_limit`; the run stays on SSH until the caller resets it (tts_save does at each run's start) |
| `token()` | `github_api.GIT_TOKEN`, else env `GITHUB_TOKEN` | the Actions have no config.py; `github_api.token_from_env()` patches the fetch headers the same way |
| `commit_files(..., ssh_on_limit=False)` | | a rate-limited 403 raises `Transient` (card_bake: the Action reports it) unless the caller asks for SSH for the run (tts_save: the bot's agent pushes; `limited`) |

Removed 2026-10-20 with the post-/update_stats run (TTSCardMaker's tts-save.yml Action owns the automatic path):
`IMAGES`, `_retry`, `_watched`, `save_affected`, `update_after_stats`.

### commands/font_check.py (Configuration block, added 2026-10-21; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `FONTS_DIR` | `"MDW/Fonts"` | the card repo's fonts folder; top level only (`stoogedings-src/` is the icon font's source) |
| `EXCLUDE_FOLDERS` | `("MDW", "Markers")` | PSDs under these path parts are not cards (card_data's list) |
| `ROLE_ID` | `"546427268751294475"` | the card makers' role the Discord message pings (`<@&id>`; `allowed_mentions` names that role only) |
| `BITE` | `"{name} has pushed a card without adding the {font} font to MDW/Fonts, bite this user."` | the user's wording, one line per (author, font) |
| `DISCORD_LIMIT` | `2000` | the webhook payload is cut to one message; the rest is in the commit comment |
| `NOBODY` | `"Someone"` | the author when no login is known (a checkout scan, the bot) |
| `CACHE_JSON` | `"font_names.json"` | the bot's per-blob cache (gitignored with its `.tmp`) |
| `BATCH` | `200` | PSDs `/fonts` downloads per run (a 15 MB blob took 1.4 s on 2026-10-21: about 6 min a batch, inside Discord's 15-minute window) |
| `API_TIMEOUT` | `60` s | per GitHub API call |
| `FILES_PAGE` | `300` | the compare and commit APIs list at most this many files; a bigger push, or a commit listing that many, is noted in the report |
| `COMMITS_PAGE` | `250` | the compare API lists at most this many commits; a bigger push is noted (later files may carry the wrong author) |
| `POOL_MIN`, `POOL_CHUNK` | `8`, `4` | `--all` parses this many PSDs or more in a process pool of half the cores, this many per task (813 PSDs: 98 s -> 9.8 s on the Mac, 2026-10-21); `/fonts` and `--repo` read blobs one by one |

The Actions' token: `GITHUB_TOKEN` from the environment when config.py supplied none (`github_api.token_from_env()`, called by font_check's and card_bake's entry points since 2026-10-21).

### commands/card_check.py (Configuration block, added 2026-10-21; not compiled; CLI only)

| Constant | Value | Meaning |
|---|---|---|
| `SPARSE_FILES` | `("commands/card_check.py", "commands/font_check.py", "commands/github_api.py", "commands/psd_parser.py", "commands/card_models.py", "commands/card_edits.py", "global_config.py")` | the import closure; TTSCardMaker's card-check.yml sparse-checks out exactly this list (`tests/test_card_check.py` asserts the closure equals it) |
| `LINE` | `"{name} has pushed {card} with: {notes}."` | the Discord line per card (card = `card_name(path)`, notes joined by "; ") |
| `CLOSING` | `"The bot's next /update_stats prints these too; the parser's own notes (no stars found, misspelt type, several creatures) also show on the website as Parser notes."` | the Discord message's last line (the validator's notes never reach the website) |
| `UNREADABLE` | `"could not be read"` | the error line's wording for a PSD psd-tools can't open or an edits file that isn't JSON |

Shared with font_check (imported): `ROLE_ID`, `DISCORD_LIMIT`, `NOBODY`, `API_TIMEOUT`, `EXCLUDE_FOLDERS`, `POOL_MIN`, `POOL_CHUNK`; the token
through `github_api.token_from_env()`. Repo paths from global_config: `TYPES_DIR`, `STARS_DIR`, `ICON_MANIFEST_PATH`, `EDITS_PATHS`.

### commands/psd_details.py (Configuration block, added 2026-10-21; not compiled)

| Constant | Value | Meaning |
|---|---|---|
| `VERSION` | `1` | the details file's `"version"` |
| `TEXT_CHARS` | `200` | characters of text kept per type layer |
| `TREE_CHILDREN`, `TREE_DEPTH` | `80`, `8` | the compact layer tree's caps; past them one `MORE` entry (`["…", "more", true]`) |
| `DRAWN` | `("pixel", "smartobject", "shape")` | digit-named layers of these kinds are drawn bar cells (stat encoding `pixel`); exposure-style adjustments are `template` / `two-row` |

The CLI's `--fonts` defaults to the clone's `MDW/Fonts`; `--out` to `DETAILS_JSON`. The pool size and chunking are font_check's `POOL_MIN` / `POOL_CHUNK`.
In `/update_stats` the `supplied` flags come from `font_check.cached_index(CACHE_JSON)` (the /fonts cache), None without it.

### commands/card_runner.py (added 2026-10-21; not compiled; the bake inside the bot by hand, design section 8 "Jetson"; the live runner is the Action)

| Name | Value | Meaning |
|---|---|---|
| `REPORT_FILE` | `"card_bake_report.md"` | the full bake report attached in Discord (the summary is the message) |
| `bake_lock` | `threading.Lock()`, module level | held by `update_stats` for its whole run and by the bake around `card_bake.run`: the two never overlap (queued commands run concurrently) |
| `/bake_cards` defaults | `card=None`, `dry_run=True` (main.py:401-407) | one card or all; render and check only unless `dry_run:False` |

The bake takes the token the bot already has: `card_bake.run` calls `github_api.token_from_env()`, which only patches the
headers when config.py supplied no `GIT_TOKEN`, and `git_commit.token()` prefers `GIT_TOKEN`; no environment variable is
needed inside the bot. The fonts dir and work dir are card_bake's (`PSD_TOOLS_FONT_DIRS` else `LOCAL_DIR_LOC/MDW/Fonts`;
`RUNNER_TEMP` else a fresh `mkdtemp`): the jetson has neither a checkout nor the env, see run-and-operate section 7.

### commands/analytics.py — reminders (lines 26-56, added 2026-10-19)

| Constant | Value | Meaning |
|---|---|---|
| `REMINDER_VERSION` | `3` (2 until 2026-10-20) | stored as `v` on each reminder once migrated; reminders without `v` are read with the old command semantics, and any reminder below the current version is migrated on load (3 = retired options stripped) |
| `MAX_INTERVAL` | `timedelta(days=366)` | upper limit for `offset` and `frequency` (longer = typo, and huge values overflow datetime) |
| `CHANNEL_RETRY` | `timedelta(hours=1)` | how long a one-shot reminder whose channel is gone waits before trying again |
| `LEGACY_RUNS_AS_BOT` | `True` | reminders saved before creator tracking keep running as the bot. Set False to run them as an unknown non-admin user instead (their admin commands would then be refused) |
| `RETIRED_OPTIONS` (2026-10-20) | `{"update_stats": (("use_local_repo", "use_local_timestamp"), <old option order>)}` | options a command no longer has. A reminder that names one drops it instead of failing: at creation (the reply says "Dropped ..."), at load migration and at fire time. The old option order maps pre-v2 positional values. Replaced `SCHEDULED_DEFAULTS` (2026-10-19 → 2026-10-20, `{"update_stats": {"use_local_repo": False}}`), which became pointless when local mode was archived |
| `INTERVAL_UNITS` | s/m/h/d/w in seconds | the units `offset`/`frequency` accept; combinations like `1d12h` are allowed |
| `WEEKDAYS`, `TIME_FORMATS` | Monday-first names; `("%H:%M", "%I%p", "%I:%M%p")` | what `when` accepts (weekday prefixes need 3+ letters) |
| `TRUE_WORDS`, `FALSE_WORDS` | true/yes/y/on/1 and false/no/n/off/0 | how boolean command options are read (any case) |
| `DATE_PREFIX`, `WORD_PREFIX`, `INTERVAL_FULL`, `INTERVAL_PART`, `OPTION_NAME` | compiled regexes | the `when`, interval and `name:value` grammars |

None of these is a config.py field; changing them is a code change. Reminder behavior in practice:
autocroissant-run-and-operate §9.

### setup.py (lines 8-48) — Cython build configuration

| Constant | Meaning |
|---|---|
| `EXCLUDE_FILES` | 8 entries never compiled, each with an in-file reason comment (config files, main.py entry point, update_bot.py's execv/execl, utils.py dynamic behavior, `__init__`s) |
| `INCLUDE_FILES` | the 7 command modules that DO get compiled (diffusion.py left with the AI code on 2026-10-20; type_emojis.py, added 2026-10-19, is in neither list, so it is never compiled) |
| `COMPILER_DIRECTIVES` | `wraparound: False` (negative indexing ban!), `boundscheck: False`, `cdivision: True`, etc. |

Why these are set this way, the build/clean commands, and the stale-`.so`-shadows-your-`.py`
trap: **autocroissant-build-and-env**.

## Runtime-settable vs restart-required

THE classic confusion in this bot. Two facts, verified by reading every setter:

1. The two `/set_*` commands on main assign to `card_repo` attributes in memory. **Neither
   writes config.py** — no file write exists in either. Changes evaporate on restart. And
   restarts are routine: every `/update` (the multi-machine pickle-sync flow) restarts the bot,
   silently resetting anything set at runtime.
2. `config.py` is the ONLY persistence, and only for the fields it has (tokens/ADMINS/music
   flags). `match_ratio` and `repository` are NOT config.py fields at all — they reset to their
   hardcoded defaults on every restart, and there is no way to persist a change short of editing
   `commands/query_card.py`.

| Slash command | Mutates (verified location) | After restart becomes |
|---|---|---|
| `/set_ratio` | `card_repo.match_ratio` (query_card.py:751-768; validates 0.0-1.0) | `0.6` (`DEFAULT_MATCH_RATIO`) — always |
| `/set_repo` | `card_repo.repository` + re-runs `populate_files` (query_card.py:771-794) | `"MichaelJSr/TTSCardMaker"` — always |

Restart-required only (no runtime setter exists): `TOKEN`, `ADMINS`, `GIT_TOKEN`,
`browser_cookies`, `vram_usage`, `KEEPVIDEO`, `POSTPROCESS` — edit config.py on that machine, then restart.

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
   (repo_traverser.py:59, main.py:383). Code that wants a forced re-parse must pass
   `force_update=True` explicitly.
2. **`use_local_repo` / `use_local_timestamp` — removed 2026-10-20.** `use_local_repo` defaulted to
   True (local mode) until 2026-10-19 and to False for one day; `use_local_timestamp` defaulted to True,
   which made local mode compare file mtimes with stored GitHub commit dates and re-archive every card
   (docs/BUGS_AND_IMPROVEMENTS.md B6). On 2026-10-20 local mode was archived: both options are gone from the
   slash command and from `update_stats()`, the traversal lives on branch `archive/local-update-stats`
   (not in main, not compiled), and stored reminders drop the options (`RETIRED_OPTIONS` above).
   (Do not run update_stats at all while merely investigating config — it mutates pickles.)
3. **Same field, two meanings:** on main `vram_usage` is only the browser_cookies fallback
   (music_player.py:37, default `""`); on branch `archive/ai-generation` it is the diffusion memory
   mode (default `"low"`). A grep for the field must check both.
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
7. Update this skill: add the field to the "fields main reads" table (making it 8), and extend
   the provenance greps. Route the code change itself through **autocroissant-change-control**.

## When NOT to use this skill

- Setting up an environment from scratch, installing requirements, Cython build/clean mechanics
  → **autocroissant-build-and-env**.
- What a flag's SUBSYSTEM actually does beyond the flag (diffusion pipeline internals →
  **autocroissant-ai-boundary** (branch `archive/ai-generation`); card parsing/validation semantics →
  **impossibility-cards-reference**; queue/dispatch machinery →
  **autocroissant-architecture-contract**).
- Starting/operating the bot, reminders in practice, the /push//update flow →
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
| card-module constants (psd_analyzer split 2026-10-20; moved 2026-10-22) | `grep -n "^UPDATE_RATE\|^EXCLUDE_FOLDERS\|^COMMITS_PAGE\|_WORKERS =" commands/card_data.py`; build constants: `grep -n "^BUDGET\|^RATE_FLOOR\|^MIN_CARDS\|^ARCHIVE_\|^REPARSE_MAX\|^QUALITY_\|^DETAILS_SHARE" commands/card_data.py`; `grep -n "TYPE_REGION_RATIO\|MISSPELT_CARD_TYPES\|^ICON_\|^EFFECT_\|STAR_CORE" commands/psd_parser.py`; `grep -n "EXPORTED_\|COLUMN_ORDER" commands/psd_analyzer.py`; repo paths: `grep -n "CARD_\|_BASE\|_DIR\|_PATH" global_config.py` |
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
| git_commit constants and the two callers | `grep -n "^[A-Z_]* = \|^cant_write\|^limited" commands/git_commit.py`; `grep -n "commit_files(" commands/tts_save.py commands/card_bake.py` |
| font_check constants and the shared alias table | `grep -n "^[A-Z_]* = " commands/font_check.py`; `grep -rn "FONT_ALIASES" --include="*.py" global_config.py commands/` (defined once, imported by psd_edit and font_check) |
| card_check constants and its sparse list | `grep -n "^[A-Z_]* = " commands/card_check.py`; `grep -n -A1 "^SPARSE_FILES" commands/card_check.py` |
| card_runner state (no hold line or edits sha since 2026-10-21) | `grep -n "^[A-Z_]* = \|^bake_lock" commands/card_runner.py`; `grep -n "bake_lock" commands/repo_traverser.py`; `grep -c "notice_edits\|_last_edits_sha" commands/card_runner.py commands/repo_traverser.py` (0 and 0) |
| global_config env overrides (exactly two: LOCAL_DIR_LOC, HISTORY_CLONE) | `grep -n "environ.get" global_config.py`; `grep -rn "AUTOCROISSANT_" --include="*.py" . ` (global_config.py and tests/test_tts_save.py only) |
| Setters never write config.py | `sed -n '670,708p' commands/query_card.py`; confirm zero `open(`/`write` there |
| update_stats defaults (force_update False both places; use_local_repo / use_local_timestamp removed 2026-10-20) | `grep -n "force_update: \|use_local" main.py commands/repo_traverser.py commands/card_data.py` (no use_local hits; every force_update default is False) |
| Icon-font constants | `grep -n "^ICON_" commands/psd_parser.py global_config.py` |
| type_emojis constants | `sed -n '14,26p' commands/type_emojis.py` |
| Reminder constants | `sed -n '26,56p' commands/analytics.py` |
| Bot identity (names only) | `grep -n "^#" config.py` shows which identity's TOKEN line is active (comments only — never print values) |
| global_config constants | `cat global_config.py` (32 lines) |
| setup.py build config | `sed -n '8,45p' setup.py` |
| Gitignore coverage (config.py, reminder.pkl, music/, *.csv, *.txt) | `grep -n "config.py\|reminder.pkl\|music/\|csv\|txt" .gitignore` |

When any command's output disagrees with this file, the repo wins: update the table, re-date the
"as of" stamps, and note the change here.

2026-10-21: query_card.py and main.py line numbers re-pointed after the atomic-save and 0.1 s tick changes (no flag or default changed; review: client.run, MAX_CONTENT, /set_ratio).

2026-10-21 (later): `FONT_ALIASES` moved from commands/psd_edit.py:47 to global_config.py:26 (32 lines now) and commands/font_check.py's
constants catalogued; no flag or default changed. Review the same day: main.py citations re-pointed after the /fonts block (:278-285, lazy
font_check import; `fonttools` added to requirements.txt).

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
