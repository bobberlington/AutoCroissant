---
name: autocroissant-config-and-flags
description: Catalog of EVERY configuration axis in the AutoCroissant Discord bot. Load this skill for any question containing "config", "flag", "setting", "default", "where is X configured", "how do I change/set X", or "why did my setting reset" — and for any mention of TOKEN, ADMINS, GIT_TOKEN, HF_TOKEN, model, lora, device_no, scheduler_name, vram_usage (low/medium/high/mps/distributed), KEEPVIDEO, POSTPROCESS, match ratio, /set_ratio, /set_repo, /set_scheduler, /set_model, /set_lora, /set_device, admin-only commands, rate limits, BREAK_LEN, UPDATE_RATE, TYPE_REGION_RATIO, EXCLUDE_FOLDERS, MUSIC_BASE_DIR, POSSIBLE_SCHEDULERS, LOCAL_DIR_LOC, TIMEZONE/PST, or pickle filenames. Contains the full 11-field config.py table (field NAMES only — values are secret), global_config.py constants, a per-module constant catalog, prod-vs-experimental status for each flag, the runtime-settable vs restart-required table (the /set_* commands do NOT persist across restart), the update_stats force_update defaults trap, the vram_usage=='mps' yt-dlp side channel, an add-a-new-config-field checklist, and re-verification greps for every table.
---

# AutoCroissant: Configuration and Flags

All facts verified against the working tree on 2026-07-11. Line numbers and defaults DRIFT —
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
| Runtime setters | `/set_ratio`, `/set_repo`, `/set_scheduler`, `/set_model`, `/set_lora`, `/set_device` | n/a | immediately — and LOST on restart |

Mental model: `config.py` is read ONCE, at import time, into module-level constants via
`getattr(config, "field", default)`. Nothing in the codebase ever re-reads it, and nothing ever
writes it. Consequences: (a) editing `config.py` does nothing until restart; (b) `/set_*` changes
do nothing after restart. Both directions of that asymmetry confuse people — see the
"Runtime-settable vs restart-required" section.

## config.py — the secret per-machine file

Ground rules:

- SECRET. It holds tokens. Never print, `cat`, quote, or commit its VALUES. Field names are
  public (listed below). File mode is `-rwx------` (0700) — keep it that way (verified
  2026-07-11: `ls -l config.py`).
- Gitignored (`.gitignore`, "Discord Bot Specific" section: `config.py`). Confirm with
  `git check-ignore -v config.py`.
- PER-MACHINE. The bot runs on the owner's Mac (`vram_usage='mps'`) and a CUDA box at different
  times; each has its own `config.py` with different values. A field you add on one machine does
  not exist on the other until someone hand-edits that machine's file. Every consumer therefore
  uses a `getattr` default — except TOKEN.
- The file must EXIST for the bot (and for any script importing `commands.*`) to even import:
  every command module does `import config`. Minimum viable `config.py` = a single `TOKEN = ...`
  line; every other field falls back to its default.
- Import side effect worth knowing: `query_card` prints a token-PRESENCE line at import
  ("Git token found, API limited to 5000 requests/hour." or the 60/hour variant,
  commands/query_card.py:42-45). Presence only — values are never printed.

### All 11 fields (as of 2026-07-11)

| Field | Required? | Default (getattr) | Read at | Status | Effect |
|---|---|---|---|---|---|
| `TOKEN` | YES — bot won't boot without it | none — the ONLY hard import: `from config import TOKEN` (main.py:9); used by `client.run(TOKEN)` (main.py:879) | main.py | prod | Discord bot token |
| `ADMINS` | no | `[]` (commands/utils.py:21) | utils.py | prod | list of Discord user IDs (ints) allowed past `perms_check` — see gate table below |
| `GIT_TOKEN` | no (strongly recommended) | `""` (commands/query_card.py:24 AND commands/psd_analyzer.py:34 — two independent reads) | query_card, psd_analyzer | prod | GitHub API auth: raises the rate limit 60 → 5000 req/hr. Used for the Authorization header (query_card.py:40, psd_analyzer.py:204) and the PyGithub client for remote timestamps (psd_analyzer.py:825, 859-860) |
| `HF_TOKEN` | no | `""` (commands/diffusion.py:29) | diffusion | experimental (AI) | HuggingFace token. Used ONLY inside `_init_flux_pipeline` (diffusion.py:267-324) to download the gated `black-forest-labs/FLUX.1-dev` repo. Irrelevant unless `model` contains "flux" |
| `model` | no | `""` (diffusion.py:24) | diffusion | experimental (AI) | `""` DISABLES pipeline init entirely (`if not MODEL: return`, diffusion.py:200-202). Otherwise dispatch by substring: "flux" → Flux from HF; "xl" → SDXL from `./models/<model>`; else SD1.5 from `./models/<model>` (diffusion.py:215-222, 328-354). Missing file raises FileNotFoundError, caught by init's try/except — bot keeps running (diffusion.py:250-253) |
| `lora` | no | `""` (diffusion.py:25) | diffusion | experimental (AI) | LoRA filename inside `./models/loras/`; loaded only if the file exists (diffusion.py:230-232) |
| `device_no` | no | `0` (diffusion.py:26) | diffusion | experimental (AI) | CUDA GPU index, passed as `gpu_id` to the CPU-offload calls — only matters for `vram_usage` low/medium on the CUDA box (diffusion.py:388, 390) |
| `scheduler_name` | no | `""` (diffusion.py:27) | diffusion | experimental (AI) | `""` or unrecognized → keep the model's default scheduler (`_configure_scheduler` returns None, diffusion.py:357-372). Recognized prefixes: "dpm++ sde" (DPMSolverSinglestep, + Karras sigmas if it endswith "karras" and model isn't flux), "euler a" (EulerAncestralDiscrete). The `POSSIBLE_SCHEDULERS` whitelist is enforced only by `/set_scheduler`, NOT for the config.py value — a typo silently falls back to the default scheduler |
| `vram_usage` | no | `"low"` (diffusion.py:28) | diffusion AND music_player (side channel!) | experimental (AI) + quirk | memory strategy — full mode table below |
| `KEEPVIDEO` | no | `True` (commands/music_player.py:54) | music_player | prod | yt-dlp `keepvideo`: keep the downloaded media file after any post-processing |
| `POSTPROCESS` | no | `False` (music_player.py:61) | music_player | prod | if True, appends a yt-dlp `FFmpegExtractAudio` postprocessor: mp3 at 320k (music_player.py:61-67) |

Prod-vs-experimental doctrine (user-stated 2026-07-11): the AI/diffusion capability is an
EXPERIMENT. The five diffusion fields plus HF_TOKEN must be absent-able: with none of them set
(and even with torch not installed) the bot must boot and run every non-AI feature. That is what
the getattr defaults + `model=''` + lazy torch import guarantee. Details and rules for keeping it
that way: see **autocroissant-ai-boundary**.

### ADMINS — which commands it gates

`perms_check(interaction)` (commands/utils.py:306-316) is INVERTED: it returns **True when the
user LACKS permission** (`interaction.user.id not in ADMINS`). Every call site reads
`if perms_check(...): send "You do not have permission"; return`. Misreading this polarity is a
known review trap — the invariant is owned by **autocroissant-architecture-contract**.

Commands gated by `perms_check` in main.py (all 11 call sites, verified 2026-07-11; line numbers
drift — re-derive with `grep -n "perms_check" main.py`):

| Slash command | main.py line |
|---|---|
| `/stop_bot` | 160 |
| `/pull` | 174 |
| `/push` | 184 |
| `/update` | 197 |
| `/set_reminder` | 225 |
| `/list_guilds` | 719 |
| `/leave_guild` | 734 |
| `/sync_global` | 743 |
| `/list_guild_members` | 758 |
| `/list_guild_channels` | 773 |
| `/get_channel_messages` | 793 |

Quirk, stated plainly: `/restart_bot` (main.py:149-155) is NOT gated — anyone in the guild can
restart the bot, while `/stop_bot` is admin-only. Observed code fact as of 2026-07-11; whether
that is intentional is unknown (candidate cleanup — route through autocroissant-change-control).

### vram_usage — the five modes

Accepted values are never validated anywhere; they are plain string comparisons in
`_apply_memory_optimizations` (diffusion.py:375-390) and `init_pipeline` (diffusion.py:210-212).
A typo (e.g. "lo") silently lands in a fallthrough: float16, VAE slicing+tiling, no offload, no
device placement. One line per mode — **the canonical full mode table (dtype, offload, VAE opts,
quantization, per-mode caveats) lives in autocroissant-ai-boundary**:

| Value | One-line behavior |
|---|---|
| `low` (default) | float16; VAE slicing+tiling; sequential CPU offload (`gpu_id=device_no`) — least VRAM, slowest |
| `medium` | float16; VAE slicing+tiling; model CPU offload (`gpu_id=device_no`) |
| `high` | float16; `_apply_memory_optimizations` applies NOTHING (no VAE opts, no offload). No code path explicitly moves the pipe to CUDA in this mode (the only pipe `.to(...)` is the mps one, line 378; line 697's `device` only seeds the `torch.Generator`) — UNVERIFIED on the CUDA box (detail: autocroissant-ai-boundary) |
| `mps` | **float32**; `pipe.to(device='mps')`, nothing else; Flux unquantized (bitsandbytes is CUDA-only); **plus the yt-dlp side channel below** |
| `distributed` | float16; VAE slicing+tiling; no CPU offload; `device_map='balanced'` when >1 CUDA GPU — wired only into the Flux init path (diffusion.py:212, 216) |

**The mps side channel (experimental/quirk — flag it whenever you touch vram_usage):**
`commands/music_player.py:37` reads
`cookies_from_browser: bool = getattr(config, "vram_usage", False) == "mps"`, and when true,
yt-dlp gets `"cookiesfrombrowser": ("safari", None, None, None)` (music_player.py:55). So a
DIFFUSION memory flag also switches the MUSIC downloader to Safari browser cookies. Rationale:
the Mac is the only machine running mps and the only one with Safari — the flag doubles as a
"am I on the Mac?" detector. Consequences: set `vram_usage='mps'` on a non-Mac and yt-dlp will
try to read Safari cookies that don't exist; run the Mac with any other value and downloads lose
the cookie support. Note the getattr default here is `False` (vs `"low"` in diffusion.py:28) —
harmless (`False == "mps"` is False) but the same field has two different defaults in two files;
keep that in mind when grepping. This hack is a candidate for replacement with an explicit field
(route through autocroissant-change-control).

## global_config.py — committed shared constants

Whole file, 9 lines (verify: `cat global_config.py`):

| Constant | Value | Meaning |
|---|---|---|
| `LOCAL_DIR_LOC` | `"~/Desktop/TTSCardMaker"` | the documented clone location of the card repo; expanded with `expanduser` by psd_analyzer's local traversal. This path convention is load-bearing across the toolchain |
| `ALIAS_PKL` | `"aliases.pkl"` | card-alias pickle (committed to git) |
| `STATS_PKL` | `"stats.pkl"` | the card stats database pickle (committed) |
| `OLD_STATS_PKL` | `"old_stats.pkl"` | archived old card versions (committed) |
| `REMIND_PKL` | `"reminder.pkl"` | reminders pickle (GITIGNORED, unlike the other three) |
| `TIMEZONE` | `ZoneInfo("America/Los_Angeles")` | reminder times are PST/PDT — `/set_reminder when:` is interpreted in this zone (ops details: autocroissant-run-and-operate) |

Pickle schemas and commit discipline are owned by **impossibility-cards-reference** and
**autocroissant-change-control** respectively.

## Module-constant catalog

Every `commands/*.py` opens with a `# Configuration` (or `# Configurable Parameters`) block.
Catalog per module, verified 2026-07-11:

### commands/psd_analyzer.py (lines 26-42)

| Constant | Value | Meaning |
|---|---|---|
| `UPDATE_RATE` | `25` | progress-message cadence during `update_stats`: an edit is queued every 25 processed cards (`num_updated % UPDATE_RATE == 0`, psd_analyzer.py:971 and 1047) |
| `TYPE_REGION_RATIO` | `0.5` | fraction of card height splitting "creature type icons" (above midline) from "inject-into-ability icons" (below). Semantics: impossibility-cards-reference |
| `EXCLUDE_FOLDERS` | `["Markers", "MDW"]` | top-level TTSCardMaker folders skipped during stats traversal |
| `EXPORTED_STATS_NAME` | `"stats"` | `/export_cards` writes `stats.csv` / `stats.txt` in the REPO ROOT (psd_analyzer.py:1561, 1568) — gitignored via `*.csv` / `*.txt` |
| `EXPORTED_RULES_NAME` | `"rules"` | `/export_rulebook` writes `rules.txt` in the repo root (psd_analyzer.py:1622) |
| `COLUMN_ORDER` | 15 field names, `'aliases'`…`'subtype'` | display/export field ordering for card text output and the CSV columns (used psd_analyzer.py:1418, 1503, 1552) |
| `MISSPELT_CARD_TYPES` | `['undread', 'tornado', 'error']` | layer names that trigger the MISSPELT TYPE validation problem (as of 2026-07-11 exactly one live card trips it: the "20 Creature Types" rulebook page, per inspect_pickle.py) |
| `CardValidator.EXCESSIVE_STAT_EXCLUSIONS` | 5 card names (psd_analyzer.py:642-648) | cards exempt from the stats>10 check |
| `CardValidator.ABILITY_EXCLUSIONS` | 18 card names (psd_analyzer.py:649-668) | cards allowed to have no ability layer |

The two exclusion sets are per-card whitelists the owner wants to eventually eliminate ("perfect
extraction" goal). Their semantics and the validation rules live in
**impossibility-cards-reference**; the elimination campaign in
**autocroissant-psd-extraction-campaign**.

### commands/query_card.py (lines 19-24)

| Constant | Value | Meaning |
|---|---|---|
| `DEFAULT_REPOSITORY` | `"MichaelJSr/TTSCardMaker"` | GitHub repo the bot queries for card images; seeds `card_repo.repository` |
| `DEFAULT_MATCH_RATIO` | `0.6` | difflib fuzzy-match cutoff; seeds `card_repo.match_ratio` |

Both seed fields of the `CardRepository` dataclass whose singleton is `card_repo`
(query_card.py:626) — which is exactly why `/set_ratio` and `/set_repo` don't persist (below).

### commands/utils.py (lines 17-21)

| Constant | Value | Meaning |
|---|---|---|
| `BREAK_LEN` | `1950` | max chunk length for `split_long_message` (Discord hard limit is 2000; 50 chars of headroom preserves ``` code fences). Queue/splitting mechanics: autocroissant-architecture-contract |
| `ADMINS` | getattr, default `[]` | see config.py table above |

### commands/music_player.py (lines 27-67)

| Constant | Value | Meaning |
|---|---|---|
| `MUSIC_BASE_DIR` | `"music/"` | download/playback directory (gitignored; the dir is kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite .gitignore's `!music/.gitkeep` exception) |
| `SKIP_FILES` | `{".gitignore", ".DS_Store"}` | filenames ignored when traversing `music/` (used music_player.py:157, 457) |
| `cookies_from_browser` | derived: `vram_usage == "mps"` | the side channel — see vram_usage section |
| `ydl_opts` | dict, music_player.py:50-58 | yt-dlp options: bestaudio, restricted filenames, `music/%(title)s-%(id)s.%(ext)s` template, `keepvideo` from KEEPVIDEO, Safari `cookiesfrombrowser` iff mps, quiet. `POSTPROCESS=True` appends mp3-320k extraction (61-67). Built ONCE at import — all three config fields are restart-required |

### commands/diffusion.py (lines 19-117)

| Constant | Value | Meaning |
|---|---|---|
| `MODELS_FOLDER` | `"./models/"` | where `.safetensors` checkpoints live (gitignored; the dirs are kept by the tracked `models/README` and `models/loras/README` stubs — no `.gitkeep` exists despite .gitignore's `!models/.gitkeep` exception) |
| `LORAS_FOLDER` | `"./models/loras/"` | where LoRA `.safetensors` live |
| `POSSIBLE_SCHEDULERS` | `["dpm++ sde", "dpm++ sde karras", "euler a"]` | whitelist enforced by `/set_scheduler` only (diffusion.py:425) |
| `SD15_RGB_FACTORS`, `SDXL_RGB_FACTORS(+_BIAS)`, `FLUX_RGB_FACTORS(+_BIAS)` | matrices, diffusion.py:34-117 | latent→RGB preview math cribbed from ComfyUI's latent_formats.py — not tunables, leave alone |

### setup.py (lines 8-45) — Cython build configuration

| Constant | Meaning |
|---|---|
| `EXCLUDE_FILES` | 8 entries never compiled, each with an in-file reason comment (config files, main.py entry point, update_bot.py's execv/execl, utils.py dynamic behavior, `__init__`s) |
| `INCLUDE_FILES` | the 8 command modules that DO get compiled |
| `COMPILER_DIRECTIVES` | `wraparound: False` (negative indexing ban!), `boundscheck: False`, `cdivision: True`, etc. |

Why these are set this way, the build/clean commands, and the stale-`.so`-shadows-your-`.py`
trap: **autocroissant-build-and-env**.

## Runtime-settable vs restart-required

THE classic confusion in this bot. Two facts, verified by reading every setter:

1. The six `/set_*` commands assign to in-memory state (module globals or `card_repo`
   attributes). **None of them writes config.py** — no file write exists in any of them.
   Changes evaporate on restart. And restarts are routine: every `/update` (the multi-machine
   pickle-sync flow) restarts the bot, silently resetting anything set at runtime.
2. `config.py` is the ONLY persistence, and only for the fields it has: the five diffusion
   fields (+ tokens/ADMINS/music flags). `match_ratio` and `repository` are NOT config.py fields
   at all — they reset to their hardcoded defaults on every restart, and there is no way to
   persist a change short of editing `commands/query_card.py`.

| Slash command | Mutates (verified location) | After restart becomes |
|---|---|---|
| `/set_ratio` | `card_repo.match_ratio` (query_card.py:582-599; validates 0.0-1.0) | `0.6` (`DEFAULT_MATCH_RATIO`) — always |
| `/set_repo` | `card_repo.repository` + re-runs `populate_files` (query_card.py:602-620) | `"MichaelJSr/TTSCardMaker"` — always |
| `/set_scheduler` | global `SCHEDULER_NAME` (diffusion.py:421-434; whitelists `POSSIBLE_SCHEDULERS`) | config.py `scheduler_name` (or `""`) |
| `/set_model` | global `MODEL` (diffusion.py:464-485; offers "flux" + `./models/*.safetensors`) | config.py `model` (or `""` = AI off) |
| `/set_lora` | global `LORA` (diffusion.py:488-512) | config.py `lora` (or `""`) |
| `/set_device` | global `DEVICE_NO` (diffusion.py:437-461; validates against `torch.cuda.device_count()`) | config.py `device_no` (or `0`) |

Restart-required only (no runtime setter exists): `TOKEN`, `ADMINS`, `GIT_TOKEN`, `HF_TOKEN`,
`vram_usage`, `KEEPVIDEO`, `POSTPROCESS` — edit config.py on that machine, then restart.

How the diffusion setters take effect: each one calls `_clear_pipelines()` (diffusion.py:515-519,
sets all three pipes to None); the NEXT generation request sees `not txt2img_pipe` and runs
`init_pipeline()` with the mutated globals (diffusion.py:677-679), telling the user
"Initializing AI pipeline... This may take a few minutes."

Each `/set_*` command called with no argument is the read path: it replies with the current
value (and the available options) without changing anything — the cheapest way to inspect live
state.

Related wart (owned by autocroissant-architecture-contract): main.py defines `slash_set_ratio`
TWICE — the second def (~main.py:313) is actually `/set_repo`'s handler; both commands work
because the decorator registered each at def time. Don't "fix" casually.

## Defaults-mismatch traps

1. **`update_stats` force_update** — the Python default and the slash default DISAGREE:
   - `update_stats(..., force_update: bool = True, ...)` — psd_analyzer.py:1173-1178
   - `/update_stats` slash wrapper passes `force_update: Optional[bool] = False` — main.py:363
   So `update_stats()` called bare from code (or from a reminder without args) FORCE-REPARSES
   every non-excluded PSD (~813 of the ~904 in the tree) and archives every old CardInfo, while
   the same command from Discord touches only changed cards. Never call it bare.
2. **Compounding it:** the slash default `use_local_repo=True` (main.py:361, matching the Python
   default at psd_analyzer.py:1175) currently routes into the removeprefix LIVE BUG that would
   corrupt stats.pkl at scale — until fixed, always pass `use_local_repo:False`. Full story and
   evidence: **autocroissant-failure-archaeology**; operational warning:
   **autocroissant-run-and-operate**. (Do not run update_stats at all while merely investigating
   config — it mutates pickles.)
3. **Same field, two defaults:** `vram_usage` defaults to `"low"` in diffusion.py:28 but `False`
   in music_player.py:37. Behaviorally harmless today; a grep for the field must check BOTH.
4. **Unvalidated config.py values:** neither `vram_usage` nor `scheduler_name` is checked against
   a whitelist at init — typos silently degrade (fallthrough memory behavior; default scheduler).
   Only the `/set_*` paths validate.

## How to add a new config field (checklist)

Story behind the rules: every field except TOKEN is getattr-defaulted precisely because there
are two machines with two different hand-maintained config.py files — a field that hard-fails
when missing takes the bot down on the OTHER machine at its next startup.

1. Choose the consuming module and add one line to its `# Configuration` block:
   `MY_FIELD = getattr(config, "my_field", <safe default>)`. Never `config.my_field` bare, never
   `from config import my_field` — only TOKEN does that, deliberately fatal.
2. Pick the default so that a machine WITHOUT the field behaves exactly as before the change.
   Test mentally against both machines: the Mac (mps, Safari, AI installed) and the CUDA box.
3. Add the field by hand to `config.py` on the machine(s) that need a non-default value. Keep
   mode 0700. NEVER commit it — confirm with `git check-ignore -v config.py` and never weaken
   .gitignore.
4. If it changes user-visible behavior, document it in `commands/help.py`'s dicts and the
   `@tree.command(description=...)` / `@app_commands.describe` text (templates:
   **autocroissant-docs-and-style**).
5. If it's AI-related: consume it ONLY in `commands/diffusion.py` behind getattr, keep torch
   imports lazy, and verify core startup is untouched (rules and the clean-venv proof:
   **autocroissant-ai-boundary**).
6. If it needs live toggling, add a `/set_*` command — and accept that the runtime change won't
   persist; config.py stays the only persistence. Say so in the command's help text.
7. Update this skill: add the field to the 11-field table (making it 12), and extend the
   provenance greps. Route the code change itself through **autocroissant-change-control**.

## When NOT to use this skill

- Setting up an environment from scratch, installing requirements, Cython build/clean mechanics
  → **autocroissant-build-and-env**.
- What a flag's SUBSYSTEM actually does beyond the flag (diffusion pipeline internals →
  **autocroissant-ai-boundary**; card parsing/validation semantics →
  **impossibility-cards-reference**; queue/dispatch machinery →
  **autocroissant-architecture-contract**).
- Starting/operating the bot, reminders in practice, the /push//update flow →
  **autocroissant-run-and-operate**.
- Diagnosing a misbehaving flag at runtime → **autocroissant-debugging-playbook**.
- The history of why a default changed → **autocroissant-failure-archaeology**.

## Provenance and maintenance

Everything above was verified 2026-07-11 by the commands below. Flags drift: regenerate each
table with its command before quoting it in the future.

| Fact / table | Re-verification command (cwd = repo root) |
|---|---|
| Master getattr map (all config.py consumers + defaults) | `grep -rn "getattr(config" --include="*.py" .` |
| TOKEN is the only hard import; run site | `grep -rn "from config import" --include="*.py" .` and `grep -n "client.run" main.py` |
| The 11 field names (NAMES ONLY — never print values) | `python3 -c "import config; print(sorted(k for k in dir(config) if not k.startswith('_')))"` |
| config.py mode 0700 + gitignored | `ls -l config.py` and `git check-ignore -v config.py` |
| perms_check gate list + the ungated /restart_bot | `grep -n "perms_check" main.py` then read each hit's enclosing `@tree.command`; `grep -n -A2 'name="restart_bot"' main.py` |
| perms_check inversion | `sed -n '306,317p' commands/utils.py` |
| psd_analyzer constants | `grep -n "UPDATE_RATE\|TYPE_REGION_RATIO\|EXCLUDE_FOLDERS\|EXPORTED_\|COLUMN_ORDER\|MISSPELT_CARD_TYPES" commands/psd_analyzer.py` |
| Exclusion set sizes/locations | `grep -n "EXCESSIVE_STAT_EXCLUSIONS\|ABILITY_EXCLUSIONS" commands/psd_analyzer.py` |
| UPDATE_RATE cadence usage | `grep -n "% UPDATE_RATE" commands/psd_analyzer.py` |
| query_card defaults | `grep -n "DEFAULT_REPOSITORY\|DEFAULT_MATCH_RATIO" commands/query_card.py` |
| BREAK_LEN | `grep -n "BREAK_LEN = " commands/utils.py` |
| music constants + side channel + ydl_opts | `grep -n "MUSIC_BASE_DIR\|SKIP_FILES\|cookiesfrombrowser\|keepvideo\|postprocessors\|preferredquality" commands/music_player.py` |
| diffusion folders/schedulers | `grep -n "MODELS_FOLDER\|LORAS_FOLDER\|POSSIBLE_SCHEDULERS" commands/diffusion.py` |
| vram mode behaviors | `sed -n '375,391p' commands/diffusion.py` plus `grep -n "device_map\|float32\|Generator" commands/diffusion.py` |
| model='' disables init; graceful failure | `sed -n '193,258p' commands/diffusion.py` |
| HF_TOKEN is Flux-only | `grep -n "HF_TOKEN" commands/diffusion.py` (all hits inside `_init_flux_pipeline` except the getattr) |
| GIT_TOKEN consumers/rate-limit prints | `grep -n "GIT_TOKEN" commands/query_card.py commands/psd_analyzer.py` |
| Setters mutate globals, never write config.py | `grep -n "global \|card_repo\." commands/diffusion.py commands/query_card.py` within lines 421-520 / 582-620; confirm zero `open(`/`write` in those ranges |
| Setter → lazy re-init path | `grep -n "_clear_pipelines\|if not txt2img_pipe" commands/diffusion.py` |
| update_stats defaults mismatch | `grep -n "force_update" main.py commands/psd_analyzer.py` |
| global_config constants | `cat global_config.py` (9 lines) |
| setup.py build config | `sed -n '8,45p' setup.py` |
| Gitignore coverage (config.py, reminder.pkl, music/, models/, *.csv, *.txt) | `grep -n "config.py\|reminder.pkl\|music/\|models/\|csv\|txt" .gitignore` |

When any command's output disagrees with this file, the repo wins: update the table, re-date the
"as of" stamps, and note the change here.
