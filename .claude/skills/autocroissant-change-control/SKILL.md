---
name: autocroissant-change-control
description: "How changes are classified, gated, and shipped in AutoCroissant. Load this BEFORE committing, pushing, merging, or reverting anything; before adding, renaming, or removing a slash command; before touching the pickles (stats.pkl, old_stats.pkl, aliases.pkl) or running /push, /pull, /update, or /update_stats (a pickle push now also publishes to the TTSCardMaker card website); before editing requirements.txt, .gitignore, or setup.py INCLUDE_FILES; before renaming or moving CardInfo/CardStats or any commands/*.py module; and whenever you are tempted to \"clean up\" odd-looking code (x[len(x)-1], double spaces, duplicate function names, the func = to_thread(func) lines). Contains: the four change classes and their gates, the /push//pull//update self-update anatomy and its pickles-ours/code-theirs merge strategy, the PICKLE commit convention and the diff_stats.py gate, the never-commit list, the add-a-slash-command checklist, a non-negotiables table with the historical incident behind each rule, and a list of changes that look safe but aren't."
---

# AutoCroissant change control

How changes get classified, gated, and shipped in this repo. AutoCroissant is a
personal Discord bot with **no CI, no README** and only an offline pytest suite (`tests/`) — the rules below are the
real safety system, and every one of them exists because something broke.
Facts and line numbers verified 2026-07-11 (repo at commit 284d13c, 192 commits); re-verified 2026-10-20
(HEAD 9fb3093, 207 commits), and after that day's optimization pass (working tree on 167f30a).

**Jargon, defined once:**

- **The pickles**: `stats.pkl` (card database, 813 cards), `old_stats.pkl` (archived
  old card versions), `aliases.pkl` (name shortcuts). They ARE the database and they
  are **committed to git**. Constants `STATS_PKL` / `OLD_STATS_PKL` / `ALIAS_PKL` in
  `global_config.py:6-10`. Since 2026-10-21 `psd_details.json` (`DETAILS_JSON`, global_config.py:11; the
  website's per-card PSD facts, commands/psd_details.py) is treated like a pickle: committed, pushed in the
  PICKLE commit, kept local on a /pull conflict.
- **PICKLE commit**: a commit containing only pickle files, message literally
  `PICKLE`, produced by the bot itself via `/push` (`commands/update_bot.py:28`).
- **Dev machine**: where a human edits code and runs `git push`.
- **Running host**: whichever machine currently runs the bot. Production is the Jetson
  (systemd `autocroissant.service`, Python 3.14 venv `~/venvs/autocroissant` since 2026-10-20); the owner's Mac at other times. It
  is the only legitimate writer of pickles.
- **TTSCardMaker**: the separate repo of ~904 card PSDs (`MichaelJSr/TTSCardMaker`,
  cloned at `~/Desktop/TTSCardMaker`). Card artwork/text lives THERE, not here. Since 2026-10-19 it also
  holds the stoogedings icon font and the card website that reads this repo's pickles
  (**ttscardmaker-companion**).

## The four change classes and their gates

| Class | Examples | Ships how | Gate before shipping |
|---|---|---|---|
| (a) Code | any `.py`, `setup.py`, skills | `git push` from a dev machine → on the Jetson: `git pull`, `~/venvs/autocroissant/bin/python setup.py build_ext --inplace`, `sudo systemctl restart autocroissant`. Discord `/pull` + `/restart_bot` or `/update` skip the Cython rebuild, so changed compiled modules stay stale — and since the 2026-10-20 split a stale `.so` can stop the bot at startup (analytics.so still imports `perms_check`), so after a code deploy always rebuild | The offline suite passes before every push: `~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (about 74 s, 356 tests on 2026-10-21; autocroissant-validation-and-qa row 1c). A change to the parser or any card module also needs the 813-card old-vs-new parse gate (validation-and-qa §4.1 method A: 0 differences, or every difference intended and written down first). A parser change that must reach the stored cards bumps `PARSER_VERSION` (psd_parser.py:22): the next `/update_stats` re-parses every card once (~813 downloads; validation-and-qa §4). Checklist below if it adds a command; non-negotiables table always |
| (b) Data (pickles) | stats.pkl, old_stats.pkl, aliases.pkl | Mutated ONLY by `/update_stats` (`stats_db.save()`, `StatsDatabase` in card_models.py; every save is atomic under its pickle's lock since 2026-10-21, `card_models.save_pickle`). Since branch work/discord-writers (2026-10-22) `/update_metadata`, `/mass_replace` and `/alias` write no pickle: each makes one commit to TTSCardMaker's `site/Card Edits.json` / `site/Card Authors.txt` (commands/discord_edits.py through `card_bake.land`) and applies it in memory; the next `/update_stats` builds it in; committed ONLY by `/push` or `/update` as a `PICKLE` commit | **diff_stats.py review before every pickle push** (see below) |
| (c) Config | config.py values (TOKEN, ADMINS, GIT_TOKEN, browser_cookies, ...) | Edited by hand per machine. NEVER committed (gitignored, `.gitignore:36` as of 2026-10-19). Values are secret — never print or quote them | New config *fields* route through autocroissant-config-and-flags |
| (d) Card content | a card's art, ability text, stats, folder; Types/ icons, the stoogedings font and its manifest | Edit in TTSCardMaker and push THERE; this repo only re-parses via `/update_stats` (and re-syncs emojis via `/sync_emojis`) | TTSCardMaker's own process; then the pickle gate applies to the re-parse. Renaming or deleting a Types/ PNG is a cross-repo change — ttscardmaker-companion §1.7 |

If your change mixes classes, split it. In particular: **never put pickle changes
and code changes in the same commit** — rationale in the next section.

## The self-update machinery (what /push, /pull, /update actually do)

All in `commands/update_bot.py` (registered in `main.py:176-210`). **The canonical
step-by-step anatomy — every message each command prints, the failure table, the
rebase fallback, and the restart mechanics — lives in autocroissant-run-and-operate
§5.** What change control needs you to know:

- **/push** → `git_push()` (update_bot.py): commits ONLY the three card pickles and the details file
  (`aliases.pkl stats.pkl old_stats.pkl psd_details.json`), message literally `PICKLE`, then
  `pull --rebase --autostash` onto new code (since 2026-10-20) and pushes, including a leftover
  unpushed PICKLE commit; "already up to date" only when nothing is ahead of origin.
- **/pull** → `git_pull()` (update_bot.py:50-118): fetch + merge, never a reset (its description
  said "hard reset" until 2026-10-20; archaeology 10.20). On merge **CONFLICT** it auto-resolves per
  file — **pickles → `checkout --ours` (running host's data wins); everything else →
  `checkout --theirs` (remote dev-machine code wins)** — committed as an
  `AUTO-RESOLVE: ...` merge commit.
- **/update** → `update_bot()` (update_bot.py:139-164): stops the three send/edit
  queue loops first (main.py:206-208), then `/push` + `/pull` + restart. With
  `force_reset:True` it instead runs `git reset --hard origin/main`
  (update_bot.py:121-136) and restarts.

### What the merge strategy implies

1. **The running host is authoritative for data; the dev machine is authoritative
   for code.** If you hand-edit or hand-commit a pickle from a dev machine and it
   conflicts, the running host's `--ours` resolution silently discards your version
   on the next /pull. Pickles flow through the bot or not at all.
2. **Never hand-edit pickles into a code commit.** The auto-resolver works per file,
   but the PICKLE-only convention is what makes bad data recoverable: when a corrupt
   snapshot was pushed on 2025-11-10 (cca0aaf, old_stats.pkl ballooned 3196 → 12041
   bytes from duplicate archiving), the fix was a clean one-commit revert (eb9aa84,
   `Revert "PICKLE"`) that touched only pickles. A mixed commit would have forced
   reverting code along with data.
   **The sanctioned recovery exception:** when a bad PICKLE commit has already been
   pushed, a human `git revert` of that commit is the approved fix (the eb9aa84
   precedent) even though the revert is not itself a bot-made PICKLE commit — allowed
   ONLY when the reverted commit touches pickles alone, and only with the diff_stats.py
   gate re-run on the result before any push. Step-by-step recovery procedure:
   autocroissant-debugging-playbook §1.
3. **`force_reset:True` is dangerous** — main.py:199 literally labels it
   `(dangerous!)`. It skips the pickle push entirely and hard-resets to
   `origin/main`, discarding ALL local state: uncommitted code edits AND every
   pickle change since the last successful /push (e.g. an afternoon of
   `/update_stats` and `/alias` work). Use it only to un-wreck a broken checkout,
   only after accepting that data loss, and ideally only right after a successful
   /push.

## Pickle-commit discipline (the data gate)

**The user-ranked #1 costliest failure class in this project is pickle data
corruption.** A committed-and-pushed bad pickle propagates to the other host on its
next /update, and to the live card website (https://michaeljsr.github.io/TTSCardMaker/)
within about 6 hours (its 6-hourly build reads the pickles on this repo's `main`;
ttscardmaker-companion §2.5).

Before ANY pickle push (i.e. before running /push or /update after anything touched
the card DB), run the diff gate from the repo root:

```
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
```

Exit 0 = no changes; exit 1 = changes found — read them; exit 2 = load error (worse:
the working pickle may be unreadable). Red flags: cards "removed" that nobody
deleted, mass path changes, type counts shifting, hundreds of cards whose only change is
`timestamp` (the signature of a sweep that used local mtimes, docs/BUGS_AND_IMPROVEMENTS.md B6: only possible from
the archived traversal on branch `archive/local-update-stats` since local mode was archived on 2026-10-20), and old_stats.pkl
growing more than about 10% in one run (`git cat-file -s HEAD:old_stats.pkl; wc -c old_stats.pkl` —
diff_stats.py reads stats.pkl only). A version is archived only when its PSD blob changed or the
card moved, so a `cards:`/`force_update` re-parse of unchanged PSDs should not grow it at all. Interpretation guide and the other
four scripts: **autocroissant-diagnostics-and-tooling**. The full acceptance
procedure for parser changes (goldens → sandbox sweep → diff review):
**autocroissant-validation-and-qa**.

**Why reminder.pkl is gitignored while the other three pickles are committed**
(`.gitignore:39`, `REMIND_PKL` in global_config.py:10): stats/old_stats/aliases are
the shared card database that must stay consistent across the two hosting machines —
syncing them is the whole point of /push//update. reminder.pkl is volatile runtime
scheduling state that mutates whenever reminders fire or reschedule; committing it
would generate a conflict on every handoff. Accordingly, `git_push` adds only the
three card pickles and `psd_details.json`, never reminder.pkl.

**Saves are atomic and locked (since 2026-10-21, B10).** `card_models.save_pickle` writes `<pickle>.tmp` beside the
target, fsyncs and `os.replace`s it, under one module-level `threading.Lock` per pickle (`_stats_lock`,
`_old_stats_lock`, query_card's `_alias_lock`); reminder.pkl has had the same shape since 2026-10-19. Keep it: a new
writer goes through `save_pickle` (a plain `open(..., 'wb')` brings B10 back), loads stay unlocked, and a `*.pkl.tmp`
left by a hard kill is gitignored and safe to delete. Story: autocroissant-failure-archaeology 10.29.

## What must NEVER be committed

Verified against `.gitignore` and `git ls-files`, 2026-07-11; line numbers re-read 2026-10-20 (53db8ee
added `reminder.pkl.tmp` and `.claude/settings.local.json` and fixed the `*.pyd`/`*.dll` patterns;
05c643e dropped the `models/` lines with the AI code; repo_history.json and type_emojis.json lines added 2026-10-20 shifted the rest by 6):

| Path | .gitignore line | Why |
|---|---|---|
| `config.py` | 36 | Secret tokens (TOKEN, GIT_TOKEN...). Per-machine. A leaked token means rotation, not deletion — git history is forever |
| `.ssh/` | 37 | Credentials, same reasoning |
| `reminder.pkl` | 39 | Volatile per-host scheduling state (see above) |
| `*.pkl.tmp` | 41 (`reminder.pkl.tmp` since 2026-10-19; widened 2026-10-21) | Transient file of an atomic pickle save (reminder.pkl; stats, old_stats and aliases since 2026-10-21). Normally renamed away within milliseconds; a hard kill can leave one behind — delete it |
| `repo_history.json`, `.tmp` | 43-44 (added 2026-10-20) | Per-machine map of repo path -> commit that last changed it, rebuilt from the history clone by commands/repo_history.py |
| `type_emojis.json`, `.tmp` | 46-47 (added 2026-10-20) | Per-machine map of bot application id -> {type emoji name -> icon blob it was made from} (commands/type_emojis.py); if deleted, startup adopts the existing emojis as current (an icon edited meanwhile then needs `/sync_emojis refresh:True`) |
| `.claude/settings.local.json` | 54 (added 2026-10-19) | Personal Claude Code settings; the shared `.claude/skills/` library IS committed |
| `music/` | 50 | yt-dlp downloads; bulky, regenerable (dir kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite the `!music/.gitkeep` exception) |
| `*.log`, `logs/`, `*.csv`, `*.txt` | 69-72 | Logs and the /export_cards //export_rulebook exports (`stats.csv`, `stats.txt`, `rules.txt`) land in the repo root; the globs keep them out |
| `*.so`, `*.c`, `*.cpp`, `*.pyd`, `*.dll`, `build/` | 7, 14-15, 19-22, 28 | Cython build artifacts; see autocroissant-build-and-env for the stale-.so trap |

**Trap inside that table:** `*.txt` is gitignored, yet `requirements.txt` is
tracked — only because already-tracked files ignore .gitignore. Any NEW `.txt` (or
`.csv`) file you create will be **silently skipped by `git add .`**. If a new text file genuinely belongs in the repo, that is a
deliberate change-control decision: either force-add it or (better) amend
.gitignore in a reviewed code commit.

## Checklist: add a new slash command

Worked example to imitate: the `/update_stats` block at main.py:374-386 (2026-10-20).

1. **Write the worker function** in the right `commands/*.py` module (query logic in
   query_card.py, parsing in psd_parser.py, the /update_stats command in repo_traverser.py (its traversal is
   commands/card_data.py since 2026-10-22, shared with the Card sync Action's build), GitHub access in
   github_api.py, admin in management.py, etc.). Plain
   `def` is fine — it will be thread-wrapped. The body must follow the queue rules:
   never `await` Discord or call `interaction.followup` directly from worker code;
   send via `queue_message` / `queue_file` / `queue_edit` / `queue_command`
   (commands/utils.py:36-60). **autocroissant-architecture-contract** owns the
   full contract (queues, 15-minute interaction-token fallback, `lacks_admin`,
   which returns True when the user is NOT an admin, utils.py).
2. **Register it in main.py**, copying the house pattern exactly:
   ```python
   my_worker = to_thread(my_worker)                # rebind FIRST, on its own line
   @tree.command(name="my_command", description="...")
   @app_commands.describe(
       some_arg='What this argument does.',
   )
   async def slash_my_command(interaction: Interaction, some_arg: Optional[bool] = True):
       await interaction.response.defer()          # or send_message(...) for quick acks
       await my_worker(interaction, some_arg)
   ```
   `to_thread` (utils.py:80-89) wraps blocking functions in `asyncio.to_thread` and
   passes `async def` workers through unchanged, so the rebind line is correct for
   both. Keep the `func = to_thread(func)` line immediately BEFORE the decorated
   wrapper — that is the convention every block follows; see "looks safe but isn't"
   for why you must not delete or fold it.
   Admin-only commands start with the `if lacks_admin(interaction): ... return`
   block (copy it from /push, main.py:189-191). The newest example of a whole admin
   command block is `/bake_cards` (main.py:399-412, added 2026-10-21; its worker is the
   uncompiled commands/card_runner.py), before it `/sync_emojis` (main.py:713-726, 2026-10-19).
3. **Add `@app_commands.describe`** for every parameter. Description wording and
   style: **autocroissant-docs-and-style**. Note Discord quirks that already shaped
   signatures here: big integers (guild/user IDs) are passed as `str` because Discord
   ints cap around 15 digits.
4. **Add a help entry** in `commands/help.py`: pick the right dict
   (`general_commands`:6, `text_commands`:29, `card_commands`:42,
   `stats_commands`:79, `music_commands`:112), key format
   `"/my_command <required> [optional]:"`, value = description string.
   `print_help` (help.py:136) dispatches on the category word.
5. **Decide Cython placement** if you created a NEW module: `setup.py` compiles only
   the 11 modules listed in `INCLUDE_FILES` (setup.py:21-33); new files are uncompiled
   by default (`type_emojis`, `repo_history`, `github_api`, `tts_save`, `git_commit`, `font_check`, `card_check`, `psd_details`, the edit pipeline's `psd_edit`, `psd_regions`, `card_bake` and `card_runner`, the card data build's `card_data`, `card_history`, `card_db` and `changelog`, and the Discord writers' `discord_edits` (2026-10-22: the jetson runs the same pure Python as the Action; tests/test_card_data.py checks they stay out) are deliberately left out; the four
   modules split out of psd_analyzer on 2026-10-20 were added so the moved code keeps its compiled semantics). TTSCardMaker's Actions
   sparse-check out bot modules by path (card-edits.yml, tts-save.yml, font-check.yml, card-check.yml: the last one's list is
   `card_check.SPARSE_FILES`; the planned card-sync.yml will check out `card_data.SPARSE_FILES`, whose test runs `--help` on
   the build, the bake and the save with exactly that list), so renaming or moving one of those files, or adding an import to one, breaks an Action; three of the lists
   have a test that copies exactly it and runs `--help` (tests/test_card_bake.py, test_font_check.py, test_card_check.py; tts-save.yml's has
   none, tests/test_tts_save.py checks only the env overrides). If you add yours to INCLUDE_FILES, the `wraparound=False` directive
   (setup.py:39) applies: **no negative indexing anywhere in the file** — use
   `x[len(x)-1]`, never `x[-1]` — and Cython 3.3 rejects re-annotating a variable
   that is already declared (declare it once; the management.py incident,
   autocroissant-failure-archaeology Entry 10.5). Compile-check rules: autocroissant-build-and-env §5. Do not add modules that use `execv`/dynamic
   registration (that is why update_bot.py and utils.py sit in EXCLUDE_FILES,
   setup.py:9-18).
6. **Ship and sync.** Push from the dev machine, then deploy on the Jetson as in class (a)
   (`/update` or `/pull` + `/restart_bot` only if no compiled module changed). Commands register with Discord at startup:
   `on_ready` (main.py:107-128) records every command in `slash_registry` (which is
   also what lets reminders invoke commands) and syncs per guild
   (`sync_guild_commands`, main.py:99-103). If the command does not appear after
   restart, run `/sync_global` (main.py:705). Client-side, Discord may need a
   reload before new commands show up.
7. **Verify** by invoking the command in Discord; do not run `python3 main.py`
   casually to test — starting the bot connects it to the live guilds
   (autocroissant-run-and-operate owns safe start procedure).

## Non-negotiables

Each rule cites the incident that created it. Rules without their incident get
ignored — read the story before you decide the rule doesn't apply to you.

| # | Rule | Rationale + incident |
|---|---|---|
| 1 | **No negative indexing in cythonized modules** (anything in setup.py INCLUDE_FILES). Write `x[len(x)-1]`. | Cythonization (f7c915c, 2025-12-02) set `wraparound=False`, making `x[-1]` undefined behavior in compiled code. dd800bc (2026-01-30) is a one-line commit converting `bboxes[-1]` → `bboxes[len(bboxes)-1]`. The same mechanical negative-index purge in f7c915c also introduced the `removeprefix` local-path bug, live until 2026-10-19 — full story in **autocroissant-failure-archaeology** Entry 1 |
| 2 | **Never collapse runs of whitespace in ability text.** | Gaps of 3+ spaces are where type icons sit; they are the injection signal for `_inject_type_names`. The 2026-02-08 whitespace saga (8 commits in one day — 7 parser commits plus a mid-saga PICKLE — 1c26747 through 3bbaa2b/4bcee6b; chronology in autocroissant-failure-archaeology Entry 2) added `\s{2,}` collapsing to fix two cards, broke others, and ended with 3bbaa2b "Fix the issue of collapsing spaces" deleting the collapsing entirely. Whitespace is load-bearing |
| 3 | **All Discord sends from worker code go through `queue_*`.** No direct awaits of Discord from workers. | Single event loop + deferred dispatch with a fallback chain that survives the 15-minute interaction-token expiry. Detail owned by **autocroissant-architecture-contract** |
| 4 | **No AI on main.** Since 2026-10-20 the AI generation code lives only on branch `archive/ai-generation`; main must run with requirements.txt alone and gets no torch-family imports. | The diffusion capability is an experiment by owner doctrine; the bot must run perfectly without it. Verified 2026-07-11: clean venv with requirements.txt only → `import main` succeeds, torch absent. Detail owned by **autocroissant-ai-boundary** (branch `archive/ai-generation`) |
| 5 | **Every pickle push gets a diff_stats.py review first.** | cca0aaf (2025-11-10) pushed a bad snapshot (old_stats.pkl 3196 → 12041 bytes, duplicate archiving); it shipped because nobody could see what changed inside a binary blob. Reverted same day (eb9aa84). diff_stats.py turns "looks fine" into numbers |
| 6 | **Pickles produced by the archived local traversal (branch `archive/local-update-stats`) are never pushed.** It exists for sandbox sweeps only; `/update_stats` is remote-only since 2026-10-20 (operating rule owned by autocroissant-run-and-operate §7). If such pickles ever reach the live checkout, they must pass the diff_stats.py gate AND an old_stats.pkl size check before any push. | History: until 2026-10-19 local mode no-op-stripped "TTSCardMaker" from absolute paths and would have misclassified every card (the removeprefix bug, fixed with `relpath`; archaeology Entry 1). Still true after the fix: with its default `use_local_timestamp=True`, local mtimes are compared with stored GitHub commit dates — 813/813 newer on 2026-10-19 — so a local run archives every card into old_stats.pkl, the cca0aaf/e5f5393 balloon pattern (docs/BUGS_AND_IMPROVEMENTS.md B6). The slash default became remote on 2026-10-20 and the same day local mode was archived: `use_local_repo`/`use_local_timestamp` no longer exist on the command, and stored reminders drop them (`RETIRED_OPTIONS` in analytics.py) |

## Changes that look safe but aren't

- **Renaming `CardInfo`/`CardStats` (in commands/card_models.py since the 2026-10-20 split, with `__module__`
  pinned to `commands.psd_analyzer`), dropping that pin, or renaming `commands/psd_analyzer.py`.** Pickle stores the
  class's import path (the pin is why the split left stats.pkl byte-identical); the committed
  stats.pkl/old_stats.pkl would fail to unpickle on every machine after the next
  pull. Schema/location changes require a data migration shipped in lockstep —
  precedent: the "Big massive refactor" (366c8d9, 2025-10-20) changed the schema and
  had to delete and rebuild the old pickle files in the same commit.
- **"Cleaning up" `x[len(x)-1]` to `x[-1]`** in any INCLUDE_FILES module. That idiom
  is deliberate (non-negotiable #1; dd800bc). Same for adding `.removeprefix()`-style
  "modernizations" without checking what the original expression did — that exact
  move created the local-path bug that lived from 2025-12-02 to 2026-10-19.
- **Adding or "fixing" a type hint in an INCLUDE_FILES module.** Cython enforces builtin annotations: `x: list`
  rejects a tuple, `x: dict` rejects a defaultdict (old_stats is one), `x: str` rejects None unless the default is None,
  `x: float` turns 1 into 1.0, `x: bool` turns None into False, `-> dict` rejects a defaultdict return. Measured with
  Cython 3.3 on 2026-10-20 (code-health audit §3.1); live example: compiled `analytics.parse_interval(None)` raises
  where the .py returns None. Only ABCs, unions, `Optional[...]` with a None default and dataclass fields are inert.
  Gate any hint change with a compiled run (autocroissant-build-and-env §5).
- **Deleting or merging a slash handler block in main.py, or changing its `name=`.** Each
  `@tree.command` registers its function at def time and `slash_registry` keys on `cmd.name`, so the
  Python def name is decorative: the two duplicate names (C5) were renamed on 2026-10-20 to
  `slash_set_repo` and `slash_delete_all_music` with no registration change (`tests/test_robustness.py`
  checks names are unique and map to the right commands). Deleting a block, or editing `name=`,
  does change what is registered and breaks stored reminders. Detail: **autocroissant-architecture-contract**.
- **Removing or moving the `func = to_thread(func)` rebind lines** because they look
  redundant next to the decorator. The rebind is what makes a blocking worker
  awaitable; break it and the command fails (or blocks the event loop) only when
  first *invoked*, not at startup — so it survives a smoke test and dies in
  production. Keep rebind-then-decorator, one block per command.
- **Adding a new `.txt`/`.csv`/`.log` file and assuming git tracks it.** The
  gitignore globs (lines 69-72 as of 2026-10-20) silently exclude it (see the trap note above).
- **Adding torch/diffusers or other AI deps to `requirements.txt`.** That breaks
  non-negotiable #4 (AI lives only on `archive/ai-generation`).
- **Hand-committing pickles from a dev machine**, or `git add -A` after a run that
  touched them. Your data loses to `--ours` on the running host, and you have broken
  the PICKLE-only revert property (see merge-strategy implications).
- **Importing card classes or `stats_db` from `commands.card_models` (or the other split modules) in
  query_card or new code.** Import them from the `commands.psd_analyzer` facade until every host has rebuilt:
  a stale monolithic psd_analyzer `.so` next to the new `.py` files would otherwise give two `stats_db`
  objects (autocroissant-architecture-contract, lazy-import cycle breaks).
- **Letting a test or script write the live pickles.** `tests/conftest.py`'s `pickles_untouched` fixture
  hashes stats.pkl, old_stats.pkl, aliases.pkl and reminder.pkl around the run and fails it on any change;
  keep new tests behind it (sandbox copies only).
- **Editing card text or stats in this repo.** Ability text and stats live in
  TTSCardMaker PSDs (class (d)); anything you "fix" in stats.pkl by hand is
  overwritten by the next `/update_stats` and was never legitimate to begin with.
- **Renaming a Types/ icon PNG in TTSCardMaker "to fix a name".** Image-layer type names,
  font tokens, emoji names and website icons all key off the file stem; the font manifest
  keeps the old entry, so old tokens keep parsing while the card shows letters. Add an alias
  instead (ttscardmaker-companion §1.7).
- **Changing `ICON_MANIFEST_PATH` (global_config.py) / `ICON_FONT_PREFIX` (psd_parser.py) or `emoji_name` rules
  in type_emojis.py** without the matching TTSCardMaker change. These are cross-repo contracts
  (values: autocroissant-config-and-flags).

## When NOT to use this skill

- Diagnosing why something is broken → **autocroissant-debugging-playbook**.
- The history/root-cause behind an incident cited here (removeprefix bug, eb9aa84
  revert, whitespace saga, the 2026-10-19 fix batch) → **autocroissant-failure-archaeology**.
- The TTSCardMaker side (icon font, card website, Types/ naming) → **ttscardmaker-companion**.
- Queue/threading/registration internals and invariants → **autocroissant-architecture-contract**.
- Running, restarting, syncing, or handing the bot between machines → **autocroissant-run-and-operate**.
- diff_stats.py and the other scripts' usage/outputs → **autocroissant-diagnostics-and-tooling**.
- Evidence standards and the parser-change acceptance procedure → **autocroissant-validation-and-qa**.
- Adding/altering config fields or flags → **autocroissant-config-and-flags**.
- Cython build mechanics, environments, dependency installs → **autocroissant-build-and-env**.
- Anything AI/diffusion-side → **autocroissant-ai-boundary** (branch `archive/ai-generation`).
- Command-description and help-text wording → **autocroissant-docs-and-style**.
- Card semantics (folders, layers, injection) → **impossibility-cards-reference**.

## Provenance and maintenance

All file:line references verified 2026-07-11 at commit 284d13c; the ones changed in this file
re-verified 2026-10-20 (HEAD 9fb3093). Line numbers drift;
re-verify with the commands below (run from `/Users/michaelsrouji/Desktop/AutoCroissant`)
before relying on any of them. If a check disagrees with this file, trust the check
and update this file.

| Fact | Re-verify with |
|---|---|
| PICKLE commits are the recent pickle history | `git log --oneline -3 -- stats.pkl old_stats.pkl aliases.pkl` |
| /push adds only the three pickles, message PICKLE | `grep -n "git.add\|PICKLE" commands/update_bot.py` |
| Merge strategy: pickles ours, code theirs | `grep -n -- "--ours\|--theirs" commands/update_bot.py` |
| force_reset behavior and its "dangerous!" label | `grep -n "force_reset" main.py commands/update_bot.py` |
| restart: `./startup.sh` if present, else re-exec main.py (the Jetson has no startup.sh) | `grep -n "execv\|execl" commands/update_bot.py` |
| Gitignore coverage (config.py, .ssh, reminder.pkl, music, txt/csv/log) | `git check-ignore -v config.py reminder.pkl music/x a.txt a.csv a.log` |
| Tracked pickles + grandfathered requirements.txt | `git ls-files '*.pkl' '*.txt'` |
| Incident hashes and their diffs | `git show --stat cca0aaf eb9aa84 366c8d9 f7c915c` ; `git show dd800bc` ; `git show 3bbaa2b 4bcee6b` |
| Cython include/exclude lists and wraparound=False | `grep -n "wraparound\|INCLUDE_FILES\|EXCLUDE_FILES" setup.py` |
| to_thread passes coroutines through; queue_* helpers | `grep -n "def to_thread\|def queue_" commands/utils.py` |
| /update_stats is remote-only; local mode archived (since 2026-10-20) | `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS` in analytics.py); `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived` |
| Rule 6 history: local-path fix lives on in the archive branch | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `git grep -c "removeprefix(" archive/local-update-stats -- archive/local_update_stats.py` (no output: the word only appears in the header's history note) |
| Stored reminders drop retired options | `grep -n -A6 "^RETIRED_OPTIONS" commands/analytics.py` |
| Handler names unique (C5 fixed 2026-10-20) | `grep -n "def slash_set_r\|def slash_delete_" main.py` (5 distinct names) |
| Registration pattern example block | `grep -n "update_stats = to_thread" main.py` |
| Help dict names and lines | `grep -n "_commands = {\|def print_help" commands/help.py` (6, 29, 42, 79, 112; 136) |
| Offline suite before a push | `~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (417 passed, 24 fork-only skips on 2026-10-21 with tests/test_card_check.py) |
| Pickle filename constants | `grep -n "PKL" global_config.py` |
| diff_stats.py gate exists and documents its exit codes | `head -20 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py` |
| Commit count / current head (date-stamp anchor) | `git rev-list --count HEAD` ; `git log --oneline -1` |
| Atomic locked saves (B10) | `grep -n "^def save_pickle\|^_stats_lock\|^_old_stats_lock" commands/card_models.py` (20, 16, 17); `grep -n "^_alias_lock\|save_pickle(" commands/query_card.py`; `git check-ignore -v stats.pkl.tmp` (.gitignore:41 `*.pkl.tmp`) |

2026-10-21: atomic locked pickle saves (B10) added to class (b) and the pickle discipline; `*.pkl.tmp` row; line numbers re-pointed.

2026-10-21 (titles): class (a) gate names the PARSER_VERSION rule (version 2: the printed title; suite 356 passed, 18 skipped; the class (a) row now says 356 tests, 74 s).

2026-10-21 (S7): `/bake_cards` is the newest admin command block (checklist step 2; help dict lines re-pointed, `card_runner` named among the uncompiled modules in step 5); tests/test_robustness.py counts 52 commands; main.py, repo_traverser.py and help.py citations re-pointed.
2026-10-21 (review): the offline-suite row reads 380 passed, 24 skipped (the S6/S7 working tree after the review fixes).

2026-10-21 (details): `psd_details.json` joins the pickle discipline (jargon bullet, the /push bullet, the reminder.pkl paragraph; `psd_details` named among the uncompiled modules in step 5). The never-commit list is unchanged: the file is a derived, committed artifact like the pickles, its `.tmp` gitignored. repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-21 (card check): `card_check` named among the uncompiled modules in step 5, with the four TTSCardMaker Actions whose sparse lists pin bot file paths; the offline-suite row reads 417 passed, 24 skipped; the sparse-list test clause names the three Actions that have one (tts-save.yml has none).

2026-10-22 (data build): the /update_stats traversal moved to the uncompiled commands/card_data.py (checklist step 1);
`card_data`, `card_history`, `card_db` and `changelog` named among the uncompiled modules in step 5, with
`card_data.SPARSE_FILES` (card-sync.yml, planned) beside the Actions' sparse lists.
