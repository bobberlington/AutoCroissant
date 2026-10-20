---
name: autocroissant-change-control
description: How changes are classified, gated, and shipped in AutoCroissant. Load this BEFORE committing, pushing, merging, or reverting anything; before adding, renaming, or removing a slash command; before touching the pickles (stats.pkl, old_stats.pkl, aliases.pkl) or running /push, /pull, /update, or /update_stats (a pickle push now also publishes to the TTSCardMaker card website); before editing requirements.txt/requirements2.txt, .gitignore, or setup.py INCLUDE_FILES; before renaming or moving CardInfo/CardStats or any commands/*.py module; and whenever you are tempted to "clean up" odd-looking code (x[len(x)-1], double spaces, duplicate function names, the func = to_thread(func) lines). Contains: the four change classes and their gates, the /push//pull//update self-update anatomy and its pickles-ours/code-theirs merge strategy, the PICKLE commit convention and the diff_stats.py gate, the never-commit list, the add-a-slash-command checklist, a non-negotiables table with the historical incident behind each rule, and a list of changes that look safe but aren't.
---

# AutoCroissant change control

How changes get classified, gated, and shipped in this repo. AutoCroissant is a
personal Discord bot with **no tests, no CI, no README** — the rules below are the
entire safety system, and every one of them exists because something broke.
Facts and line numbers verified 2026-07-11 (repo at commit 284d13c, 192 commits); re-verified 2026-10-19
(HEAD a7a4bb3, 193 commits, plus an uncommitted working-tree batch: psd_analyzer fixes and icon-font
parsing, commands/type_emojis.py, the reminder rewrite, management.py Cython fix).

**Jargon, defined once:**

- **The pickles**: `stats.pkl` (card database, 813 cards), `old_stats.pkl` (archived
  old card versions), `aliases.pkl` (name shortcuts). They ARE the database and they
  are **committed to git**. Constants `STATS_PKL` / `OLD_STATS_PKL` / `ALIAS_PKL` in
  `global_config.py:5-8`.
- **PICKLE commit**: a commit containing only pickle files, message literally
  `PICKLE`, produced by the bot itself via `/push` (`commands/update_bot.py:22`).
- **Dev machine**: where a human edits code and runs `git push`.
- **Running host**: whichever machine currently runs the bot (the owner's Mac or a
  CUDA box, at different times). It receives code via the bot's own `/pull` /
  `/update` Discord commands and is the only legitimate writer of pickles.
- **TTSCardMaker**: the separate repo of ~904 card PSDs (`MichaelJSr/TTSCardMaker`,
  cloned at `~/Desktop/TTSCardMaker`). Card artwork/text lives THERE, not here. Since 2026-10-19 it also
  holds the stoogedings icon font and the card website that reads this repo's pickles
  (**ttscardmaker-companion**).

## The four change classes and their gates

| Class | Examples | Ships how | Gate before shipping |
|---|---|---|---|
| (a) Code | any `.py`, `setup.py`, skills | `git push` from a dev machine → running host picks it up via `/pull` (+ `/restart_bot`) or `/update` from Discord | Checklist below if it adds a command; non-negotiables table always |
| (b) Data (pickles) | stats.pkl, old_stats.pkl, aliases.pkl | Mutated ONLY by bot commands (`/update_stats`, `/alias`, `/update_metadata`, `/mass_replace` — all four call the pickle save, psd_analyzer.py:1327/1399/1447/1534 as of 2026-10-20 + query_card's save_aliases); committed ONLY by `/push` or `/update` as a `PICKLE` commit | **diff_stats.py review before every pickle push** (see below) |
| (c) Config | config.py values (TOKEN, ADMINS, GIT_TOKEN, HF_TOKEN, model, vram_usage, ...) | Edited by hand per machine. NEVER committed (gitignored, `.gitignore:36` as of 2026-10-19). Values are secret — never print or quote them | New config *fields* route through autocroissant-config-and-flags |
| (d) Card content | a card's art, ability text, stats, folder; Types/ icons, the stoogedings font and its manifest | Edit in TTSCardMaker and push THERE; this repo only re-parses via `/update_stats` (and re-syncs emojis via `/sync_emojis`) | TTSCardMaker's own process; then the pickle gate applies to the re-parse. Renaming or deleting a Types/ PNG is a cross-repo change — ttscardmaker-companion §1.7 |

If your change mixes classes, split it. In particular: **never put pickle changes
and code changes in the same commit** — rationale in the next section.

## The self-update machinery (what /push, /pull, /update actually do)

All in `commands/update_bot.py` (registered in `main.py:171-204`). **The canonical
step-by-step anatomy — every message each command prints, the failure table, the
rebase fallback, and the restart mechanics — lives in autocroissant-run-and-operate
§5.** What change control needs you to know:

- **/push** → `git_push()` (update_bot.py): commits ONLY the three card pickles
  (`aliases.pkl stats.pkl old_stats.pkl`), message literally `PICKLE`, then
  `pull --rebase --autostash` onto new code (since 2026-10-20) and pushes, including a leftover
  unpushed PICKLE commit; "already up to date" only when nothing is ahead of origin.
- **/pull** → `git_pull()` (update_bot.py:33-101): fetch + merge — NOT the "hard
  reset" its Discord description claims (stale text; fixing descriptions is
  autocroissant-docs-and-style territory). On merge **CONFLICT** it auto-resolves per
  file — **pickles → `checkout --ours` (running host's data wins); everything else →
  `checkout --theirs` (remote dev-machine code wins)** — committed as an
  `AUTO-RESOLVE: ...` merge commit.
- **/update** → `update_bot()` (update_bot.py:122-147): stops the three send/edit
  queue loops first (main.py:200-202), then `/push` + `/pull` + restart. With
  `force_reset:True` it instead runs `git reset --hard origin/main`
  (update_bot.py:104-119) and restarts.

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
3. **`force_reset:True` is dangerous** — main.py:194 literally labels it
   `(dangerous!)`. It skips the pickle push entirely and hard-resets to
   `origin/main`, discarding ALL local state: uncommitted code edits AND every
   pickle change since the last successful /push (e.g. an afternoon of
   `/update_stats` and `/alias` work). Use it only to un-wreck a broken checkout,
   only after accepting that data loss, and ideally only right after a successful
   /push.

## Pickle-commit discipline (the data gate)

**The user-ranked #1 costliest failure class in this project is pickle data
corruption.** A committed-and-pushed bad pickle propagates to the other host on its
next /update — and, once TTSCardMaker's GitHub Pages site is enabled, to the public card
website within about 6 hours (it builds from the pickles on this repo's `main`;
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
`timestamp` (the signature of a sweep that used local mtimes, BUGS_AND_IMPROVEMENTS B6: only possible from
the archived traversal on branch `archive/local-update-stats` since local mode was archived on 2026-10-20), and old_stats.pkl
growing more than about 10% in one run (`git cat-file -s HEAD:old_stats.pkl; wc -c old_stats.pkl` —
diff_stats.py reads stats.pkl only). Interpretation guide and the other
four scripts: **autocroissant-diagnostics-and-tooling**. The full acceptance
procedure for parser changes (goldens → sandbox sweep → diff review):
**autocroissant-validation-and-qa**.

**Why reminder.pkl is gitignored while the other three pickles are committed**
(`.gitignore:39`, `REMIND_PKL` in global_config.py:8): stats/old_stats/aliases are
the shared card database that must stay consistent across the two hosting machines —
syncing them is the whole point of /push//update. reminder.pkl is volatile runtime
scheduling state that mutates whenever reminders fire or reschedule; committing it
would generate a conflict on every handoff. Accordingly, `git_push` adds only the
three card pickles and never reminder.pkl.

## What must NEVER be committed

Verified against `.gitignore` and `git ls-files`, 2026-07-11; line numbers re-read 2026-10-19 after an
(uncommitted) `.gitignore` update that added `reminder.pkl.tmp` and `.claude/settings.local.json` and
fixed the `*.pyd`/`*.dll` patterns:

| Path | .gitignore line | Why |
|---|---|---|
| `config.py` | 36 | Secret tokens (TOKEN, GIT_TOKEN, HF_TOKEN...). Per-machine. A leaked token means rotation, not deletion — git history is forever |
| `.ssh/` | 37 | Credentials, same reasoning |
| `reminder.pkl` | 39 | Volatile per-host scheduling state (see above) |
| `reminder.pkl.tmp` | 41 (added 2026-10-19) | Transient file from the atomic reminder save (2026-10-19 rewrite). Normally renamed away within milliseconds; a hard kill can leave one behind — delete it |
| `.claude/settings.local.json` | 52 (added 2026-10-19) | Personal Claude Code settings; the shared `.claude/skills/` library IS committed |
| `music/` | 44 | yt-dlp downloads; bulky, regenerable (dir kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite the `!music/.gitkeep` exception) |
| `models/` | 48 | Multi-GB diffusion .safetensors live here on AI-enabled machines (dirs kept by the tracked `models/README` / `models/loras/README` stubs — no `.gitkeep` exists despite the `!models/.gitkeep` exception) |
| `*.log`, `logs/`, `*.csv`, `*.txt` | 67-70 | Logs and the /export_cards //export_rulebook exports (`stats.csv`, `stats.txt`, `rules.txt`) land in the repo root; the globs keep them out |
| `*.so`, `*.c`, `*.cpp`, `*.pyd`, `*.dll`, `build/` | 7, 14-15, 19-22, 28 | Cython build artifacts; see autocroissant-build-and-env for the stale-.so trap |

**Trap inside that table:** `*.txt` is gitignored, yet `requirements.txt` and
`requirements2.txt` are tracked — only because already-tracked files ignore
.gitignore. Any NEW `.txt` (or `.csv`) file you create will be **silently skipped by
`git add .`**. If a new text file genuinely belongs in the repo, that is a
deliberate change-control decision: either force-add it or (better) amend
.gitignore in a reviewed code commit.

## Checklist: add a new slash command

Worked example to imitate: the `/update_stats` block at main.py:355-369 (2026-10-19).

1. **Write the worker function** in the right `commands/*.py` module (query logic in
   query_card.py, parsing in psd_analyzer.py, admin in management.py, etc.). Plain
   `def` is fine — it will be thread-wrapped. The body must follow the queue rules:
   never `await` Discord or call `interaction.followup` directly from worker code;
   send via `queue_message` / `queue_file` / `queue_edit` / `queue_command`
   (commands/utils.py:38-62). **autocroissant-architecture-contract** owns the
   full contract (queues, 15-minute interaction-token fallback, `perms_check`
   inversion — it returns True when the user LACKS permission, utils.py:306).
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
   `to_thread` (utils.py:68-77) wraps blocking functions in `asyncio.to_thread` and
   passes `async def` workers through unchanged, so the rebind line is correct for
   both. Keep the `func = to_thread(func)` line immediately BEFORE the decorated
   wrapper — that is the convention every block follows; see "looks safe but isn't"
   for why you must not delete or fold it.
   Admin-only commands start with the `if perms_check(interaction): ... return`
   block (copy it from /push, main.py:187-189). The newest example of a whole admin
   command block is `/sync_emojis` (main.py:749-762, added 2026-10-19).
3. **Add `@app_commands.describe`** for every parameter. Description wording and
   style: **autocroissant-docs-and-style**. Note Discord quirks that already shaped
   signatures here: big integers (seeds, IDs) are passed as `str` because Discord
   ints cap around 15 digits.
4. **Add a help entry** in `commands/help.py`: pick the right dict
   (`general_commands`:6, `text_commands`:22, `card_commands`:31,
   `stats_commands`:61, `ai_commands`:80, `music_commands`:102), key format
   `"/my_command <required> [optional]:"`, value = description string.
   `print_help` (help.py:126) dispatches on the category word.
5. **Decide Cython placement** if you created a NEW module: `setup.py` compiles only
   the modules listed in `INCLUDE_FILES` (setup.py:21-30); new files are uncompiled
   by default (the newest example: `commands/type_emojis.py`, 2026-10-19, deliberately
   left out). If you add yours to INCLUDE_FILES, the `wraparound=False` directive
   (setup.py:36) applies: **no negative indexing anywhere in the file** — use
   `x[len(x)-1]`, never `x[-1]` — and Cython 3.3 rejects re-annotating a variable
   that is already declared (declare it once; the management.py incident,
   autocroissant-failure-archaeology Entry 10.5). Compile-check rules: autocroissant-build-and-env §5. Do not add modules that use `execv`/dynamic
   registration (that is why update_bot.py and utils.py sit in EXCLUDE_FILES,
   setup.py:9-18).
6. **Ship and sync.** Push from the dev machine; on the running host run `/update`
   (or `/pull` + `/restart_bot`). Commands register with Discord at startup:
   `on_ready` (main.py:108-127) records every command in `slash_registry` (which is
   also what lets reminders invoke commands) and syncs per guild
   (`sync_guild_commands`, main.py:99-103). If the command does not appear after
   restart, run `/sync_global` (main.py:741). Client-side, Discord may need a
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
| 6 | **Pickles produced by the archived local traversal (branch `archive/local-update-stats`) are never pushed.** It exists for sandbox sweeps only; `/update_stats` is remote-only since 2026-10-20 (operating rule owned by autocroissant-run-and-operate §7). If such pickles ever reach the live checkout, they must pass the diff_stats.py gate AND an old_stats.pkl size check before any push. | History: until 2026-10-19 local mode no-op-stripped "TTSCardMaker" from absolute paths and would have misclassified every card (the removeprefix bug, fixed with `relpath`; archaeology Entry 1). Still true after the fix: with its default `use_local_timestamp=True`, local mtimes are compared with stored GitHub commit dates — 813/813 newer on 2026-10-19 — so a local run archives every card into old_stats.pkl, the cca0aaf/e5f5393 balloon pattern (BUGS_AND_IMPROVEMENTS B6). The slash default became remote on 2026-10-20 and the same day local mode was archived: `use_local_repo`/`use_local_timestamp` no longer exist on the command, and stored reminders drop them (`RETIRED_OPTIONS` in analytics.py) |

## Changes that look safe but aren't

- **Renaming or moving `CardInfo`/`CardStats` (psd_analyzer.py:123, :94), or renaming
  `commands/psd_analyzer.py`.** Pickle stores the class's import path; the committed
  stats.pkl/old_stats.pkl would fail to unpickle on every machine after the next
  pull. Schema/location changes require a data migration shipped in lockstep —
  precedent: the "Big massive refactor" (366c8d9, 2025-10-20) changed the schema and
  had to delete and rebuild the old pickle files in the same commit.
- **"Cleaning up" `x[len(x)-1]` to `x[-1]`** in any INCLUDE_FILES module. That idiom
  is deliberate (non-negotiable #1; dd800bc). Same for adding `.removeprefix()`-style
  "modernizations" without checking what the original expression did — that exact
  move created the local-path bug that lived from 2025-12-02 to 2026-10-19.
- **Deleting a "duplicate" function definition in main.py.** `slash_set_ratio` is
  defined twice (the second is /set_repo's handler) and `slash_delete_song` twice
  (find them with the provenance grep; line numbers drift). Both commands work because each decorator
  registered its function at def time; Python then silently rebinds the name.
  Renaming/merging/reordering these blocks changes what is registered. Wart detail:
  **autocroissant-architecture-contract**.
- **Removing or moving the `func = to_thread(func)` rebind lines** because they look
  redundant next to the decorator. The rebind is what makes a blocking worker
  awaitable; break it and the command fails (or blocks the event loop) only when
  first *invoked*, not at startup — so it survives a smoke test and dies in
  production. Keep rebind-then-decorator, one block per command.
- **Adding a new `.txt`/`.csv`/`.log` file and assuming git tracks it.** The
  gitignore globs (lines 67-70 as of 2026-10-19) silently exclude it (see the trap note above).
- **Adding a dependency to `requirements.txt` that belongs in `requirements2.txt`**
  (or vice versa). requirements.txt = core bot; requirements2.txt = AI enablement
  (torch, diffusers, etc., first line is the commented CUDA index URL). Breaking the
  split breaks non-negotiable #4.
- **Hand-committing pickles from a dev machine**, or `git add -A` after a run that
  touched them. Your data loses to `--ours` on the running host, and you have broken
  the PICKLE-only revert property (see merge-strategy implications).
- **Trusting /pull's own description.** It says "hard reset" but performs a merge;
  the actual reset is `/update force_reset:True` — and that one really does discard
  everything local (see force_reset danger above).
- **Editing card text or stats in this repo.** Ability text and stats live in
  TTSCardMaker PSDs (class (d)); anything you "fix" in stats.pkl by hand is
  overwritten by the next `/update_stats` and was never legitimate to begin with.
- **Renaming a Types/ icon PNG in TTSCardMaker "to fix a name".** Image-layer type names,
  font tokens, emoji names and website icons all key off the file stem; the font manifest
  keeps the old entry, so old tokens keep parsing while the card shows letters. Add an alias
  instead (ttscardmaker-companion §1.7).
- **Changing `ICON_MANIFEST_PATH` / `ICON_FONT_PREFIX` in psd_analyzer.py or `emoji_name` rules
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
re-verified 2026-10-19 (HEAD a7a4bb3 + uncommitted working tree). Line numbers drift;
re-verify with the commands below (run from `/Users/michaelsrouji/Desktop/AutoCroissant`)
before relying on any of them. If a check disagrees with this file, trust the check
and update this file.

| Fact | Re-verify with |
|---|---|
| PICKLE commits are the recent pickle history | `git log --oneline -3 -- stats.pkl old_stats.pkl aliases.pkl` |
| /push adds only the three pickles, message PICKLE | `grep -n "git.add\|PICKLE" commands/update_bot.py` |
| Merge strategy: pickles ours, code theirs | `grep -n -- "--ours\|--theirs" commands/update_bot.py` |
| force_reset behavior and its "dangerous!" label | `grep -n "force_reset" main.py commands/update_bot.py` |
| restart via startup.sh with execl fallback | `grep -n "execv\|execl" commands/update_bot.py` |
| Gitignore coverage (config.py, .ssh, reminder.pkl, music, models, txt/csv/log) | `git check-ignore -v config.py reminder.pkl music/x models/x a.txt a.csv a.log` |
| Tracked pickles + grandfathered requirements*.txt | `git ls-files '*.pkl' '*.txt'` |
| Incident hashes and their diffs | `git show --stat cca0aaf eb9aa84 366c8d9 f7c915c` ; `git show dd800bc` ; `git show 3bbaa2b 4bcee6b` |
| Cython include/exclude lists and wraparound=False | `grep -n "wraparound\|INCLUDE_FILES\|EXCLUDE_FILES" setup.py` |
| to_thread passes coroutines through; queue_* helpers | `grep -n "def to_thread\|def queue_" commands/utils.py` |
| /update_stats is remote-only; local mode archived (since 2026-10-20) | `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS` in analytics.py); `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived` |
| Rule 6 history: local-path fix lives on in the archive branch | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115); `git grep -c "removeprefix(" archive/local-update-stats -- archive/local_update_stats.py` (no output: the word only appears in the header's history note) |
| Stored reminders drop retired options | `grep -n -A6 "^RETIRED_OPTIONS" commands/analytics.py` |
| Duplicate handler names still present | `grep -n "def slash_set_ratio\|def slash_delete_song" main.py` |
| Registration pattern example block | `grep -n "update_stats = to_thread" main.py` |
| Help dict names and lines | `grep -n "_commands = {\|def print_help" commands/help.py` |
| Pickle filename constants | `grep -n "PKL" global_config.py` |
| diff_stats.py gate exists and documents its exit codes | `head -20 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py` |
| Commit count / current head (date-stamp anchor) | `git rev-list --count HEAD` ; `git log --oneline -1` |
