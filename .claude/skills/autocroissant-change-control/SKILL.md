---
name: autocroissant-change-control
description: How changes are classified, gated, and shipped in AutoCroissant. Load this BEFORE committing, pushing, merging, or reverting anything; before adding, renaming, or removing a slash command; before touching the pickles (stats.pkl, old_stats.pkl, aliases.pkl) or running /push, /pull, /update, or /update_stats; before editing requirements.txt/requirements2.txt, .gitignore, or setup.py INCLUDE_FILES; before renaming or moving CardInfo/CardStats or any commands/*.py module; and whenever you are tempted to "clean up" odd-looking code (x[len(x)-1], double spaces, duplicate function names, the func = to_thread(func) lines). Contains: the four change classes and their gates, the /push//pull//update self-update anatomy and its pickles-ours/code-theirs merge strategy, the PICKLE commit convention and the diff_stats.py gate, the never-commit list, the add-a-slash-command checklist, a non-negotiables table with the historical incident behind each rule, and a list of changes that look safe but aren't.
---

# AutoCroissant change control

How changes get classified, gated, and shipped in this repo. AutoCroissant is a
personal Discord bot with **no tests, no CI, no README** — the rules below are the
entire safety system, and every one of them exists because something broke.
Facts and line numbers verified 2026-07-11 (repo at commit 284d13c, 192 commits).

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
  cloned at `~/Desktop/TTSCardMaker`). Card artwork/text lives THERE, not here.

## The four change classes and their gates

| Class | Examples | Ships how | Gate before shipping |
|---|---|---|---|
| (a) Code | any `.py`, `setup.py`, skills | `git push` from a dev machine → running host picks it up via `/pull` (+ `/restart_bot`) or `/update` from Discord | Checklist below if it adds a command; non-negotiables table always |
| (b) Data (pickles) | stats.pkl, old_stats.pkl, aliases.pkl | Mutated ONLY by bot commands (`/update_stats`, `/alias`, `/update_metadata`, `/mass_replace` — all four call the pickle save, psd_analyzer.py:1224/1296/1344/1431 + query_card's save_aliases); committed ONLY by `/push` or `/update` as a `PICKLE` commit | **diff_stats.py review before every pickle push** (see below) |
| (c) Config | config.py values (TOKEN, ADMINS, GIT_TOKEN, HF_TOKEN, model, vram_usage, ...) | Edited by hand per machine. NEVER committed (gitignored, `.gitignore:34`). Values are secret — never print or quote them | New config *fields* route through autocroissant-config-and-flags |
| (d) Card content | a card's art, ability text, stats, folder | Edit the PSD in TTSCardMaker and push THERE; this repo only re-parses via `/update_stats` | TTSCardMaker's own process; then the pickle gate applies to the re-parse |

If your change mixes classes, split it. In particular: **never put pickle changes
and code changes in the same commit** — rationale in the next section.

## The self-update machinery (what /push, /pull, /update actually do)

All in `commands/update_bot.py` (registered in `main.py:171-204`). **The canonical
step-by-step anatomy — every message each command prints, the failure table, the
rebase fallback, and the restart mechanics — lives in autocroissant-run-and-operate
§5.** What change control needs you to know:

- **/push** → `git_push()` (update_bot.py:16-30): commits ONLY the three card pickles
  (`aliases.pkl stats.pkl old_stats.pkl`), message literally `PICKLE`, then pushes —
  nothing else, ever.
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
next /update.

Before ANY pickle push (i.e. before running /push or /update after anything touched
the card DB), run the diff gate from the repo root:

```
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
```

Exit 0 = no changes; exit 1 = changes found — read them; exit 2 = load error (worse:
the working pickle may be unreadable). Red flags: cards "removed" that nobody
deleted, mass path changes, type counts shifting. Interpretation guide and the other
four scripts: **autocroissant-diagnostics-and-tooling**. The full acceptance
procedure for parser changes (goldens → sandbox sweep → diff review):
**autocroissant-validation-and-qa**.

**Why reminder.pkl is gitignored while the other three pickles are committed**
(`.gitignore:37`, `REMIND_PKL` in global_config.py:8): stats/old_stats/aliases are
the shared card database that must stay consistent across the two hosting machines —
syncing them is the whole point of /push//update. reminder.pkl is volatile runtime
scheduling state that mutates whenever reminders fire or reschedule; committing it
would generate a conflict on every handoff. Accordingly, `git_push` adds only the
three card pickles and never reminder.pkl.

## What must NEVER be committed

Verified against `.gitignore` and `git ls-files`, 2026-07-11:

| Path | .gitignore line | Why |
|---|---|---|
| `config.py` | 34 | Secret tokens (TOKEN, GIT_TOKEN, HF_TOKEN...). Per-machine. A leaked token means rotation, not deletion — git history is forever |
| `.ssh/` | 35 | Credentials, same reasoning |
| `reminder.pkl` | 37 | Volatile per-host scheduling state (see above) |
| `music/` | 40 | yt-dlp downloads; bulky, regenerable (dir kept by the tracked `music/.gitignore` stub — no `.gitkeep` exists despite the `!music/.gitkeep` exception) |
| `models/` | 44 | Multi-GB diffusion .safetensors live here on AI-enabled machines (dirs kept by the tracked `models/README` / `models/loras/README` stubs — no `.gitkeep` exists despite the `!models/.gitkeep` exception) |
| `*.log`, `logs/`, `*.csv`, `*.txt` | 60-63 | Logs and the /export_cards //export_rulebook exports (`stats.csv`, `stats.txt`, `rules.txt`) land in the repo root; the globs keep them out |
| `*.so`, `*.c`, `*.cpp`, `build/` | 7, 14-28 | Cython build artifacts; see autocroissant-build-and-env for the stale-.so trap |

**Trap inside that table:** `*.txt` is gitignored, yet `requirements.txt` and
`requirements2.txt` are tracked — only because already-tracked files ignore
.gitignore. Any NEW `.txt` (or `.csv`) file you create will be **silently skipped by
`git add .`**. If a new text file genuinely belongs in the repo, that is a
deliberate change-control decision: either force-add it or (better) amend
.gitignore in a reviewed code commit.

## Checklist: add a new slash command

Worked example to imitate: the `/update_stats` block at main.py:351-365.

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
   block (copy it from /push, main.py:183-186).
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
   by default. If you add yours to INCLUDE_FILES, the `wraparound=False` directive
   (setup.py:36) applies: **no negative indexing anywhere in the file** — use
   `x[len(x)-1]`, never `x[-1]`. Do not add modules that use `execv`/dynamic
   registration (that is why update_bot.py and utils.py sit in EXCLUDE_FILES,
   setup.py:9-18).
6. **Ship and sync.** Push from the dev machine; on the running host run `/update`
   (or `/pull` + `/restart_bot`). Commands register with Discord at startup:
   `on_ready` (main.py:106-124) records every command in `slash_registry` (which is
   also what lets reminders invoke commands) and syncs per guild
   (`sync_guild_commands`, main.py:97-101). If the command does not appear after
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
| 1 | **No negative indexing in cythonized modules** (anything in setup.py INCLUDE_FILES). Write `x[len(x)-1]`. | Cythonization (f7c915c, 2025-12-02) set `wraparound=False`, making `x[-1]` undefined behavior in compiled code. dd800bc (2026-01-30) is a one-line commit converting `bboxes[-1]` → `bboxes[len(bboxes)-1]`. The same mechanical negative-index purge in f7c915c also introduced the live `removeprefix` path bug — full story in **autocroissant-failure-archaeology** |
| 2 | **Never collapse runs of whitespace in ability text.** | Gaps of 3+ spaces are where type icons sit; they are the injection signal for `_inject_type_names`. The 2026-02-08 whitespace saga (8 commits in one day — 7 parser commits plus a mid-saga PICKLE — 1c26747 through 3bbaa2b/4bcee6b; chronology in autocroissant-failure-archaeology Entry 2) added `\s{2,}` collapsing to fix two cards, broke others, and ended with 3bbaa2b "Fix the issue of collapsing spaces" deleting the collapsing entirely. Whitespace is load-bearing |
| 3 | **All Discord sends from worker code go through `queue_*`.** No direct awaits of Discord from workers. | Single event loop + deferred dispatch with a fallback chain that survives the 15-minute interaction-token expiry. Detail owned by **autocroissant-architecture-contract** |
| 4 | **AI stays behind the boundary.** Core bot must run with only requirements.txt installed; torch-family deps go in requirements2.txt only; heavy imports stay lazy (pattern since bf9478e, 2026-01-26). | The diffusion capability is an experiment by owner doctrine; the bot must run perfectly without it. Verified 2026-07-11: clean venv with requirements.txt only → `import main` succeeds, torch absent. Detail owned by **autocroissant-ai-boundary** |
| 5 | **Every pickle push gets a diff_stats.py review first.** | cca0aaf (2025-11-10) pushed a bad snapshot (old_stats.pkl 3196 → 12041 bytes, duplicate archiving); it shipped because nobody could see what changed inside a binary blob. Reverted same day (eb9aa84). diff_stats.py turns "looks fine" into numbers |
| 6 | **Until the removeprefix bug is fixed, run `/update_stats` with `use_local_repo:False`** (as of 2026-07-11 the bug is live and unfixed; the slash default is True — main.py:361 — which is the dangerous value). | `_process_local_files` (psd_analyzer.py:1002) no-op-strips "TTSCardMaker" from absolute paths, so one local-mode run would misclassify every card and corrupt stats.pkl at scale. Any local-mode run must be followed by the diff_stats.py gate before any push. Root-cause story owned by **autocroissant-failure-archaeology**; the fix itself is a candidate code change that routes through this skill's class (a) process |

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
  move created the live bug.
- **Deleting a "duplicate" function definition in main.py.** `slash_set_ratio` is
  defined twice (main.py:303 and :313 — the second is /set_repo's handler) and
  `slash_delete_song` twice (:687, :694). Both commands work because each decorator
  registered its function at def time; Python then silently rebinds the name.
  Renaming/merging/reordering these blocks changes what is registered. Wart detail:
  **autocroissant-architecture-contract**.
- **Removing or moving the `func = to_thread(func)` rebind lines** because they look
  redundant next to the decorator. The rebind is what makes a blocking worker
  awaitable; break it and the command fails (or blocks the event loop) only when
  first *invoked*, not at startup — so it survives a smoke test and dies in
  production. Keep rebind-then-decorator, one block per command.
- **Adding a new `.txt`/`.csv`/`.log` file and assuming git tracks it.** The
  gitignore globs (lines 60-63) silently exclude it (see the trap note above).
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

## When NOT to use this skill

- Diagnosing why something is broken → **autocroissant-debugging-playbook**.
- The history/root-cause behind an incident cited here (removeprefix bug, eb9aa84
  revert, whitespace saga) → **autocroissant-failure-archaeology**.
- Queue/threading/registration internals and invariants → **autocroissant-architecture-contract**.
- Running, restarting, syncing, or handing the bot between machines → **autocroissant-run-and-operate**.
- diff_stats.py and the other scripts' usage/outputs → **autocroissant-diagnostics-and-tooling**.
- Evidence standards and the parser-change acceptance procedure → **autocroissant-validation-and-qa**.
- Adding/altering config fields or flags → **autocroissant-config-and-flags**.
- Cython build mechanics, environments, dependency installs → **autocroissant-build-and-env**.
- Anything AI/diffusion-side → **autocroissant-ai-boundary**.
- Command-description and help-text wording → **autocroissant-docs-and-style**.
- Card semantics (folders, layers, injection) → **impossibility-cards-reference**.

## Provenance and maintenance

All file:line references verified 2026-07-11 at commit 284d13c. Line numbers drift;
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
| /update_stats slash default use_local_repo=True | `grep -n "use_local_repo" main.py` |
| removeprefix live bug still present (rule 6 stands until this greps empty) | `grep -n "removeprefix" commands/psd_analyzer.py` |
| Duplicate handler names still present | `grep -n "def slash_set_ratio\|def slash_delete_song" main.py` |
| Registration pattern example block | `grep -n "update_stats = to_thread" main.py` |
| Help dict names and lines | `grep -n "_commands = {\|def print_help" commands/help.py` |
| Pickle filename constants | `grep -n "PKL" global_config.py` |
| diff_stats.py gate exists and documents its exit codes | `head -20 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py` |
| Commit count / current head (date-stamp anchor) | `git rev-list --count HEAD` ; `git log --oneline -1` |
