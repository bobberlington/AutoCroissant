---
name: autocroissant-failure-archaeology
description: The chronicle of every major AutoCroissant investigation, dead end, rejected fix, and revert — so nobody re-fights a settled battle. Load this whenever a task or question sounds like "why is this code like this", "has this been tried before", "who broke/fixed X", "when did X change", "revert", "regression", "history", "post-mortem", "why was X removed/deleted", or names any of: the removeprefix / UNKNOWN-type local-mode bug (fixed 2026-10-19 with relpath), the 2026-10-19 fix batch (scheduled /update_stats pinned remote, temp-PSD leak, .DS_Store type, force_update default, management.py Cython 3.3 compile error, reminder True→False parser, reminders running as the bot), whitespace collapsing in ability text, the old_stats.pkl balloon / duplicate archiving, the reverted PICKLE, TYPE/WHITESPACE MISMATCH, "am forehead", the Big massive refactor (366c8d9), cythonization (f7c915c), Text_Export.jsx, convert_nf4_flux.py, aliases.py, descriptions.pkl, metadata.pkl, requirements-linux. Every entry gives Symptom → Root cause → Evidence (commit hashes and diffs) → Status. Also load it BEFORE re-proposing an old idea or asking why odd-looking historical code exists (the act of cleaning it up is gated by autocroissant-change-control). Live-symptom triage belongs to autocroissant-debugging-playbook; making a new fix belongs to autocroissant-change-control / autocroissant-psd-extraction-campaign.
---

# AutoCroissant Failure Archaeology

Date of record: **2026-07-11**, extended **2026-10-19** (Entry 1 closed, Entry 10 added) and **2026-10-20** (Entries 10.10-10.11). Repo: `/Users/michaelsrouji/Desktop/AutoCroissant`, single branch `main`, 193 commits (HEAD a7a4bb3 "skills and memories") plus an uncommitted 2026-10-19 working-tree batch, no tests, no CI. Every claim below was verified against `git show` diffs, not commit messages — **messages here are sometimes jokes** (see "am forehead", Entry 8.4). When you extend this file, do the same.

Entry format, always:

- **Symptom** — what a user or the owner saw.
- **Root cause** — the mechanism, stated so it predicts all observations.
- **Evidence** — commit hashes, the exact diff lines, byte counts, measurements.
- **Status** — `fixed in <hash>` / `reverted in <hash>` / `OPEN` / `superseded by <hash or skill>`.

Read-only git commands (`git log`, `git show`, `git diff`) are always safe here. Never run mutating git commands or the bot itself while investigating.

## Index

| # | Battle | Status |
|---|---|---|
| 1 | The removeprefix local-mode bug (flagship) | **FIXED** 2026-10-19 (relpath; uncommitted working tree); local mode itself archived 2026-10-20 (Entry 10.12) |
| 2 | The Feb 8 2026 whitespace saga | settled by 3bbaa2b + 4bcee6b |
| 3 | The reverted PICKLE & duplicate archiving | fixed in e7befd5 (+4e03190) |
| 4 | TYPE/WHITESPACE MISMATCH check | removed in 081b1fd; smarter revival = campaign candidate |
| 5 | The Big Massive Refactor | settled (366c8d9) |
| 6 | The cythonization wave | settled; caused Entry 1 (now fixed) |
| 7 | Dead ends & deleted artifacts | each closed, see table |
| 8 | Smaller settled battles | each fixed, see subsections |
| 9 | Lore corrections (wrong attributions refuted by diffs) | reference |
| 10 | The 2026-10-19/20 fix batch (B2, C7, C8, C9, C11, Cython 3.3, reminder parser and privileges, .gitignore; B6 mitigated, then retired by archiving local mode; C13(a) fixed; C10 and other dead helpers archived) | fixed in the working tree, uncommitted as of 2026-10-20 |

---

## Entry 1 — The removeprefix local-mode bug (FLAGSHIP, FIXED 2026-10-19)

**Status: FIXED in the working tree on 2026-10-19** (not yet committed; record the hash here when it is). Was BUGS_AND_IMPROVEMENTS B1. For the 7+ months it was live, production data stayed healthy only because every update ran in remote mode. The story below is kept because the failure mode (a mechanical sweep changing semantics in an unexercised code path) will happen again.

### Symptom

Running `/update_stats` with `use_local_repo:True` — which was the **slash-command default** until 2026-10-20 (`main.py:365`; the default became False that day, Entry 10.10) — would have made every one of the 813 cards come back `UNKNOWN TYPE`, store wrong (absolute-style) paths, and mass-archive every card into `old_stats.pkl` as "path changed". One local-mode run corrupts `stats.pkl` at scale.

### Root cause (full causal chain)

Line numbers in this chain are the 2026-07-11 ones (pre-fix).

1. `update_stats` (psd_analyzer.py:1173) in local mode sets `local_path = expanduser(LOCAL_DIR_LOC)` (psd_analyzer.py:1207; `LOCAL_DIR_LOC = "~/Desktop/TTSCardMaker"` in global_config.py:4) — an **absolute** path.
2. `_process_local_files` walks it (`for folder, _, files in walk(local_path):`, psd_analyzer.py:991), so `full_path` is absolute.
3. psd_analyzer.py:1002 computes:
   ```python
   relative_path = full_path.removeprefix("TTSCardMaker").strip('/')
   ```
   `str.removeprefix` only strips a **leading** literal. An absolute path starts with `/Users/...`, not `TTSCardMaker`, so removeprefix is a **no-op** and `.strip('/')` leaves `Users/<you>/Desktop/TTSCardMaker/Creatures/...`.
4. `CardClassifier.classify` splits that on `/` and reads the top folder → `"Users"` → matches no classifier → **UNKNOWN type** for every card.
5. The stored path no longer equals the old repo-relative path → the path-change check archives the old CardInfo for **every** card → mass archiving + wrong paths saved.

Reproduce the string bug safely in pure Python (no repo state touched):

```bash
python3 -c "
from os.path import expanduser, relpath
p = expanduser('~/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd')
print('current  :', repr(p.removeprefix('TTSCardMaker').strip('/')))
print('pre-2025 :', repr(p.split('TTSCardMaker')[-1].strip('/')))
print('candidate:', repr(relpath(p, expanduser('~/Desktop/TTSCardMaker')))) "
```

Verified output 2026-07-11: `current` keeps the absolute prefix; `pre-2025` and `candidate` both give `Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd`.

### How it got in: the discipline rule CAUSED the regression

Cythonization commit **f7c915c** (2025-12-02 20:18, "cythonized the code") compiled psd_analyzer with `wraparound=False` (negative indexing banned) and mechanically purged every `[-1]`: the diff converts `folders[-1]` → `folders[len(folders) - 1]`, `bboxes[-1]` → `bboxes[len(bboxes) - 1]`, etc. This one line got a different treatment:

```diff
-                relative_path = full_path.split("TTSCardMaker")[-1].strip('/')
+                relative_path = full_path.removeprefix("TTSCardMaker").strip('/')
```

`split(x)[-1]` (take text AFTER the substring, anywhere in the string) is **not** equivalent to `removeprefix(x)` (strip x only if the string STARTS with it). The correct code had existed since the Big Massive Refactor (`git show 366c8d9:commands/psd_analyzer.py | grep -n 'relative_path ='` → line 909, split-based).

Verify: `git show f7c915c -- commands/psd_analyzer.py | grep -B5 -A5 removeprefix`

### The sister bug that proves the failure mode (and why this one survived)

The **same commit** f7c915c introduced a second UNKNOWN-type bug: `CardClassifier.classify` got `folders = dirname(relative_path)` — a *string* — so `top_folder = folders[0]` was a single character (`"C"`), classifying every card UNKNOWN **in both local and remote modes**. That one broke production and was found and fixed in 43 days (**fb47b5d**, 2026-01-15: `folders = relative_path.split('/')` + `name = folders.pop()`).

The removeprefix bug produces the identical symptom but **only in local mode**, which production never exercises — so it survived 10+ months (2025-12-02 → 2026-10-19). Lesson: a regression in an unexercised code path is invisible; the fix gate for parser changes is a measured sweep, not "it ran fine" (see autocroissant-validation-and-qa).

### Evidence that production data stayed healthy (measured 2026-07-11)

- All 813 `stats.pkl` paths are repo-relative (no `Users/` prefix), and the newest timestamp (Anubisath Guardian @ 2026-06-18) equals TTSCardMaker's latest commit date — i.e. timestamps came from the GitHub API, so real updates have been running **remote mode** (`use_local_repo:False`).
- Re-check anytime: `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` (read-only) → expect `813 cards`, `creature 297 ... creatures 7`, `1 problem card`, newest 2026-06-18 (counts drift as TTSCardMaker grows).

### The fix (2026-10-19)

`_process_local_files` computes (psd_analyzer.py:1226 on 2026-10-19; since local mode was archived on 2026-10-20
the function lives in `archive/local_update_stats.py` on branch `archive/local-update-stats`, line 115; main no
longer carries it):

```python
relative_path = relpath(full_path, local_path).replace('\\', '/')
```

This is exactly the candidate from the campaign skill's Phase 0. The `.replace` keeps paths `/`-separated if a
Windows host ever runs local mode.

**How it was verified** (2026-10-19; the sandbox was not kept):
- A sandboxed, forced LOCAL sweep ran on a copy of the repo and pickles in /tmp. Result: 813 cards, 0 typed
  UNKNOWN, "0 cards changed location", and types, paths and stats identical to the stored remote-sweep data.
- The only differences were 59 abilities and 2 star counts. The git-HEAD parser and the new parser gave identical
  output on every PSD (0 differences across all 813), so these 61 cards are PSDs edited after the bot's last sweep,
  not parser changes. That staleness is tracked as BUGS_AND_IMPROVEMENTS D5.

### Status and what replaced the operating rule

- **FIXED** (working tree, 2026-10-19). Signature to re-check: `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py`
  hits (:115) and `grep -rn removeprefix commands/` is empty.
- The old rule "always pass `use_local_repo:False`" was retired for the PATH reason only. A second local-mode hazard
  found the same day keeps remote mode the recommended default: with the default `use_local_timestamp:True`, local
  file mtimes are compared with stored GitHub commit times, and on 2026-10-19 all 813 local PSDs were newer. A
  default local run would therefore archive every card into old_stats.pkl (the Entry 3 balloon pattern). Tracked
  as BUGS_AND_IMPROVEMENTS B6. On 2026-10-20 the default itself became remote (Entry 10.10), and later that
  day local mode was archived out of the bot (Entry 10.12): `/update_stats` is GitHub-only. The current
  operating rule lives in autocroissant-run-and-operate §7.
- Scheduled runs were separately hardened (Entry 10.1).

---

## Entry 2 — The Feb 8 2026 whitespace saga

**Settled lesson: whitespace in ability text is load-bearing. NEVER collapse runs of spaces.** Gaps of 3+ spaces (`\s{3,}`, `_gap_pattern`, psd_analyzer.py:413 as of 2026-10-20) are where type icons visually sit — they are the *input signal* for type injection, and leftover gaps in stored text are *evidence* of undetected icons.

### Symptom

The Freezer and Mini Doomer parsed with junk spacing/quotes around injected `[type]` tokens. The first fix — collapsing whitespace — corrupted every multi-line and multi-space ability instead.

### The one-day chronology (8 commits, 2026-02-08, all verified by diff)

| Time | Commit | What the diff shows |
|---|---|---|
| 04:34 | **1c26747** "fix freezer and mini doomer cases" | Rewrote injection to before/after string surgery; stripped quotes; **added collapsing**: `ability_text = sub(r'\s{2,}', ' ', ability_text)` (and `_spacing_pattern` re-pointed from `\s+([:;,\.\?!])` to `\s{2,}`) |
| 05:33 | **d6d1278** | Collapsing narrowed to `sub(r'[ \t]{2,}', ' ', ...)` — because `\s` includes `\n`: the 04:34 version was **merging multi-line abilities onto one line**. Injection rewritten to segment-append (`new_line` list) |
| 06:08 | **c199084** | Quote-strip now also strips `\n`; prefix-space handling on injected tokens |
| 06:38 | **9bdbeb8** | `lstrip()` result lines; same `\n`-strip for the longest-text fallback |
| 06:38 | **d31b948** "PICKLE" | Data snapshot regenerated **with the collapsing code still live** |
| 16:23 | **c7b3191** "fix bug with cards that have no ability outputting an ability" | Removed the `longest_text` fallback entirely — no-ability cards had been assigned a random long text layer as their ability; now they get `NO ABILITY LAYER` |
| 16:33 | **3bbaa2b** "Fix the issue of collapsing spaces" | **Removed collapsing entirely**: deleted `self._spacing_pattern = re_compile(r'\s{2,}')` and deleted `ability_text = sub(r'[ \t]{2,}', ' ', ability_text)` |
| 19:12 | **4bcee6b** | Trailing-newline cleanup: final strip becomes `.strip(' \'"\n')`; leftover-types append no longer adds a trailing `\n` |

### Before / after (the load-bearing regex lines)

Pre-saga and post-saga, the **only** whitespace transformation applied to ability text is punctuation tightening, applied AFTER injection:

```python
ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text)   # psd_analyzer.py:727 as of 2026-10-20
```

The rejected transformations (do not reintroduce): `sub(r'\s{2,}', ' ', ...)` (1c26747, also ate newlines) and `sub(r'[ \t]{2,}', ' ', ...)` (d6d1278, still destroyed gap evidence and legitimate spacing).

### Status

Settled by **3bbaa2b + 4bcee6b**. Proof it works: The Freezer's stored ability is byte-identical to `parse_one.py "Field/1 Stars/The_Freezer.psd"` output, including the internal newlines (golden cards owned by autocroissant-validation-and-qa). Anyone proposing whitespace normalization in `_process_abilities` / `_inject_type_names` must read this entry first — it cost a full day and one mid-saga PICKLE.

---

## Entry 3 — The reverted PICKLE and the duplicate-archiving family

### Symptom

`old_stats.pkl` (the archive of superseded card versions) ballooned on routine updates — unchanged cards were being re-archived every run.

### Evidence: the old_stats.pkl byte trace (from `git cat-file -s <commit>:old_stats.pkl`)

| Commit | Date | Bytes | Event |
|---|---|---|---|
| 793abf1 | 2025-10-26 | 66 | first post-refactor archive |
| e380cee | 2025-11-03 | 3,196 | normal growth |
| **cca0aaf** | 2025-11-10 00:26 | **12,041** | balloon #1 — committed by the bot's own git identity ("Auto Croissant") via the self-update flow |
| **eb9aa84** | 2025-11-10 20:02 | 3,196 | owner **reverted** cca0aaf same day (`Revert "PICKLE"`; stats.pkl 305113→304623 too) |
| **637698b** | 2025-11-25 | **12,039** | balloon #2, near-identical size — **not** reverted |
| 76baa15 | 2026-01-15 | 26,054 | still growing |
| **e5f5393** | 2026-01-21 | **331,553** | balloon #3: mass re-parse after the bracket-format change (169a60a, 2026-01-19 "types in descriptions are now between brackets") re-archived essentially the whole DB — `force_update` had just been born (f94d95f, Jan 18) |
| 532b34f | 2026-02-07 | 77,075 | shrank — data rebuilt/cleaned after the guard fixes below; the exact cleanup procedure is not recorded in git (bot-identity PICKLE) |
| d31b948 | 2026-02-08 | 77,359 | mid-whitespace-saga PICKLE (Entry 2) — normal growth |
| 8753489 | 2026-03-18 | 77,780 | routine PICKLE — normal growth |
| 284d13c | 2026-06-20 | 82,881 | today: 218 names / 223 archived versions — stable for 4+ months |

### Root cause (mechanisms, from diffs)

Two compounding archive bugs:

1. **Double archive on path change** (present at the Nov 10 incident): `_check_for_path_change` archived the old CardInfo itself, *and* the main update block archived it again (`git show cca0aaf~1:commands/psd_analyzer.py` — archive at both ~line 721 and ~line 899). Every moved card → two archive copies per run.
2. **Archive-on-force** (created when `force_update` landed in f94d95f, Jan 18): forced runs sent EVERY card down the update branch (`if not force_update and not should_update: ... else:`), and the archive guard inside was only `if not is_new and name in self.db.stats:` — so every unchanged existing card got archived on every forced run.

### The fix

**e7befd5** (2026-01-31 00:58) tightened the guard in **both** the remote and local traversal paths:

```diff
-                if not is_new and name in self.db.stats:
+                if should_update and not is_new and name in self.db.stats:
```

Now archiving requires the card to have actually changed (timestamp newer or path changed), even under `force_update`. Trade-off accepted knowingly: a forced re-parse that changes *content* without a timestamp/path change is NOT archived.

Companion fixes the same month stopped the pickles being *saved* when nothing changed at all — see Entry 8.3/8.4 (fb47b5d + 4e03190).

### Status

Fixed in **e7befd5**; balloon #1 reverted in **eb9aa84**; old_stats stable since 2026-02 (77KB → 82.9KB over four months of routine PICKLEs). If old_stats jumps >10% in one PICKLE again, suspect a new archive-path bug and diff with `diff_stats.py` before pushing (autocroissant-change-control gate).

---

## Entry 4 — TYPE/WHITESPACE MISMATCH check: added, then removed

### Symptom

Six days in January 2026, the validator emitted `TYPE / WHITESPACE MISMATCH (N types, M gaps)` — and it flagged healthy cards while a separate injection bug was placing types at wrong offsets.

### Chronology (times matter — the check landed 32 minutes after the change that broke its premise)

| When | Commit | What the diff shows |
|---|---|---|
| 2026-01-24 17:01 | **0490195** "fix issue of types at beginning/end of ability not being injected" | Injection extended: detect leading/trailing whitespace, inject a leading type, middle types (reverse order), trailing type. **Bug shipped inside it**: middle-match indices were computed on `core_text`, but the leading `[type] ` was prepended *first*, shifting every subsequent `match.start()` — types landed at wrong offsets |
| 2026-01-24 17:33 | **fed8a83** "output a problem if num types does not match num gaps in ability" | Added: `if num_matches != num_types: if num_matches > 0 and num_types > 0: card.problems.append("TYPE / WHITESPACE MISMATCH (...)")` |
| 2026-01-30 21:49 | **081b1fd** "fix bug with types being improperly placed in abilities" | Full rewrite: `_sort_by_position` becomes 40px row-grouping (top-to-bottom then left-to-right); `_inject_type_names` becomes **line-by-line**; the MISMATCH problem block **deleted** in the same diff |

### Why it thrashed (diff-supported reconstruction)

1. **Legitimate mismatches exist.** A type at the start or end of a line consumes a type icon without any mid-text `\s{3,}` gap — the *exact* case 0490195 had just added support for. Whole-text gap counting therefore under-counts on healthy cards, so count inequality is not a reliable error signal.
2. **The concurrent placement bug polluted the data.** With 0490195's index-shift bug live, gap/type accounting was wrong on the very cards used to judge the check.
3. **The rewrite chose a forgiving invariant instead**: inject per line while types remain, then **append leftover types to the last line** rather than raise a problem — visible-but-mispositioned beats false alarms. Current code, psd_analyzer.py:820-824 (2026-10-20):

```python
        # Append remaining types to the last line
        if type_index < len(types) and result_lines:
            remaining = ' '.join(f"[{t}]" for t in types[type_index:])
            last_line_index = len(result_lines) - 1
            result_lines[last_line_index] = result_lines[last_line_index].rstrip('\n') + f" {remaining}"
```

Related same-evening hardening (2026-01-30/31): **34be7e5** consolidated the gap regex to `_gap_pattern = re_compile(r'\s{3,}')`; **d0bb28c** replaced the "top 400px" rule with `TYPE_REGION_RATIO = 0.5` (icons in the top half are creature types, bottom half are injection candidates); **dd800bc** converted a surviving `bboxes[-1]` to `bboxes[len(bboxes) - 1]` (Entry 6); **70daf60/cdfecea** iterated injection details.

### Status

**Removed in 081b1fd.** Restoring a *smarter* mismatch detector (e.g. bbox-anchored: compare icon x/y against text-line bboxes instead of counting gaps) is a ranked candidate in autocroissant-psd-extraction-campaign's solution menu — go there, not back to fed8a83's count comparison, and account for the legitimate-mismatch cases above.

---

## Entry 5 — The Big Massive Refactor (366c8d9, 2025-10-20)

- **Symptom/motive**: the pre-refactor bot was dict-pickles + ad-hoc modules (`metadata.pkl`, `descriptions.pkl`-era remnants, `channel_text.py`).
- **What the diff shows** (`git show 366c8d9 --stat`): 4,223 insertions / 1,752 deletions across 16 files; `channel_text.py` renamed → `analytics.py`; `management.py` born; and **stats.pkl, old_stats.pkl, metadata.pkl were DELETED from the repo** (`Bin → 0`) — a deliberate schema reset. Pickles were re-seeded fresh by the next PICKLE (793abf1, 2025-10-26, a 66-byte old_stats).
- **Consequences that still bind today**:
  - `stats.pkl` values are `CardInfo`/`CardStats` **@dataclasses defined in commands/psd_analyzer.py** → unpickling requires this repo on `sys.path` with `config.py` present (the import chain runs at load). Renaming/moving CardInfo, CardStats, or `commands/psd_analyzer.py` breaks every existing pickle — autocroissant-change-control class rules apply.
  - Any pickle from a checkout ≤ `366c8d9~1` is **legacy plain-dict format** and will not load through current code paths.
  - `metadata.pkl` never returned (its role folded into stats.pkl); `descriptions.pkl` was already dead (Entry 7).
- **Status**: settled architecture. Schema details are owned by impossibility-cards-reference; the pickles-in-git decision rationale by autocroissant-architecture-contract.

---

## Entry 6 — The cythonization wave (Dec 2025) and its long tail

- **f7c915c** (2025-12-02 20:18, "cythonized the code"): `setup.py` born (181 lines) — compiles command modules with `wraparound=False`, `boundscheck=False`, `cdivision=True` (setup.py:33-45); `cython` added to requirements.txt; `.gitignore` +73/−14 (87 changed lines) of build-artifact patterns; and a mechanical sweep converting every `x[-1]` to `x[len(x) - 1]` across frankenstein/psd_analyzer/query_card (negative indexing is banned under `wraparound=False`). **Cost of the sweep**: the removeprefix bug (Entry 1, fixed 2026-10-19) and the `dirname` classify bug (fixed fb47b5d) — both born in this one commit.
- **45de566** (same evening, 21:25, "cythonized more files"): moved `management.py`, `music_player.py`, `diffusion.py` from the exclude list INTO `INCLUDE_FILES` — their original wary comments ("Complex async operations with discord.py", "ThreadPoolExecutor with Process/Queue-like issues", "PyTorch callbacks and dynamic function registration") survive today as ironic annotations on the INCLUDE list (setup.py:27-29). Also converted music_player's remaining `[-1]`s.
- **Excluded from compilation, with reasons** (setup.py:9-18 comments, current): `main.py` (entry point), `config.py`/`global_config.py` (configuration), `commands/update_bot.py` (uses execv/execl for restart), `commands/utils.py` (core utilities, decorators, dynamic behavior), `__init__` files, setup.py itself. Net: **8 of 10 command modules compiled** as of 2026-07-11; since 2026-10-19 there are 11 command modules, because the new `commands/type_emojis.py` is not in INCLUDE_FILES (uncompiled by default).
- **dd800bc** (2026-01-30 23:15): a `bboxes[-1]` had crept back into psd_analyzer (`max_height = max(bboxes[-1][1].y // 3, card_mid_y)`); converted to `bboxes[len(bboxes) - 1]`. Nothing enforces the idiom automatically — it is maintained by hand, and this commit is proof stragglers happen. Do NOT "clean up" `x[len(x) - 1]` back to `x[-1]` in the 8 compiled modules.
- **bf9478e** (2026-01-26 17:02, "lazily import torch and diffusers"): closed the era — `import torch` at module top replaced by `get_torch()` / `torch_available()`, so the compiled+core bot runs with zero AI overhead. Doctrine and mechanics owned by autocroissant-ai-boundary.
- **Today (2026-10-19)**: no `.so`/`.c` artifacts exist in the working tree. The stale-`.so`-shadows-your-`.py` trap and build/clean procedure are owned by autocroissant-build-and-env.
- **Cython 3.3 tail (2026-10-19)**: management.py had silently stopped compiling under Cython 3.3 — Entry 10.5.
- **Status**: settled discipline; its one long-lived casualty (Entry 1) was fixed 2026-10-19. The meta-lesson: mechanical rule-enforcement sweeps need per-line semantic review — `split(x)[-1] ≠ removeprefix(x)`.

---

## Entry 7 — Dead ends and deleted artifacts

Recovered via `git log --diff-filter=D --name-only`. For each: what it was, why abandoned, what replaced it.

| Artifact | Born | Died | What it was / why abandoned / replacement |
|---|---|---|---|
| `venv/` (1,468 files, 380k lines) | 737f78e 2023-11-15 | **ed18602** 2023-11-15 | The whole virtualenv was committed on day one, deleted the same day (-380,450 lines, +4-line requirements.txt). Replacement: requirements files. |
| `aliases.py` | initial commits | **308e17c** 2023-11-15 | Hardcoded dict of Bloons filename aliases PLUS the original `"."`-prefix command dispatch table (`.help`, `.restart`, `.pull`...). Replaced by `aliases.pkl` (runtime-mutable via alias commands) — the oldest pickle lineage in the repo. |
| `commands/Text_Export.jsx` | 2a5a799 2023-11-17 | **7fc6f63** 2025-01-19 "In progress work on psd_analyzer" | A third-party **Photoshop ExtendScript** ("TextConvert.Export 1.1" by Bramus, customized with `write_stats_at_end`/`last_path`). The original extraction pipeline: open PSDs in Photoshop, batch-export text layers to text files, build the descriptions database from the exports. Abandoned once in-Python `psd-tools` parsing matured. Replacement: `PSDParser` in commands/psd_analyzer.py. |
| `descriptions/creatures.csv` | descriptions era | **c91d7b6** 2025-01-31 | CSV sidecar of the exported-text era. Same fate as above. |
| `descriptions.pkl` | 2a5a799 2023-11-17 | **bb58387** 2025-03-10 | The exported-text database itself. Fully superseded by `stats.pkl` as the single source. |
| `commands/convert_nf4_flux.py` | 1efe116 2024-08-20 | **f40152e** 2025-01-15 | 144 lines vendored from transformers internals (`_replace_with_bnb_linear`, `create_quantized_param`...) to hand-quantize Flux to NF4 before libraries supported it. Superseded the day diffusers/transformers shipped native `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` — used in current diffusion.py:262-287. |
| `requirements-linux.txt`, `requirements-linux2.txt` | b22dd42 2024-07-18 | **3f9a96d** 2024-08-18 | One-month experiment: Linux/CUDA-box dep lists pinning torch wheels via a Stanford mirror (`-f https://cs.stanford.edu/~nfliu/...`). Superseded by the single `requirements2.txt` with a pip index-URL line — commented out by **ca0b821** (2025-12-02) so plain installs are platform-neutral (CUDA box uncomments it). As of 2026-07-11 an **uncommitted** reorg makes requirements.txt core-only and requirements2.txt the full AI set — details owned by autocroissant-build-and-env. |
| `commands/tools.py` | 5c3b1ab 2024-07-14 (brought `to_thread`) | not dead | **Correction**: not deleted — **renamed** to `commands/utils.py` in f40152e (2025-01-15), `commands/{tools.py => utils.py}` with 0 content change. |
| `music/README`, `models/output/README` | — | b4af57e 2024-12-20, b22dd42 2024-07-18 | Placeholder READMEs to keep empty dirs in git; dropped once .gitignore/dir conventions settled. |
| `metadata.pkl` | fbc407b 2025-05-05 ("added the META pkl...") | **366c8d9** 2025-10-20 | Dict-pickle sidecar of the pre-refactor era; folded into CardInfo fields in stats.pkl (Entry 5). |

---

## Entry 8 — Smaller settled battles (each verified by diff)

### 8.1 Message splitting vs code fences

- **Symptom**: long messages with multiple ``` code blocks split mid-fence, breaking Discord rendering; then splitting crashed outright.
- **Root cause & fixes**: old `split_long_message` only handled a message that both started AND ended with a single fence. **e21b986** (2026-01-18) rewrote it with an `in_code` fence-tracking loop — but also removed the `max_length` default, breaking callers that passed no argument. **d699b3f** (2026-01-26) restored `max_length: int = BREAK_LEN`.
- **Status**: fixed; current `split_long_message` at utils.py:236, BREAK_LEN=1950. (e21b986 also introduced `ABILITY_EXCLUSIONS`, now psd_analyzer.py:838 as of 2026-10-20.)

### 8.2 Deploy items / card.types clobber + force_update born

- **Symptom**: item cards (the "deploy items") showed wrong types.
- **Root cause**: after per-layer processing correctly appended above-midline icons to `card.types` (psd_analyzer.py:507-510 as of 2026-10-20), a later block **overwrote** it with the below-midline bbox list: `card.types = [name for name, _ in type_bboxes if name]` — i.e. types became the inline-ability icons.
- **Fix**: **f94d95f** (2026-01-18) deleted the overwrite. Same commit: added `MISSPELT_CARD_TYPES = ['undread', 'tornado']` validator (now `+ 'error'`, psd_analyzer.py:45 as of 2026-10-19 — today's single problem card "20 Creature Types" trips `tornado`), and threaded the `force_update` parameter end-to-end (slash param born here; its interaction with archiving is Entry 3).
- **Status**: fixed in f94d95f.

### 8.3 Don't overwrite stats with empty stats

- **Symptom risk**: a failed or empty traversal (API error → zero files seen) left `dirty_files` empty, then the unconditional `prune_clean_cards(); save()` at the end of `update_stats` would archive EVERY card and save a gutted database.
- **Fix**: **fb47b5d** (2026-01-15 17:23) gated it: `if (num_updated > 0): prune_clean_cards(); save()`. (Same commit fixed the `dirname` classify bug — Entry 1's sister.)
- **Status**: fixed in fb47b5d, gate corrected 20 minutes later (8.4).

### 8.4 Save pickles only when something changed — the "am forehead" commit

- **Symptom**: even with 8.3's gate, pickles were re-saved (and clean-pruned) on every run with zero changes.
- **Root cause**: `num_updated` is the **progress counter** — it increments for every traversed file (`num_updated % UPDATE_RATE` drives progress messages) — so `num_updated > 0` was true on any non-empty traversal.
- **Fix**: **4e03190** (2026-01-15 17:43), commit message literally **"am forehead"**: the returned/gating counter switched to `num_new` (increments for new cards and content-updated cards, not moves, not unchanged). **2ff52b2** (18:55 same day) is only the annotation cleanup (`-> tuple[list[str], int]`, parenthesized returns).
- **Status**: fixed in 4e03190. This is the canonical example of why you read diffs, not messages: the commit that made the pickle-save gate real is titled "am forehead", and the commit lore had mis-attributed the fix (see Entry 9).

### 8.5 Alias case-insensitivity

- **Symptom**: aliases stored with original casing; lowercase lookups missed them.
- **Fix**: **626e7e4** (2025-12-18): alias keys AND targets lowercased both at insert (`alias_key = f"{alias.lower()}.png"`) and on load from aliases.pkl (dict rebuilt lowercased), plus case-insensitive `endswith` target matching (query_card.py).
- **Status**: fixed in 626e7e4.

### 8.6 Reminders reschedule if past due

- **Symptom**: recurring reminders whose fire time passed while the bot was offline never fired again (or fired stale).
- **Fix**: **105198f** (2025-12-02 — note: NOT Nov 3; see Entry 9): on init, any reminder with a `frequency` whose `when <= now` rolls forward `while when <= now: when += frequency`, then saves. One-shot past-due reminders are not rescheduled. Same commit introduced the shared `TIMEZONE` constant in global_config.py.
- **Status**: fixed in 105198f; the `while` loop was replaced on 2026-10-19 by a one-step jump to the next future slot (Entry 10.6). Reminder ops owned by autocroissant-run-and-operate.

### 8.7 Seed as string (Discord 15-digit int limit)

- **Symptom class**: Discord integer options cap around 15 digits; RNG seeds are 64-bit.
- **Settled convention**: `/ai`'s `seed` param is `Optional[str]`, converted with `int(seed)` + error reply. The in-code comment at main.py:514: "ints are limited to 15 digits or less by discord, so I need to take it as a str and then convert it to an int". Do not "fix" the param type back to int.

---

## Entry 9 — Lore corrections (claims you may hear that the diffs refute)

Earlier internal notes/briefs contain these errors. The diffs win. Do not propagate:

| Wrong claim | Diff-verified truth |
|---|---|
| "Save pickles only when changed = 2ff52b2 + 637698b" | = **fb47b5d** (gate, wrong counter) + **4e03190 "am forehead"** (correct counter `num_new`); 2ff52b2 is annotation cleanup only. **637698b is a PICKLE data commit** (2025-11-25, bot identity, old_stats 3,196→12,039 — balloon #2 in Entry 3) |
| "0490195 landed Jan 30" | 2026-01-**24** 17:01 — 32 minutes before fed8a83, which is why the MISMATCH check judged a freshly-changed injector (Entry 4) |
| "e21b986 and d699b3f landed Jan 24" | e21b986 = 2026-01-**18** 19:10; d699b3f = 2026-01-**26** 16:19 |
| "bf9478e (lazy torch) landed Jan 18" | 2026-01-**26** 17:02 |
| "105198f (reminders) landed 2025-11-03" | 2025-**12-02** 16:56 — same day as ca0b821 (16:58) and cythonization f7c915c (20:18) |
| "tools.py was deleted" | **renamed** to commands/utils.py in f40152e, content unchanged |
| "b22dd42/3f9a96d = 2024-08-11" | b22dd42 = 2024-**07-18**; 3f9a96d = 2024-**08-18** |
| "the whitespace saga = 7 commits" | 7 parser commits **+ 1 mid-saga PICKLE (d31b948)** = 8 commits on 2026-02-08 (Entry 2 table) |

---

## Entry 10 — The 2026-10-19/20 fix batch (uncommitted working tree)

These fixes landed together on 2026-10-19 alongside the stoogedings icon-font parsing (impossibility-cards-reference
§4.1), `commands/type_emojis.py` and the reminder rewrite. None had a commit hash when this was written: once
committed, replace "working tree" with the hash. Line numbers are 2026-10-19 readings.

### 10.1 Scheduled `/update_stats` inherited the dangerous default (was B2)

- **Symptom (latent)**: a reminder with `command:/update_stats` and no options ran local mode unattended, which
  while Entry 1 was live meant corrupting stats.pkl on a schedule.
- **Root cause**: reminders call the slash callback from `slash_registry`, so they get the slash defaults
  (`use_local_repo=True` at the time, main.py:365; the default became False on 2026-10-20, which makes
  this pin redundant but harmless).
- **Fix**: `SCHEDULED_DEFAULTS = {"update_stats": {"use_local_repo": False}}` (commands/analytics.py:38). It is
  applied when a reminder is created (the reply says "Added `use_local_repo:False`"; an explicit value is kept), when
  a pre-v2 reminder is migrated (forced, because old values were never checked), and at fire time (forced for
  legacy reminders with no creator).
- **Verified**: code read 2026-10-19. Still worth keeping after Entry 1's fix, because local mode with local
  timestamps re-archives everything (BUGS_AND_IMPROVEMENTS B6).
- **Superseded 2026-10-20** (Entry 10.12): with local mode archived the option no longer exists, so
  `SCHEDULED_DEFAULTS` was removed. `RETIRED_OPTIONS` now strips `use_local_repo`/`use_local_timestamp` from
  stored reminders instead.

### 10.2 Remote-mode temp PSD downloads leaked (was C7)

- **Symptom**: every remote `/update_stats` left one `urlretrieve` temp file per parsed PSD in the OS temp dir.
- **Fix**: `try: card = self.parser.parse(local_file, path)` / `finally: remove(local_file)` (psd_analyzer.py:1178-1181 on 2026-10-19; 1144-1147 on 2026-10-20).
- **Verified**: code read 2026-10-19. The next live remote sweep is the first real exercise.

### 10.3 `.DS_Store` became a 36th card type on macOS (was C8)

- **Symptom**: local type population (`_populate_types_from_local`) listed `.ds_store` as a type. Harmless, since
  no layer is named that, but it polluted `all_types`.
- **Fix**: `if not file.startswith('.')` (psd_analyzer.py:1079 on 2026-10-19; 1045 on 2026-10-20, in `_populate_types_from_local`, which the diagnostics scripts still use).
- **Verified**: the 2026-10-19 sandboxed local sweep (Entry 1) ran through this code with 0 type differences.

### 10.4 `update_stats()` Python default force-reparsed everything (was C9)

- **Symptom (trap)**: the Python default was `force_update=True` and only the slash wrapper passed False, so a bare
  programmatic call re-parsed all 813 cards.
- **Fix**: the Python default is now `force_update: bool = False` (psd_analyzer.py:1401 on 2026-10-19; 1289 on 2026-10-20), matching main.py:363.
  Code that wants a forced re-parse must now say `force_update=True` (the validation recipes already do).

### 10.5 management.py stopped compiling under Cython 3.3

- **Symptom**: `python3 setup.py build_ext --inplace` failed with
  `commands/management.py:246:12: 'target_message' redeclared`. `cythonize()` raises `CompileError` before
  `setup()` runs, and `clean_artifacts()` has already deleted old artifacts, so a host that tried to compile got NO
  compiled modules and ran plain Python without anyone noticing.
- **Root cause**: in `add_reactions`, the `else` branch re-declared the variable with an annotation
  (`target_message: Message | None = None`) after the `if` branch had assigned it. Plain Python accepts that;
  Cython 3.3 does not.
- **Fix**: declare `target_message: Message | None = None` once before the `try`.
- **Verified 2026-10-19**: all 8 INCLUDE_FILES modules compile with Cython 3.3.0 and setup.py's directives in a
  scratch build. Re-checked while writing this entry: HEAD's management.py raises `CompileError`, the working-tree
  one compiles to C. Consequence: hosts that silently ran uncompiled will compile from now on, so the stale-.so
  trap (autocroissant-build-and-env §5) now applies to them.

### 10.6 Reminder catch-up replayed every missed run

- **Symptom**: after a stall while running, the old `check_reminder` advanced a repeating reminder by one
  `frequency` per pass (`when += frequency`), so it fired once a second until it caught up. The startup path
  (Entry 8.6) used a `while when <= now` loop.
- **Fix**: `_next_run(when, frequency, now)` jumps to the first slot after now in one step (analytics.py:216),
  used at startup and in the scheduler.

### 10.7 The old reminder command parser turned True into False

- **Symptom**: a stored command like `/update_stats True True False` ran with every boolean False.
- **Root cause**: the old scheduler used `utils.parse_named_args` → `convert_value`, which checks
  `value.lower() in {"true", "false"}` and then returns `value == "true"`. Only lowercase `true` became True.
- **Fix**: analytics.py now has its own parser (`_tokenize`, `_bind`, `_convert`). Options are written as
  `name:value` and checked against the real slash signature. Pre-v2 reminders are read with the old boolean meaning
  (`legacy=True`), so they keep doing what they actually did, and are rewritten to canonical form once
  `slash_registry` is filled.
- **Deleted 2026-10-20 (was C10)**: `convert_value` and `parse_named_args` had no callers after the rewrite and were
  deleted from main. They were put on an archive branch first, but the owner had that branch deleted the same day
  (dead code is deleted, not archived). The code is still in a7a4bb3: `git show a7a4bb3:commands/utils.py`. See 10.13.

### 10.8 Reminders ran with the bot's own permissions

- **Symptom**: any scheduled command passed admin checks.
- **Root cause**: the fake interaction's user is the bot (`FakeUser` id 1011264680683110433 in
  `utils.make_fake_interaction`), and that id is in `ADMINS` in the dev Mac's config. `/set_reminder` was already
  admin-only, but a reminder kept admin power after its creator lost it. `/remove_reminder` had no gate at all.
  Also, a stored command that failed to parse hit `continue` before the reschedule step, so it re-fired its error
  message every second.
- **Fix**: reminders store `creator_id`/`creator_name` and fire as a `_ReminderUser` stand-in, so the slash
  callback's own `perms_check` runs again at fire time. Legacy reminders with no creator still run as the bot
  (`LEGACY_RUNS_AS_BOT = True`, analytics.py:37) and `/list_reminders` labels them. `/remove_reminder` is limited
  to admins or the creator. Each reminder is isolated (a crash disables only that one), and a bad command reports
  once per due slot.
- Operating detail: autocroissant-run-and-operate §9. Flow and invariant: autocroissant-architecture-contract.

### 10.9 .gitignore gaps (was C11, plus a Cython-era quirk)

- **Symptom (latent)**: the reminder rewrite's atomic save writes `reminder.pkl.tmp` in the repo root,
  which nothing ignored, so a crash-leftover could be committed by `git add -A`. Separately, since
  f7c915c the lines `*.so      # Linux/Mac`, `*.pyd     # Windows`, `*.dll     # Windows (rare)` matched
  nothing, because .gitignore has no trailing comments; `*.so` was covered by another line, `*.pyd` and
  `*.dll` by none.
- **Fix** (working tree, 2026-10-19): `reminder.pkl.tmp` added (line 41), the three patterns split from
  their comments, and `.claude/settings.local.json` ignored (personal settings; the skills stay committed).
- **Verified**: `git check-ignore -v reminder.pkl.tmp a.pyd a.dll` matches all three.

### 10.10 Local mode was the default, and local timestamps re-archive everything (B6, mitigated 2026-10-20, then retired by 10.12)

- **Symptom (latent, found 2026-10-19)**: with the path bug fixed, a default `/update_stats` (local mode,
  `use_local_timestamp:True`) would still have treated all 813 cards as updated: every local PSD's mtime
  was newer than its stored timestamp (a GitHub commit date), because the TTSCardMaker clone is shallow
  and freshly checked out. Each such card is re-parsed, its current version appended to old_stats.pkl
  (the Entry 3 balloon pattern) and its timestamp replaced by an mtime.
- **Root cause**: `_should_update_card` compares the stored timestamp with whatever timestamp the
  traversal supplies; local mtimes and commit dates are not comparable. Unchanged.
- **Mitigation (2026-10-20)**: the DEFAULT changed. `use_local_repo` now defaults to False in the slash
  command (main.py:365, describe string "Parse the local clone instead of GitHub (default False). Pair
  with use_local_timestamp:False.") and in `update_stats()` (psd_analyzer.py:1399); help.py says the same.
  Local mode is opt-in; the safe recipe is `git pull` the clone, then
  `use_local_repo:True use_local_timestamp:False`.
- **Verified**: the coordinator re-measured 813/813 newer mtimes and the 1-commit clone; defaults read
  from code 2026-10-20.
- **Superseded the same day** by Entry 10.12: the options were removed rather than defaulted.

### 10.11 Parser accepted token spellings the font draws as letters (was C13(a), fixed 2026-10-20)

- **Symptom (latent)**: `IconVocabulary.normalize` dropped whitespace, `_`, `-`, dots and apostrophes, so
  e.g. `[fragile.equip]`, `[fragile  equip]` or `[ fire]` set in stoogedings parsed as known icons with no
  problem recorded, while the font (which only accepts words joined by one space, `_`, `-` or nothing)
  drew them as letters on the card. Variants were stripped too, so `[fire: no]` was accepted.
- **Fix**: `IconVocabulary.spellings(text)` registers exactly the font's forms (words split on `[ _-]`,
  rejoined with " ", "_", "-" or ""), and `lookup` is an exact lowercase match; variants are matched
  without stripping spaces. Those examples are now `UNKNOWN ICON TOKEN`. `[N.M.E]` still works through
  the manifest alias `n.m.e`.
- **Verified 2026-10-20** (coordinator): the Photoshop test card parses as before, plus lookup checks.
- **Part (b), fixed the same day**: the manifest never deletes entries, so tokens for a renamed or deleted icon
  still parsed cleanly while the rebuilt font drew letters. `IconVocabulary.build` now skips manifest entries
  whose name is not in the current Types/ or Types/Stars/ listing (codepoint included). Verified with a fake
  removed entry (token and alias rejected, codepoint dropped) and the Photoshop test card (unchanged).

### 10.12 Local mode archived (2026-10-20)

- **Why**: local mode (`/update_stats use_local_repo:True`) was the source of Entry 1 (wrong paths, live
  2025-12-02 → 2026-10-19) and of B6 (Entry 10.10: with its default `use_local_timestamp:True` every card
  counts as updated). Production only ever ran remote mode, so the second traversal was a second set of bugs to
  maintain. The owner asked on 2026-10-20 to deprecate it and take it off the command.
- **What changed** (working tree): `/update_stats` and `update_stats()` lost `use_local_repo` and
  `use_local_timestamp` (main.py:361-365, psd_analyzer.py:1287); `traverse_local` and `_process_local_files`
  moved, as plain functions taking a `RepositoryTraverser`, to `archive/local_update_stats.py` with
  `update_stats_local()` for sandbox sweeps (never imported by the bot, not in INCLUDE_FILES). The same day,
  at the owner's request, that file left main for its own local branch, `archive/local-update-stats`
  (commit e2835a7 on a7a4bb3); main has no `archive/` folder. The offline
  helpers `_populate_types_from_local` and `_icons_from_local` stayed in psd_analyzer.py for parse_one.py and
  gap_trace.py. In analytics.py, `SCHEDULED_DEFAULTS` (Entry 10.1) became `RETIRED_OPTIONS` (analytics.py:41):
  stored reminders drop the two options at load (`REMINDER_VERSION` 2 → 3) and at fire time, pre-v2
  positional values are mapped by the old option order first, and a new reminder naming one gets a
  "Dropped ..." note instead of an error.
- **Verified 2026-10-20**: all 8 INCLUDE_FILES compile with Cython 3.3.0; an offline migration test with fake
  reminders (`/update_stats use_local_repo:False` → `/update_stats`; legacy `/update_stats True False True True`
  → `output_problematic_cards:False force_update:False`, the old parser reading capitalised `True` as False;
  `/sync_emojis refresh:True` unchanged); the archived traversal parsed a one-card temp clone with a
  repo-relative path.
- **Status**: done (working tree, uncommitted). Revive local mode only after fixing its change signal (B6).

---

### 10.13 Dead helpers deleted (2026-10-20)

- **Why**: the owner first asked that deprecated code go to archive branches. Later the same day they had the two
  branches made for it deleted, along with the code. Rule now (autocroissant-architecture-contract, "Retiring a
  feature"): dead code with no callers is deleted; archive branches are only for retired features. A vulture scan of main.py, global_config.py, commands/ and the skill scripts (callers
  included), plus grep for every call site, found these with no callers:
  - `utils.convert_value` and `utils.parse_named_args`: the old scheduler's argument parser (10.7, was C10).
  - `utils.url_to_pilimage`, `utils.cv2_to_pil`, `utils.pil_to_cv2`: image conversions from the diffusion work. The
    bot uses only `url_to_cv2image`/`cv2discordfile` (/frankenstein) and `pildiscordfile` (/ai).
  - `CardRepository._exact_match_pattern` (query_card.py `__post_init__`): compiled in 366c8d9 (Entry 5) and
    never read. It was broken too: the string isn't an f-string, so it matched the literal text `{pattern}`.
  - All of these are deleted. Old code: `git show a7a4bb3:commands/utils.py` and
    `git show a7a4bb3:commands/query_card.py`. The branches `archive/old-scheduler-arg-parser` (3d3b571) and
    `archive/unused-helpers` (3b2effd) were deleted on the owner's request.
- **What changed** (working tree): the functions, the attribute, and the imports only they used (`cvtColor`,
  `COLOR_BGR2RGB`, `COLOR_RGB2BGR`, `lru_cache`) were deleted. `convert_value` carried `@lru_cache(maxsize=128)`.
  Watch for that when cutting functions: the decorator must go with the function, or it silently decorates the next
  one (here it would have cached `split_long_message`).
- **Verified 2026-10-20**: all 8 INCLUDE_FILES compile with Cython 3.3.0 using setup.py's COMPILER_DIRECTIVES. The
  compiled and source modules import offline (update_bot and music_player need GitPython and yt-dlp, which the
  scratch venv lacked). grep finds no remaining references.
- **Kept on purpose** (reachable, or used by tooling): gap/type injection (fallback for cards not in the icon font),
  `RETIRED_OPTIONS` and the reminder migration, `_populate_types_from_local`/`_icons_from_local` (diagnostics
  scripts), `global_config.LOCAL_DIR_LOC` (diagnostics scripts), every slash command and the AI and music features.
  `analytics.parse_time_str` (a one-line public wrapper of `_parse_when`, written in the rewrite and uncalled) was
  left in place as a candidate.

## How to add an entry

Use this template (the evidence bar — one mechanism must explain ALL observations including negatives, numbers predicted before measured — is owned by autocroissant-research-methodology; retired ideas from the idea lifecycle land here too):

```markdown
## Entry N — <name> (<date>)
- **Symptom**: what was observed, by whom, in which mode/command.
- **Root cause**: the mechanism. It must explain every observation, including "why didn't X also break".
- **Evidence**: commit hashes + the exact diff lines (`git show <hash> -- <file>`), byte counts,
  measurements, script outputs. Quote diffs, never trust commit messages.
- **Status**: fixed in <hash> / reverted in <hash> / OPEN / superseded by <hash or skill-name>.
```

Rules: date-stamp volatile facts; label unproven fixes "candidate"; if the entry closes an OPEN item, update the Index table and Entry 1-style operating rules; if it retires an idea, say what replaced it. Keep the "one home per fact" discipline — summarize + point to sibling skills rather than duplicating them.

## When NOT to use this skill

- **A live symptom to triage right now** ("bot won't start", "stats look wrong today") → **autocroissant-debugging-playbook**. Come back here only to check whether the battle was already fought.
- **You are about to make/ship a fix** → **autocroissant-change-control** (gates, pickle discipline) and **autocroissant-psd-extraction-campaign** (the executable plan for parser work; its Phase 0 was Entry 1's fix, done 2026-10-19).
- **Operating the bot** (running /update_stats safely, machine handoff) → **autocroissant-run-and-operate**.
- **How the parser/schema works today** (not how it got here) → **impossibility-cards-reference**; system design/invariants → **autocroissant-architecture-contract**.
- **Evidence standards for new investigations** → **autocroissant-research-methodology**; measurement scripts → **autocroissant-diagnostics-and-tooling**.

## Provenance and maintenance

Written 2026-07-11 against HEAD = 284d13c ("PICKLE", 2026-06-20). Extended 2026-10-19 against HEAD = a7a4bb3 ("skills and memories", which also committed the requirements split) plus the uncommitted 2026-10-19 working-tree batch (Entry 10). Re-verification one-liners for every volatile fact (cwd = repo root):

| Fact | Re-verify with |
|---|---|
| New history since this file | `git log --oneline -5` (anything above a7a4bb3 → review entries; a commit containing the Entry 10 batch → record its hash in Entries 1 and 10) |
| Entry 1 fixed | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115 on the archive branch; was psd_analyzer.py:1226 before the archive) and `grep -rn removeprefix commands/` (expect empty); once committed, `git log -S removeprefix --oneline -- commands/psd_analyzer.py` shows the fix commit above f7c915c |
| Entry 1 origin diff | `git show f7c915c -- commands/psd_analyzer.py \| grep -B5 -A5 removeprefix`; old code: `git show 366c8d9:commands/psd_analyzer.py \| grep -n 'relative_path ='` |
| Local mode archived (Entry 10.12) | `grep -rn 'use_local' main.py commands/` (only `RETIRED_OPTIONS`, analytics.py:41-44); `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived`; `grep -n '^REMINDER_VERSION' commands/analytics.py` (3) |
| Dead helpers deleted (Entry 10.13) | `git branch --list 'archive/*'` (one branch: archive/local-update-stats); `grep -rn 'convert_value\|parse_named_args\|url_to_pilimage\|cv2_to_pil\|pil_to_cv2\|_exact_match_pattern' main.py commands/` (expect empty) |
| Exact token spellings (Entry 10.11) | `grep -n -A8 "def spellings" commands/psd_analyzer.py` |
| Entry 10 fixes present | `grep -n 'remove(local_file)\|startswith(.\.)\|force_update: bool = False' commands/psd_analyzer.py` (1147, 1045, 1000/1289); `grep -n '^RETIRED_OPTIONS\|^LEGACY_RUNS_AS_BOT\|^def _next_run' commands/analytics.py` (41, 37, 221); `grep -n 'target_message: Message' commands/management.py` (one hit, before the `try`) |
| B6 hazard (only relevant if local mode is revived from the archive) | unpickle stats.pkl with stub classes and compare each `card.timestamp` with `os.path.getmtime(~/Desktop/TTSCardMaker/<card.path>)`: 813/813 newer on 2026-10-19 |
| Data still healthy / counts | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` (813 cards, 218/223 old, newest 2026-06-18 — all drift with TTSCardMaker) |
| old_stats byte trace | `for c in 793abf1 e380cee cca0aaf eb9aa84 637698b 76baa15 e5f5393 532b34f d31b948 8753489 284d13c; do echo "$c $(git cat-file -s $c:old_stats.pkl)"; done` |
| Whitespace-saga diffs | `git show 1c26747 3bbaa2b 4bcee6b -- commands/psd_analyzer.py`; surviving cleanup line: `grep -Fn "ability_text = sub" commands/psd_analyzer.py` (:730 on 2026-10-20) |
| MISMATCH lifecycle | `git show fed8a83 \| grep -A8 MISMATCH`; `git show 081b1fd \| grep -B3 MISMATCH`; leftover-append: `grep -n 'Append remaining types' commands/psd_analyzer.py` (:823 on 2026-10-20) |
| Archive guard | `git show e7befd5`; current: `grep -n 'should_update and not is_new' commands/psd_analyzer.py` |
| Save gate ("am forehead") | `git show 4e03190 \| grep 'num_new > 0'`; `git show fb47b5d \| grep 'num_updated > 0'` |
| Deleted artifacts list | `git log --diff-filter=D --name-only --format='DELETED-IN %h %ad %s' --date=short \| head -40` |
| Cython include/exclude + directives | `sed -n '9,45p' setup.py`; straggler check: `grep -rn '\[-1\]' commands/psd_analyzer.py commands/query_card.py commands/frankenstein.py commands/music_player.py` |
| Line numbers cited (drift-prone) | psd_analyzer.py: `grep -n 'def update_stats\|_gap_pattern = \|MISSPELT_CARD_TYPES = \|ABILITY_EXCLUSIONS' commands/psd_analyzer.py` → 1287 / 416 / 45 / 841 as of 2026-10-20 (`relpath(full_path` and `expanduser(LOCAL_DIR_LOC)` left with local mode; the 2026-07-11 values quoted in Entries 1-8 were relpath-line 1002 / 1173 / expanduser 1207 / 348 / 42 / 649) |
| Commit dates quoted | `git log --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M' --no-walk <hash>...` |
