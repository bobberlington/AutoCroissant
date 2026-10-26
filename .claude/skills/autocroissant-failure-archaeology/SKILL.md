---
name: autocroissant-failure-archaeology
description: "The chronicle of every major AutoCroissant investigation, dead end, rejected fix, and revert — so nobody re-fights a settled battle. Load this whenever a task or question sounds like \"why is this code like this\", \"has this been tried before\", \"who broke/fixed X\", \"when did X change\", \"revert\", \"regression\", \"history\", \"post-mortem\", \"why was X removed/deleted\", or names any of: the removeprefix / UNKNOWN-type local-mode bug (fixed 2026-10-19 with relpath), the 2026-10-19 fix batch (scheduled /update_stats pinned remote, temp-PSD leak, .DS_Store type, force_update default, management.py Cython 3.3 compile error, reminder True→False parser, reminders running as the bot), whitespace collapsing in ability text, the old_stats.pkl balloon / duplicate archiving, the reverted PICKLE, TYPE/WHITESPACE MISMATCH, \"am forehead\", the Big massive refactor (366c8d9), cythonization (f7c915c), Text_Export.jsx, convert_nf4_flux.py, aliases.py, descriptions.pkl, metadata.pkl, requirements-linux. Every entry gives Symptom → Root cause → Evidence (commit hashes and diffs) → Status. Also load it BEFORE re-proposing an old idea or asking why odd-looking historical code exists (the act of cleaning it up is gated by autocroissant-change-control). Live-symptom triage belongs to autocroissant-debugging-playbook; making a new fix belongs to autocroissant-change-control / autocroissant-psd-extraction-campaign."
---

# AutoCroissant Failure Archaeology

Date of record: **2026-07-11**, extended **2026-10-19** (Entry 1 closed, Entry 10 added), **2026-10-20** (Entries 10.10-10.28), **2026-10-21** (10.29-10.30) and **2026-10-25** (10.32: the pickles retired; the entries before it describe the pickle era). Repo: `/Users/michaelsrouji/Desktop/AutoCroissant`, branch `main` (193 commits at a7a4bb3 when Entry 10 was written) plus archive branches `archive/ai-generation` and `archive/local-update-stats`, no CI (an offline pytest suite in `tests/` since 2026-10-20). Every claim below was verified against `git show` diffs, not commit messages — **messages here are sometimes jokes** (see "am forehead", Entry 8.4). When you extend this file, do the same.

Entry format, always:

- **Symptom** — what a user or the owner saw.
- **Root cause** — the mechanism, stated so it predicts all observations.
- **Evidence** — commit hashes, the exact diff lines, byte counts, measurements.
- **Status** — `fixed in <hash>` / `reverted in <hash>` / `OPEN` / `superseded by <hash or skill>`.

Read-only git commands (`git log`, `git show`, `git diff`) are always safe here. Never run mutating git commands or the bot itself while investigating.

## Index

| # | Battle | Status |
|---|---|---|
| 1 | The removeprefix local-mode bug (flagship) | **FIXED** 2026-10-19 (relpath; committed in 53db8ee); local mode itself archived 2026-10-20 (Entry 10.12) |
| 2 | The Feb 8 2026 whitespace saga | settled by 3bbaa2b + 4bcee6b |
| 3 | The reverted PICKLE & duplicate archiving | fixed in e7befd5 (+4e03190) |
| 4 | TYPE/WHITESPACE MISMATCH check | removed in 081b1fd; smarter revival = campaign candidate |
| 5 | The Big Massive Refactor | settled (366c8d9) |
| 6 | The cythonization wave | settled; caused Entry 1 (now fixed) |
| 7 | Dead ends & deleted artifacts | each closed, see table |
| 8 | Smaller settled battles | each fixed, see subsections |
| 9 | Lore corrections (wrong attributions refuted by diffs) | reference |
| 10 | The 2026-10-19/20 fix batch (B2, C7, C8, C9, C11, Cython 3.3, reminder parser and privileges, .gitignore; B6 mitigated, then retired by archiving local mode; C13(a) fixed; C10 and other dead helpers deleted; blob-sha updates; AI moved off main; sha links, emoji sync in /update_stats, TTS save (10.17-10.19); C12 (10.26); optimization pass: C1-C3, B4, B5, C6, B7, C5 (10.20-10.25), psd_analyzer split (10.27), network/speed (10.28); 2026-10-21: B10 atomic locked pickle saves (10.29), the 0.1 s queue tick and grouped /query_ability output (10.30; the grouping reverted 2026-10-23)) | committed 2026-10-20: 53db8ee (10.1-10.13), bacab06 + 9fb3093 (10.15), 05c643e (10.16), e13d754 (10.17-10.18), bad0496 + ce94585 + c4caaf9 (10.19), df01761 (10.26) |

---

## Entry 1 — The removeprefix local-mode bug (FLAGSHIP, FIXED 2026-10-19)

**Status: FIXED on 2026-10-19**, committed in 53db8ee. Was docs/BUGS_AND_IMPROVEMENTS.md B1. For the 7+ months it was live, production data stayed healthy only because every update ran in remote mode. The story below is kept because the failure mode (a mechanical sweep changing semantics in an unexercised code path) will happen again.

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
  not parser changes. That staleness is tracked as docs/BUGS_AND_IMPROVEMENTS.md D5.

### Status and what replaced the operating rule

- **FIXED** (2026-10-19, 53db8ee). Signature to re-check: `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py`
  hits (:115) and `grep -rn removeprefix commands/` is empty.
- The old rule "always pass `use_local_repo:False`" was retired for the PATH reason only. A second local-mode hazard
  found the same day keeps remote mode the recommended default: with the default `use_local_timestamp:True`, local
  file mtimes are compared with stored GitHub commit times, and on 2026-10-19 all 813 local PSDs were newer. A
  default local run would therefore archive every card into old_stats.pkl (the Entry 3 balloon pattern). Tracked
  as docs/BUGS_AND_IMPROVEMENTS.md B6. On 2026-10-20 the default itself became remote (Entry 10.10), and later that
  day local mode was archived out of the bot (Entry 10.12): `/update_stats` is GitHub-only. The current
  operating rule lives in autocroissant-run-and-operate §7.
- Scheduled runs were separately hardened (Entry 10.1).

---

## Entry 2 — The Feb 8 2026 whitespace saga

**Settled lesson: whitespace in ability text is load-bearing. NEVER collapse runs of spaces.** Gaps of 3+ spaces (`\s{3,}`, `_gap_pattern`, psd_parser.py:324 as of 2026-10-20) are where type icons visually sit — they are the *input signal* for type injection, and leftover gaps in stored text are *evidence* of undetected icons.

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
ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text)   # psd_parser.py:841 as of 2026-10-20
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

Fixed in **e7befd5**; balloon #1 reverted in **eb9aa84**; old_stats stable since 2026-02 (77KB → 82.9KB over four months of routine PICKLEs). Since 9fb3093 (2026-10-20, Entry 10.15) the guard is `old.path != path or getattr(old, "blob_sha", None) != sha`: history is appended only when the PSD's blob changed or the card moved, so re-parses of unchanged PSDs via `cards:` or `force_update` archive nothing. If old_stats jumps >10% in one PICKLE again, suspect a new archive-path bug and diff with `diff_stats.py` before pushing (autocroissant-change-control gate).

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
3. **The rewrite chose a forgiving invariant instead**: inject per line while types remain, then **append leftover types to the last line** rather than raise a problem — visible-but-mispositioned beats false alarms. Current code, psd_parser.py:939-943 (2026-10-20):

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
  - `stats.pkl` values are `CardInfo`/`CardStats` **@dataclasses pickled as `commands.psd_analyzer.*`** (defined in commands/card_models.py since the 2026-10-20 split, `__module__` pinned) → unpickling requires this repo on `sys.path` with `config.py` present (the import chain runs at load). Renaming/moving CardInfo, CardStats, or `commands/psd_analyzer.py` breaks every existing pickle — autocroissant-change-control class rules apply.
  - Any pickle from a checkout ≤ `366c8d9~1` is **legacy plain-dict format** and will not load through current code paths.
  - `metadata.pkl` never returned (its role folded into stats.pkl); `descriptions.pkl` was already dead (Entry 7).
- **Status**: settled architecture. Schema details are owned by impossibility-cards-reference; the pickles-in-git decision rationale by autocroissant-architecture-contract.

---

## Entry 6 — The cythonization wave (Dec 2025) and its long tail

- **f7c915c** (2025-12-02 20:18, "cythonized the code"): `setup.py` born (181 lines) — compiles command modules with `wraparound=False`, `boundscheck=False`, `cdivision=True` (setup.py:36-48 on 2026-10-20); `cython` added to requirements.txt; `.gitignore` +73/−14 (87 changed lines) of build-artifact patterns; and a mechanical sweep converting every `x[-1]` to `x[len(x) - 1]` across frankenstein/psd_analyzer/query_card (negative indexing is banned under `wraparound=False`). **Cost of the sweep**: the removeprefix bug (Entry 1, fixed 2026-10-19) and the `dirname` classify bug (fixed fb47b5d) — both born in this one commit.
- **45de566** (same evening, 21:25, "cythonized more files"): moved `management.py`, `music_player.py`, `diffusion.py` from the exclude list INTO `INCLUDE_FILES` — their original wary comments ("Complex async operations with discord.py", "ThreadPoolExecutor with Process/Queue-like issues", "PyTorch callbacks and dynamic function registration") survive as ironic annotations on the INCLUDE list (setup.py:31-32 on main; the diffusion line left with the AI code, Entry 10.16). Also converted music_player's remaining `[-1]`s.
- **Excluded from compilation, with reasons** (setup.py:9-18 comments, current): `main.py` (entry point), `config.py`/`global_config.py` (configuration), `commands/update_bot.py` (uses execv/execl for restart), `commands/utils.py` (core utilities, decorators, dynamic behavior), `__init__` files, setup.py itself. Net: **8 of 10 command modules compiled** as of 2026-07-11; **7 of 10** since 2026-10-20 (`commands/type_emojis.py` arrived uncompiled on 2026-10-19; `diffusion.py` left main, Entry 10.16).
- **dd800bc** (2026-01-30 23:15): a `bboxes[-1]` had crept back into psd_analyzer (`max_height = max(bboxes[-1][1].y // 3, card_mid_y)`); converted to `bboxes[len(bboxes) - 1]`. Nothing enforces the idiom automatically — it is maintained by hand, and this commit is proof stragglers happen. Do NOT "clean up" `x[len(x) - 1]` back to `x[-1]` in the 7 compiled modules.
- **bf9478e** (2026-01-26 17:02, "lazily import torch and diffusers"): closed the era — `import torch` at module top replaced by `get_torch()` / `torch_available()`, so the compiled+core bot runs with zero AI overhead. Doctrine and mechanics owned by autocroissant-ai-boundary (branch `archive/ai-generation`).
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
| `commands/convert_nf4_flux.py` | 1efe116 2024-08-20 | **f40152e** 2025-01-15 | 144 lines vendored from transformers internals (`_replace_with_bnb_linear`, `create_quantized_param`...) to hand-quantize Flux to NF4 before libraries supported it. Superseded the day diffusers/transformers shipped native `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` — used in `_init_flux_pipeline`, `archive/ai-generation:commands/diffusion.py:262-290` (removed from main, Entry 10.16). |
| `requirements-linux.txt`, `requirements-linux2.txt` | b22dd42 2024-07-18 | **3f9a96d** 2024-08-18 | One-month experiment: Linux/CUDA-box dep lists pinning torch wheels via a Stanford mirror (`-f https://cs.stanford.edu/~nfliu/...`). Superseded by the single `requirements2.txt` with a pip index-URL line — commented out by **ca0b821** (2025-12-02) so plain installs are platform-neutral (CUDA box uncomments it). The 2026-07-11 reorg (committed in a7a4bb3) made requirements.txt core-only and requirements2.txt the full AI set; requirements2.txt left main with the AI code (Entry 10.16). |
| `commands/tools.py` | 5c3b1ab 2024-07-14 (brought `to_thread`) | not dead | **Correction**: not deleted — **renamed** to `commands/utils.py` in f40152e (2025-01-15), `commands/{tools.py => utils.py}` with 0 content change. |
| `music/README`, `models/output/README` | — | b4af57e 2024-12-20, b22dd42 2024-07-18 | Placeholder READMEs to keep empty dirs in git; dropped once .gitignore/dir conventions settled. |
| `metadata.pkl` | fbc407b 2025-05-05 ("added the META pkl...") | **366c8d9** 2025-10-20 | Dict-pickle sidecar of the pre-refactor era; folded into CardInfo fields in stats.pkl (Entry 5). |

---

## Entry 8 — Smaller settled battles (each verified by diff)

### 8.1 Message splitting vs code fences

- **Symptom**: long messages with multiple ``` code blocks split mid-fence, breaking Discord rendering; then splitting crashed outright.
- **Root cause & fixes**: old `split_long_message` only handled a message that both started AND ended with a single fence. **e21b986** (2026-01-18) rewrote it with an `in_code` fence-tracking loop — but also removed the `max_length` default, breaking callers that passed no argument. **d699b3f** (2026-01-26) restored `max_length: int = BREAK_LEN`.
- **Status**: fixed; current `split_long_message` at utils.py:212, BREAK_LEN=1950. (e21b986 also introduced `ABILITY_EXCLUSIONS`, now psd_parser.py:950 as of 2026-10-20.)

### 8.2 Deploy items / card.types clobber + force_update born

- **Symptom**: item cards (the "deploy items") showed wrong types.
- **Root cause**: after per-layer processing correctly appended above-midline icons to `card.types` (psd_parser.py:536-539 as of 2026-10-20), a later block **overwrote** it with the below-midline bbox list: `card.types = [name for name, _ in type_bboxes if name]` — i.e. types became the inline-ability icons.
- **Fix**: **f94d95f** (2026-01-18) deleted the overwrite. Same commit: added `MISSPELT_CARD_TYPES = ['undread', 'tornado']` validator (now `+ 'error'`, psd_parser.py:24 as of 2026-10-20 — today's single problem card "20 Creature Types" trips `tornado`), and threaded the `force_update` parameter end-to-end (slash param born here; its interaction with archiving is Entry 3).
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
- **Settled convention**: `/ai`'s `seed` param is `Optional[str]`, converted with `int(seed)` + error reply. The in-code comment at `archive/ai-generation:main.py:524` (`/ai` left main, Entry 10.16): "ints are limited to 15 digits or less by discord, so I need to take it as a str and then convert it to an int". Do not "fix" the param type back to int.

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

## Entry 10 — The 2026-10-19/20 fix batch

These fixes landed together on 2026-10-19 alongside the stoogedings icon-font parsing (impossibility-cards-reference
§4.1), `commands/type_emojis.py` and the reminder rewrite. "Working tree" below means uncommitted when written; 10.1-10.13 were
committed in 53db8ee, 10.15 in bacab06 and 9fb3093, 10.16 in 05c643e, 10.17-10.19 in e13d754, bad0496, ce94585 and
c4caaf9, 10.26 in df01761; 10.20-10.25, 10.27 and 10.28 are the 2026-10-20 optimization pass; 10.29-10.30 the
2026-10-21 owner decisions. Line numbers are
2026-10-19 readings unless an entry says otherwise.

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
  timestamps re-archives everything (docs/BUGS_AND_IMPROVEMENTS.md B6).
- **Superseded 2026-10-20** (Entry 10.12): with local mode archived the option no longer exists, so
  `SCHEDULED_DEFAULTS` was removed. `RETIRED_OPTIONS` stripped (removed 2026-10-25) `use_local_repo`/`use_local_timestamp` from
  stored reminders instead.

### 10.2 Remote-mode temp PSD downloads leaked (was C7)

- **Symptom**: every remote `/update_stats` left one `urlretrieve` temp file per parsed PSD in the OS temp dir.
- **Fix**: `try: card = self.parser.parse(local_file, path)` / `finally: remove(local_file)` (psd_analyzer.py:1178-1181 on 2026-10-19; 1664-1668 on 2026-10-20, now around a blob download; repo_traverser.py:364-368 after the split; card_data.py:481-488 since 2026-10-22, plus :478 when the commit lookup fails).
- **Verified**: code read 2026-10-19. The next live remote sweep is the first real exercise.

### 10.3 `.DS_Store` became a 36th card type on macOS (was C8)

- **Symptom**: local type population (`_populate_types_from_local`) listed `.ds_store` as a type. Harmless, since
  no layer is named that, but it polluted `all_types`.
- **Fix**: `if not file.startswith('.')` (psd_analyzer.py:1079 on 2026-10-19; 1502 on 2026-10-20 and repo_traverser.py:212 after the split, card_data.py:301 since 2026-10-22, in `_populate_types_from_local`, which the diagnostics scripts still use).
- **Verified**: the 2026-10-19 sandboxed local sweep (Entry 1) ran through this code with 0 type differences.

### 10.4 `update_stats()` Python default force-reparsed everything (was C9)

- **Symptom (trap)**: the Python default was `force_update=True` and only the slash wrapper passed False, so a bare
  programmatic call re-parsed all 813 cards.
- **Fix**: the Python default is now `force_update: bool = False` (psd_analyzer.py:1401 on 2026-10-19; repo_traverser.py:486 after the 2026-10-20 split, :59 since 2026-10-22), matching main.py:383.
  Code that wants a forced re-parse must now say `force_update=True` (the validation recipes already do).

### 10.5 management.py stopped compiling under Cython 3.3

- **Symptom**: `python3 setup.py build_ext --inplace` failed with
  `commands/management.py:210:12: 'target_message' redeclared`. `cythonize()` raises `CompileError` before
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
- **Fix**: `_next_run(when, frequency, now)` jumps to the first slot after now in one step (analytics.py:203),
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
  callback's own `lacks_admin` runs again at fire time. Legacy reminders with no creator still run as the bot
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
  traversal supplies; local mtimes and commit dates are not comparable. (`_should_update_card` survives only on
  `archive/local-update-stats`; main compares blob shas, Entry 10.15.)
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
  helpers `_populate_types_from_local` and `_icons_from_local` stayed in psd_analyzer.py (repo_traverser.py since the split, card_data.py since 2026-10-22) for parse_one.py and
  gap_trace.py. In analytics.py, `SCHEDULED_DEFAULTS` (Entry 10.1) became `RETIRED_OPTIONS` (analytics.py:44):
  stored reminders drop the two options at load (`REMINDER_VERSION` 2 → 3) and at fire time, pre-v2
  positional values are mapped by the old option order first, and a new reminder naming one gets a
  "Dropped ..." note instead of an error.
- **Verified 2026-10-20**: all 8 INCLUDE_FILES compile with Cython 3.3.0; an offline migration test with fake
  reminders (`/update_stats use_local_repo:False` → `/update_stats`; legacy `/update_stats True False True True`
  → `output_problematic_cards:False force_update:False`, the old parser reading capitalised `True` as False;
  `/sync_emojis refresh:True` unchanged); the archived traversal parsed a one-card temp clone with a
  repo-relative path.
- **Status**: done (53db8ee). Revive local mode only after fixing its change signal (B6).

---

### 10.13 Dead helpers deleted (2026-10-20)

- **Why**: the owner first asked that deprecated code go to archive branches. Later the same day they had the two
  branches made for it deleted, along with the code. Rule now (autocroissant-architecture-contract, "Retiring a
  feature"): dead code with no callers is deleted; archive branches are only for retired features. A vulture scan of main.py, global_config.py, commands/ and the skill scripts (callers
  included), plus grep for every call site, found these with no callers:
  - `utils.convert_value` and `utils.parse_named_args`: the old scheduler's argument parser (10.7, was C10).
  - `utils.url_to_pilimage`, `utils.cv2_to_pil`, `utils.pil_to_cv2`: image conversions from the diffusion work. The
    bot uses only `url_to_cv2image`/`cv2discordfile` (/frankenstein) and `pildiscordfile` (/ai, since moved off main, 10.16).
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
  `RETIRED_OPTIONS` and the reminder migration (removed 2026-10-25), `_populate_types_from_local`/`_icons_from_local` (diagnostics
  scripts), `global_config.LOCAL_DIR_LOC` (diagnostics scripts), every slash command and the AI and music features (AI later moved off main, 10.16).
  `analytics.parse_time_str` (a one-line public wrapper of `_parse_when`, written in the rewrite and uncalled) was
  left in place as a candidate, and deleted in the 2026-10-20 code-health pass.

### 10.14 History rewritten to drop an attribution trailer (2026-10-20)

- **Why**: 9280fea (a PICKLE commit made on the jetson) carried a `Co-Authored-By` line; the owner asked to remove it.
- **What changed**: 9280fea, 2d5e85c, c2a9c2a were re-created as c7f2d19, 53db8ee, f2013eb (same trees, authors and
  dates; 2d5e85c's message shortened) and force-pushed with a lease. The jetson was moved to c7f2d19 (same tree).
  Anyone who pulled before: `git fetch && git reset --hard origin/main`.
- **Prevention**: the jetson now has the Mac's global commit-msg hook (`~/.git-hooks`), which strips such lines.

### 10.15 /update_stats: timestamp sweep replaced by blob shas (2026-10-20)

- **Why**: every run asked GitHub for each PSD's commits (~1,600 requests, minutes) even when nothing changed, and
  only saved when a PSD changed, so author/edit/duplicate changes were lost on quiet runs.
- **What changed**: `card.blob_sha`; tree with If-None-Match (304 ends the run); PSDs by blob sha, 4 downloads
  ahead; small files cached by sha; `cards:` option; save when anything changed. Each re-parsed card still costs one commit lookup
  (`_get_remote_timestamp`). The ETag cache is in memory, so the first run after a restart gets the full tree.
  First run adopted shas with 31 requests. Measured: nothing changed = 1 request, 0.4 s.
- **Follow-up (9fb3093, same day)**: old_stats history is appended only when the blob changed or the card moved, so
  re-parsing unchanged PSDs via `cards:` or `force_update` no longer archives them (Entry 3).

### 10.16 AI generation moved off main (2026-10-20)

- **Why**: the owner asked to take the torch/AI code out of main.
- **What changed**: branch `archive/ai-generation` (ed21cd2) is main at faebfc7 with AI intact, plus a note. On main:
  `commands/diffusion.py`, the six AI commands, `init_pipeline`, the "ai" help category, `pildiscordfile`,
  `requirements2.txt`, `models/` and its .gitignore lines are gone; setup.py builds 7 modules. S1 fixed:
  `config.browser_cookies` (old configs keep `vram_usage == "mps"`). C4 fixed. Bug B3 lives on with the branch.

### 10.17 Card image links pinned to commit shas (was C14, 2026-10-20)

- **Symptom**: `/query` and ability-search images linked `raw.githubusercontent.com/<repo>/main/<path>`, so an old
  Discord message showed whatever the PNG looks like today.
- **Root cause**: `get_card_url` built the URL from `main`, not from the commit that last changed that PNG.
- **Evidence**: e13d754, query_card.py: `-return f"https://raw.githubusercontent.com/{self.repository}/main/{quote(...)}"`
  → `+return raw_url(self.repository, path, self.blobs.get(path))`; new commands/repo_history.py walks the history
  clone (`~/Documents/TTSCardMaker-history.git`) for {path: (blob, commit)} into repo_history.json; refreshed at
  startup and on /update_stats with a changed tree; falls back to `main` when a path is unknown.
- **Status**: fixed in e13d754.

### 10.18 Emoji sync moved inside /update_stats; weekly reminder retired (was Task 1b, 2026-10-20)

- **Symptom**: the startup sync only uploaded MISSING emojis, so an edited icon kept its old emoji. A weekly
  reminder (`f590a57e`, `/sync_emojis`, Sun 3 AM) covered new icons between restarts.
- **Root cause**: no record of which icon blob each emoji was made from.
- **Evidence**: e13d754, type_emojis.py `sync_changed_emojis` (+ `_load_blobs`/`_save_blobs`, type_emojis.json):
  on a changed tree, upload or replace emojis whose blob changed (with `_no` variants) and delete the ones whose icon
  is gone; `update_stats` queues it when `traverser.tree is not None`. Then the reminder was removed with
  `/remove_reminder reminder_id:f590a57e`.
- **Status**: fixed in e13d754. `/sync_emojis refresh:True` is still the manual full re-upload (V1).

### 10.19 TTS save kept current by the bot (was Task 2, 2026-10-20)

- **Symptom**: the Tabletop Simulator save drifted from the repo (new cards missing from decks, unpinned links,
  old names), and it was fixed by hand.
- **Root cause**: nothing tied the save to card changes.
- **Evidence**: bad0496 (`python -m commands.tts_save`), ce94585 (`/update_tts_save`; `update_after_stats` after a
  /update_stats whose tree touched images, TTS/decks.json, MDW/Card Edits.json or the save), c4caaf9 (when
  `GIT_TOKEN` can't write: commit through the history clone and push over SSH, no force).
- **Status**: shipped; operations in autocroissant-run-and-operate §TTS save. The run after /update_stats
  (`save_affected`, `update_after_stats`, `_retry`, `_watched`) was deleted later on 2026-10-20: TTSCardMaker's
  `.github/workflows/tts-save.yml` runs `python -m commands.tts_save` on PNG / `TTS/decks.json` /
  `site/Card Edits.json` pushes (`MDW/Card Edits.json` before the website files moved, 2026-10-20), daily and on dispatch
  (ttscardmaker-companion §2.6); it first symlinked its
  workspace to `~/Desktop/TTSCardMaker`, replaced by the `AUTOCROISSANT_LOCAL_DIR` / `AUTOCROISSANT_HISTORY_CLONE`
  env overrides in global_config.py. `/update_tts_save` stays as the manual run. At the cutover (2026-10-24)
  tts-save.yml was deleted: card-sync.yml's TTS save step runs the same updater.

### 10.20 help.py and describe strings matched to the code (was C1, C2, C3, 2026-10-20)

- **Symptom**: /help and the /pull description said /pull "does a hard reset" (main.py also said "than"); 4 of 51
  commands had no /help entry; /export_cards said only_ability defaults to True.
- **Root cause**: help.py and the describe strings were edited separately from the code (docs-and-style §2.4's
  checklist was skipped), and the only_ability describe string stated no default.
- **Evidence**: help.py /pull, /export_cards and 3 new entries (`/list_guild_channels`, `/get_channel_messages`,
  `/view_old_metadata`; the fourth, /list_guild_members, was removed: 10.23); main.py /pull description and the
  /export_cards describe strings with `Default = ...`. `tests/test_robustness.py::test_help_covers_every_command`
  (50 registered = 50 documented) and `::test_help_and_describe_match_the_code`.
- **Status**: fixed 2026-10-20 (robustness pass; uncommitted when written).

### 10.21 /play outside a voice channel raised TypeError (was B4, 2026-10-20)

- **Symptom**: `/play` with the caller not in voice logged `TypeError: object NoneType can't be used in 'await'
  expression`; the reply "You are not in a voice channel." still arrived because it was queued first.
- **Root cause**: `init_vc` did `await queue_message(...)`; queue_message is sync and returns None. The exception
  also stopped `slash_play` before `play_music`, so nothing was downloaded: just dropping the `await` would have
  started downloads with no voice client.
- **Evidence**: `init_vc` now returns False after queueing the reply and True when connected; `/play` and
  `/play_all` return on False. `test_init_vc_outside_voice_replies`: reply queued, no exception, `play_music`/`play_all`
  never called.
- **Status**: fixed 2026-10-20 (robustness pass).

### 10.22 Dispatch/edit loops no longer die on one bad message (was B5, 2026-10-20)

- **Symptom**: one poisoned send stopped `process_dispatch_queue`/`process_edit_queue` for good and the bot went
  silent until a restart. Vectors: a deleted or invisible channel (`client.get_channel()` None, AttributeError inside
  the fallback), content over 2,000 characters (a 2,001+-char /query_ability, /play or /query_rulebook option echoed
  back, a long /queue), Forbidden in a channel without Send Messages, or any non-HTTP error (ValueError, TypeError).
  The fallback also re-sent attachments as 0-byte files (the failed send had read them to the end).
- **Root cause**: a `tasks.loop` stops on an uncaught exception, the except caught only HTTPException/AttributeError,
  and the fallback ran inside the except with no guard.
- **Evidence**: main.py `_dispatch`/`_edit`/`_fallback_send`/`_split_content`; each drained item is in its own try
  that logs `Dispatch: dropped a message for channel N: <error>`; the fallback calls `File.reset()` and skips a None
  channel; long content is split with `split_long_message` (edits: first chunk edited, the rest sent). The command loop
  guards each item too and keeps a reference to each task; `run_command` logs a failed function with its name and
  traceback. 9 tests in `tests/test_robustness.py` (all fail on 7f6bd05): missing channel, Forbidden/400/ValueError/
  TypeError in the fallback, a 6,000+-char /queue-like message (4+ chunks, each <= 2,000, file on the last),
  1,000-byte file re-sent whole, edit loop, command loop.
- **Status**: fixed 2026-10-20 (robustness pass). The 2026-10-19 `_channel_missing` reminder guard stays.

### 10.23 /list_guild_members removed: it needed the members intent (was C6, 2026-10-20)

- **Symptom**: `/list_guild_members` always answered "Intents.members must be enabled to use this."
- **Root cause**: main.py builds `Intents.default()` + `message_content`; `guild.chunk()` needs the privileged
  members intent. Setting `intents.members = True` in code without turning on Server Members Intent in the Developer
  Portal makes login fail (PrivilegedIntentsRequired), so the bot would not start.
- **Evidence**: the command, its import, `management.list_guild_members` (and its `__all__` entry, plus the now
  unused `ClientException` import) were deleted; it never had a help entry. 50 commands registered.
- **Status**: removed 2026-10-20. To bring it back: enable the intent in the portal first, then in code (with
  `Client(chunk_guilds_at_startup=False)`), then restore the function from 7f6bd05.

### 10.24 Startup survives a GitHub 5xx (was B7, 2026-10-20)

- **Symptom**: after a restart during a GitHub outage, /query found nothing until the next /update_stats and the
  emoji sync failed, so [fire]-style tokens stayed text (seen on the jetson 2026-10-20). A `/set_repo` typo had the
  same effect mid-session, and blocked the event loop while it fetched.
- **Root cause**: `populate_files` called `response.json()` on whatever came back and cleared `git_files` first (a
  JSON 401/403/404 body emptied the list; an HTML 5xx page raised JSONDecodeError in `run_command`). The emoji
  listing raised on the 5xx and published nothing.
- **Evidence**: `github_api.get_with_retry` (in utils.py until the 2026-10-20 code-health pass; 3 tries, 1 s then 2 s apart, on a connection error, 429 or 5xx);
  `populate_files` keeps the last good tree, list and ETag on any non-200/304 or bad body and prints
  "Card list: GitHub tree fetch for <repo> failed (...); keeping the last good list (N entries)"; `set_repository`
  defers, fetches in a thread and restores the old repository on failure; `_list_type_icons` retries and falls back to
  `card_repo._tree` ("using the last good card tree"); `sync_type_emojis` still loads the bot's existing emojis
  when the listing fails ("using the N existing emojis"). Tests: 503 x3, HTML 200, 401, connection errors, non-dict
  JSON, retry-then-success, /set_repo 404, emoji fallback.
- **Status**: fixed 2026-10-20 (robustness pass). /update_stats' own tree fetch is unchanged (B9).

### 10.25 Duplicate slash function names renamed (was C5, 2026-10-20)

- **Symptom**: main.py defined `slash_set_ratio` twice (the second was /set_repo's handler) and `slash_delete_song`
  twice (the second was /delete_all_music's).
- **Root cause**: copy-paste; harmless because `@tree.command` registers the function object at def time and
  `slash_registry` keys on `cmd.name`.
- **Evidence**: the second defs are now `slash_set_repo` and `slash_delete_all_music`; no `name=` changed.
  `test_handler_names_are_unique` checks every top-level def in main.py is unique and that set_ratio, set_repo,
  delete_song and delete_all_music map to the right callbacks.
- **Status**: fixed 2026-10-20 (robustness pass).

### 10.26 Pillow and PyNaCl declared (was C12, 2026-10-20)

- **Symptom**: requirements.txt listed neither, yet type_emojis.py imports PIL at module level and voice needs PyNaCl;
  both arrived only as transitive dependencies (psd-tools, PyGithub).
- **Root cause**: requirements written from direct imports of the original code; type_emojis.py (2026-10-19) added PIL.
- **Evidence**: df01761, requirements.txt `+Pillow`, `+PyNaCl` (autocroissant-build-and-env §3).
- **Status**: fixed in df01761.

### 10.27 psd_analyzer.py split behind a facade; shallow-path crash fixed (code-health pass, 2026-10-20)

- **Symptom**: psd_analyzer.py was 2,347 lines mixing models, edits, parser, traversal and commands. Compiled,
  `CardClassifier.classify("Card.psd")` segfaulted (exit 139) and `"Items/Card.psd"` died with SIGBUS (138): with
  `boundscheck=False` the folder indexing read out of bounds (the .py raised IndexError/ValueError). No such PSD
  existed, but one stray commit would have killed the jetson bot during /update_stats.
- **Root cause**: growth by accretion; the classifier assumed every path had its folder rule's depth.
- **Evidence**: commands/card_models.py, card_edits.py, psd_parser.py, repo_traverser.py hold the code (the traversal moved on to the uncompiled card_data.py on 2026-10-22);
  psd_analyzer.py re-exports it and keeps the metadata/export commands and init_psd. `CardInfo.__module__ =
  CardStats.__module__ = "commands.psd_analyzer"` (card_models.py:151): re-dumped stats.pkl and old_stats.pkl were
  byte-identical in all four .py/.so combinations. The four modules joined INCLUDE_FILES (11 compiled) so the moved
  code keeps its compiled semantics; query_card imports through the facade so a stale monolithic .so cannot leave
  two `stats_db` objects. `classify` returns UNKNOWN for a path too shallow for its rule (psd_parser.py:243-253).
  Gates: parse_all/compare over 813 cards, 0 differences on .py and compiled; `tests/test_split_contract.py`.
  Also deleted as dead: `analytics.parse_time_str`, `get_old_stats_as_dict`, `CardInfo.from_dict`, two unread
  FakeInteraction attributes; GitHub access moved to commands/github_api.py (uncompiled); renames `perms_check` ->
  `lacks_admin`, `num_new` -> `num_changed`, `dirty_files` -> `seen_cards`.
- **Status**: done 2026-10-20 (uncommitted when written). Deploys must rebuild: a stale analytics .so imports
  `perms_check` and stops the bot at startup (autocroissant-run-and-operate §3).

### 10.28 Network and speed pass (2026-10-20)

- **Symptom**: /query_ability plain queries took ~17 ms of fuzzy matching each; /update_stats waited ~0.9 s per
  changed PSD on 2 serial commit lookups; startup fetched the 598 KB tree twice; `import main` took ~1.1 s / 185 MB.
- **Root cause**: no upper-bound prefilter before `SequenceMatcher.ratio()`; `totalCount` + `commits[0]` on a
  per_page=1 client; each tree user kept its own fetch; cv2, yt_dlp and PyGithub imported at module level.
- **Evidence**: query_card `_fuzzy_labels` (skip when `2*min(la,lb)/(la+lb) <= match_ratio`: 60 short queries
  1138 → 63 ms, results identical) and the exact-key fast path in `_find_closest_match` (100 exact names 424 → 0 ms);
  repo_traverser `_commit_page` (per_page=100, 2-thread pool: 6 PSDs 16-18 s → 6.5-8.7 s, commit requests 25 → 6;
  forced sweep 2,714 → 1,810 requests by the audit's count, which took all 904 tracked PSDs; the 813 outside
  Markers/MDW give ≈ 2,450 → 1,630, analysis-toolkit Recipe 4) with one PyGithub client per thread (a shared one
  returned another PSD's commits 5-6 times in 500 under load); the tree ETag handed to the card list and shared with
  the emoji listing (startup 200 + 200 → 200 + 304); lazy imports (`import main` 1059-1278 → 646-720 ms, 185 →
  122.5 MB RSS); TTS save objects cached by sha (repeat no-change run 9 requests / 6.4 MB → 3 / 0 MB). Tests:
  `tests/test_network_speed.py`. Parser gate: 813 cards, 0 differences (.py and compiled).
- **Status**: done 2026-10-20 (uncommitted when written). Not done: lazy GitPython, per-thread `requests.Session`
  for blobs (the tick latency and the one message per result: 10.30); details in
  (`docs/Finished/optimization-pass-2026-10-20/network-speed-results.md`).

### 10.29 Card pickle saves made atomic and locked (was B10, 2026-10-21)

- **Symptom**: `StatsDatabase.save` and `save_aliases` wrote stats.pkl, old_stats.pkl and aliases.pkl in place with
  no lock, so a kill mid-write (`/restart_bot`, `/update`) or two writers (`/update_stats` beside `/mass_replace`, a
  reminder's `/update_stats` beside `/push`) could leave a truncated pickle that the load only catches as EOFError.
- **Root cause**: the plain `open(path, 'wb'); dump(...)` path. reminder.pkl got the safe pattern on 2026-10-19; the
  card pickles never did.
- **Evidence**: `card_models.save_pickle` (card_models.py:20-34: temp file beside the target, flush + fsync,
  `os.replace`, temp removed on any failure) under one module-level `threading.Lock` per pickle (`_stats_lock`,
  `_old_stats_lock`, query_card's `_alias_lock`); loads unchanged and unlocked. Bytes identical to the old path
  (`pickle.dumps` equality in `tests/test_pickle_saves.py`, which also kills a write after partial bytes and runs
  two saving threads). `.gitignore` ignores `*.pkl.tmp`. Parser gate: 813 cards, 0 differences.
- **Status**: done 2026-10-21 (uncommitted when written).

### 10.30 Queue tick 0.1 s; /query_ability listed cards 10 per message (owner decisions, 2026-10-21; the listing reverted 2026-10-23)

- **Symptom**: a reply waited for the next 1 s drain (502 ms mean, ~99% of /query's bot-side time, 10.28), and
  /query_ability sent one message per matching card (85 on average, max 763) through the one dispatch loop.
- **Root cause**: `@tasks.loop(seconds=1)` on the three drains; one `queue_message` per result in `query_ability`.
- **Evidence**: the three drains tick every 0.1 s (main.py:880-912; ordering, per-item try, `_running`, the fallback
  and the split untouched); `check_reminder` moved to its own 1 s loop `process_reminders` (main.py:874-876), started
  and stopped with the drains, so the reminder heartbeat is unchanged. Idle cost (`docs/Finished/optimization-pass-2026-10-20/bench/idle.py` on a snapshot, 10 s,
  no Discord): 1.29 ms CPU/s with 0 reminders, 1.53 with 200 (0.33 / 0.40 at 1 s), 2 threads alive.
  `query_ability` (query_card.py:577-623) posts one match as before (its image URL, a preview; since 2026-10-21 its name when
  the PNG is not in the tree: `get_card_url` returned None and the content=None message was dropped by Discord); more as
  `<name> — <url>` lines, `RESULTS_PER_MESSAGE` = 10 per message through `queue_long_message`.
  Tests: `tests/test_query_output.py` (0, 1, 10, 11, 25 results; limit; long links; a card without an image, alone or listed).
- **Status**: done 2026-10-21 (uncommitted when written). The listing was reverted 2026-10-23 (owner: "discord has a
  limit on the number of images it shows per message", so a 10-link message previewed only a few cards): one message
  per card again, its image URL alone, its name when the PNG is not in the tree; `RESULTS_PER_MESSAGE` deleted. The
  0.1 s tick stayed until 2026-10-25, when the drains began waking on enqueue (B-1, branch work/instant; reply hop
  50 ms → about 1 ms, 1 s fallback poll; autocroissant-architecture-contract). Do not re-batch image links. Since
  2026-10-25 (branch work/grid; owner: "we can only do the query_ability ... improvement if we send a grid or some big
  image with all the cards") two or more results come as grid images (commands/card_grid.py, `post_grids`): 20 cards
  per JPEG, one message per grid with its cards' list; 'damage' (186-188 cards) 188 sends → 10. Measured offline with a
  fake channel (~/Documents/Tools/wt/scratch/grid/measure.py: 150 ms per send, 4 MB/s upload, 5 sends per 5 s): about
  191 s → 9 s (cached pictures; 13 s with all 186 decoded from local PNGs, the jetson's first run adds their download).

### 10.31 Deleted PSDs archived, not deleted (was B8, 2026-10-22)

- **Symptom**: a card whose PSD was deleted stayed in stats.pkl until some other card changed (`prune_clean_cards`
  ran only when `num_changed > 0`); once pruned, its record sat in old_stats untagged, indistinguishable from history.
- **Root cause**: the prune sat behind the change gate; nothing marked a record as "deleted".
- **Evidence**: 0166948 and f50f0a2 prune on every full run (`_can_prune`: no failed new-name download, truncated or
  empty tree). The owner then asked to archive rather than delete ("to preserve history for potential future features
  or views"): `CardInfo.archived` (plain-dict tag: at, path, blob, commit, renamed_to, backfill), `StatsDatabase.archived()`
  the one read filter, `restore()` for a PSD back, psd_details records of archived blobs kept. A review found a kill between
  the saves after a same-blob restore lost the record (restore popped it, old_stats.pkl was written first); fixed by
  restoring a copy and dropping the tagged copy only after stats.pkl is written (`save()`, `drop_stale_archives`), and by
  `add_history` replacing an equal last version (a kill after an edit doubled history). A second review found: a kill
  after a same-blob move doubled history (the old record's path already followed the card), a kill during a re-parsed
  restore left the card untagged in old_stats only, a `/update_metadata` stub under an archived name was archived over the
  real card, and a blobless record matched any stub at its path. Fixed: `add_history` compares all but the path and keeps
  a tagged copy for `save()` to untag; `_tagged_copy` needs full equality without a blob; `/update_metadata` refuses
  archived names. Also `/pull` (as before this work) could drop a card from both pickles (stats.pkl from the archiving
  side, old_stats.pkl from the other); `update_bot._rescue_merged_cards` now restores it after the merge. A third
  review: a rescued listed card sat untagged (a permanent fake old version), so it is tagged now; another PSD under an
  archived name took the old author (restore needs the tag's path or blob); star moves moved any off-rule card and
  could clobber a CustomDeck key. Tests:
  `tests/test_archive.py`, the archive and kill cases in `tests/test_update_stats.py`.
- **Status**: done 2026-10-22 on branch work/archive-deleted (not yet on main when written). TTSCardMaker e0e26b2f on
  branch work/archived-field (the site build knows the `archived` field; was 1acc2862) lands first, before the first
  pickle push: every re-parsed card carries `archived = None`, so each build would warn of an unknown field.

### 10.32 The pickles, /push and the legacy /update_stats retired (data build step 10, 2026-10-25)

- **Symptom**: after the cutover (2026-10-24 23:02) the bot read the Card sync Action's `data` branch, but the repo
  still carried the frozen `stats.pkl`, `old_stats.pkl`, `aliases.pkl` and `psd_details.json`, the `CARD_SYNC` switch
  (its False branch the rollback), `/push` (a reply saying it was retired), the legacy `/update_stats` parse
  (repo_traverser.py, `card_data.update` with its commits-API lookups and the one-time history cleanup/backfill), the
  legacy `/fonts` cache and every pickle writer.
- **Root cause**: decision 6 of docs/Campaigns/DATA_BUILD_ACTION.md (step 10, "within a day of clean runs"); owner
  ruling 2026-10-25 ~04:00: do it now.
- **Evidence**: branch work/pickles. Deleted: the four root files, commands/repo_traverser.py, commands/card_runner.py
  (bake_lock, the startup temp-PSD sweep), commands/history_backfill.py with its two JSON files (their once-only fixes are
  in the data), `card_db.export_pickles`/`dump_pickle`/`from-pickles`, `card_models.save_pickle`/`StatsDatabase.save`/
  `load`/`rescue_dropped`, query_card's aliases.pkl registry, update_bot's push and /pull rescue, font_check's
  `repo_fonts` cache, card_data's legacy flag, `CardBuild`'s commits API and aliases.pkl reads, `card_edits.ALIASES_MOVED`,
  `card_history.record_edits`/`dedupe`/`commits_of`, the `CardInfo.__module__` pin. Kept (read-only, historical):
  card_db's allow-list unpickler (`load_pickle`, `read_pickles`, `read_cards`), the build's `--seed` from an
  AutoCroissant commit (with the PICKLE-commit recovery, `card_history.bot_pickles`), `card_history refine` (now
  read-only), changelog's `history --pickles`, inspect_pickle.py and diff_stats.py. TTSCardMaker work/pickles: site.yml
  always checks out `data` (no `CARD_DATA` switch) and build_site.py lost its pickle loader and `aliases.pkl`.
- **Status**: built 2026-10-25 on work/pickles in both repos; 46 slash commands. The pickles stay in git history.

### 10.33 The rest of the pickle era and finished tools removed (cleanup, 2026-10-25)

- **Symptom**: after 10.32 the read-only pickle side (10.32's "Kept" list) and several finished one-time tools stayed.
- **Root cause**: owner rulings 2026-10-25 ~06:00-07:00: delete deprecated and dead code, all pickle-era code included.
- **Evidence**: branch work/clean. Removed: card_db's allow-list unpickler (`SeedUnpickler`, `load_pickle`,
  `read_pickles`, `ALLOWED`) and `read_cards`' pickle forms; the build's `--seed` (`_seed_folder`, `_bot_file`,
  `bot_pickles`, `BOT_CLONE`; `--empty` starts new card data instead) with card_history's PICKLE recovery, seed pass
  (`refine`, `collapse`, `normalize`, `misreads`) and `refine` CLI; changelog's `history --pickles`; the diagnostics
  scripts' pickle sides; the rules skill's loadpkl.py; the reminder.pkl conversion; the jetson's fonts cache
  (`refresh_cache`, `FONTS_CACHE`); `discord_archive.py`, `discord_match.py`, `png_recompress.py`, psd_details' backfill
  CLI and txt2_repair's `--check`/`--in-place`; the Discord edit credit (`changelog.CREDIT`, `card_bake.DISCORD`); the
  one-path `*_PATHS` tuples. TTSCardMaker work/clean: card-sync.yml's seed, bootstrap and rollback switches; the card and
  font checks folded into card-guards.yml's check job. Kept: the read-only diagnostics on db/ JSON.
- **Status**: built 2026-10-25 on work/clean in both repos.

### 10.34 Bake bugs: kerning, typed colour, dropped characters, text past the panel (was B12 and plan items, 2026-10-25)

- **Symptom**: (a) Binge Eater's ", 1T" typed after its olive "SPD" printed olive; (b) a whole-ability edit dropped
  what the parser's reading strips (Deepnest's "reroll :3" baked "reroll:3"), and a part edit removing a space before
  punctuation ("+100% Lifesteal ." -> ".") was skipped as redundant; (c) Autopsy Report's appended sentence printed a
  third line under its panel (told, not fitted); (d) B12: Pandamonium's Health is kerned off in its Txt2 story, on in
  TySh, and `story_style_drift` passed it.
- **Root cause**: (a) `edit_styles` gave an insertion the nearest character's fill, the coloured word beside it;
  (b) the single-layer path set the website's text (the parser's reading, `normalize_ability`) as given, and
  `_part_same` compared parts by that rule; (c) no fit rule in the bake (the website's preview shrinks the text);
  (d) the Txt2 kerning (story feature 11: 0 off) is no STORY_MIRROR key, and the engine kerns as TySh says.
- **Evidence**: branch work/bakebugs. (a) text typed between two runs or at a line's end in a line of several colours
  takes `body_fill`; inside a run, that run's. (b) `from_reading` projects the edit over the layer (`project`, as over
  several layers); `_part_same` keeps quotes and spaces. (c) `fit_below`: past the old text onto the design below
  (`ink_beyond`) or off the canvas, the Ability is scaled by 0.94 down to 40% (the preview's `fit`) until it is no lower
  than the old text; a tabbed layer or no fitting size: as before, told. (d) `story_style_drift` compares kerning on
  characters that end a pair (Photoshop writes a paragraph's first character's TySh kerning off); 220 of 813 PSDs differ
  (196 the template's Health label, 42 Abilities, 10 titles), so the bake does not refuse it: `follow_kerning` sets the
  edited layer's TySh from its story, `whole_redraw` too, and the after-save gate refuses only new drift. Measured over
  the 57 drifted titles and Abilities: 4 drew closer to their pixels (Soul Sanctum IoU 0.67 -> 0.995, Colosseum of Fools
  0.55 -> 0.98), none worse; an edit on them spliced instead of redrawn whole (17.3% and 7.7% bad -> 0.3% and 0.6%).
  Proof bakes (`PSD_DIRTY_VERIFY=1`, base vs branch): 11 other cards byte-identical (Mini Doomer's title too), The Hive
  keeps its trailing blank, Frostmourne's off-canvas line fitted at 88%.
- **Status**: built 2026-10-25 on work/bakebugs.

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

Written 2026-07-11 against HEAD = 284d13c ("PICKLE", 2026-06-20). Extended 2026-10-19 against HEAD = a7a4bb3 ("skills and memories", which also committed the requirements split) plus the 2026-10-19 working-tree batch (Entry 10, since committed in 53db8ee). Re-verification one-liners for every volatile fact (cwd = repo root):

| Fact | Re-verify with |
|---|---|
| New history since this file | `git log --oneline -5` (anything above 167f30a, or the commits of the 2026-10-20 optimization pass, → review entries) |
| Entry 1 fixed | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115 on the archive branch; was psd_analyzer.py:1226 before the archive) and `grep -rn removeprefix commands/` (expect empty); `git log -S removeprefix --oneline -- commands/psd_analyzer.py` shows the fix 53db8ee above f7c915c |
| Entry 1 origin diff | `git show f7c915c -- commands/psd_analyzer.py \| grep -B5 -A5 removeprefix`; old code: `git show 366c8d9:commands/psd_analyzer.py \| grep -n 'relative_path ='` |
| Local mode archived (Entry 10.12) | `grep -rn 'use_local' main.py commands/` (finds nothing since 2026-10-25); `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived`; `grep -n '^REMINDER_VERSION' commands/analytics.py` (3) |
| Dead helpers deleted (Entry 10.13) | `git branch --list 'archive/*'` (two branches: archive/ai-generation, archive/local-update-stats); `grep -rn 'convert_value\|parse_named_args\|url_to_pilimage\|cv2_to_pil\|pil_to_cv2\|_exact_match_pattern' main.py commands/` (expect empty) |
| Exact token spellings (Entry 10.11) | `grep -n -A8 "def spellings" commands/psd_parser.py` |
| Entry 10 fixes present | `grep -n "remove(local_file)\|startswith('.')\|force_update: bool = False" commands/card_data.py commands/repo_traverser.py` (card_data.py 478/488, 301, 234/658; repo_traverser.py 61); `grep -n '^LEGACY_RUNS_AS_BOT\|^def next_run' commands/analytics.py`; `grep -n 'target_message: Optional\[Message\]' commands/management.py` (one hit, before the `try`) |
| B6 hazard (only relevant if local mode is revived from the archive) | unpickle stats.pkl with stub classes and compare each `card.timestamp` with `os.path.getmtime(~/Desktop/TTSCardMaker/<card.path>)`: 813/813 newer on 2026-10-19 |
| Data still healthy / counts | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` (813 cards, 218/223 old, newest 2026-06-18 — all drift with TTSCardMaker) |
| old_stats byte trace | `for c in 793abf1 e380cee cca0aaf eb9aa84 637698b 76baa15 e5f5393 532b34f d31b948 8753489 284d13c; do echo "$c $(git cat-file -s $c:old_stats.pkl)"; done` |
| Whitespace-saga diffs | `git show 1c26747 3bbaa2b 4bcee6b -- commands/psd_analyzer.py`; surviving cleanup line: `grep -Fn "ability_text = sub" commands/psd_parser.py` (:778 on 2026-10-20) |
| MISMATCH lifecycle | `git show fed8a83 \| grep -A8 MISMATCH`; `git show 081b1fd \| grep -B3 MISMATCH`; leftover-append: `grep -n 'Append remaining types' commands/psd_parser.py` (:876 on 2026-10-20) |
| Archive guard | `git show e7befd5`; current: `grep -n -A3 'def _history' commands/card_data.py` |
| Save gate ("am forehead") | `git show 4e03190 \| grep 'num_new > 0'`; `git show fb47b5d \| grep 'num_updated > 0'` |
| Deleted artifacts list | `git log --diff-filter=D --name-only --format='DELETED-IN %h %ad %s' --date=short \| head -40` |
| Cython include/exclude + directives | `sed -n '9,45p' setup.py`; straggler check: `grep -rn '\[-1\]' commands/*.py` (expect empty) |
| Line numbers cited (drift-prone) | since the 2026-10-20 split: `grep -n 'def update_stats' commands/repo_traverser.py` → 59 (since 2026-10-22); `grep -n '_gap_pattern = \|MISSPELT_CARD_TYPES = \|ABILITY_EXCLUSIONS: ' commands/psd_parser.py` → 318 / 23 / 894 (1827 / 686 / 50 / 1262 in psd_analyzer.py before the split) (`relpath(full_path` and `expanduser(LOCAL_DIR_LOC)` left with local mode; the 2026-07-11 values quoted in Entries 1-8 were relpath-line 1002 / 1173 / expanduser 1207 / 348 / 42 / 649) |
| Commit dates quoted | `git log --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M' --no-walk <hash>...` |
| Entries 10.29-10.30 present | `grep -n "^def save_pickle\|^_stats_lock\|^_old_stats_lock" commands/card_models.py` (20, 16, 17); `grep -n "^_alias_lock" commands/query_card.py` (33; `RESULTS_PER_MESSAGE` gone since 2026-10-23); `grep -n "tasks.loop" main.py` (806 at 1 s; 823, 847, 884 at seconds=0, paced by the wake-up wait) |

2026-10-20 (later): the optimization-pass reports moved from `~/Documents/ac-optimization/` to `docs/Finished/optimization-pass-2026-10-20/`
(`docs/Finished/optimization-pass-2026-10-20/`); its scripts to `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/`.

2026-10-21: Entries 10.29-10.30 added; main.py, card_models.py, query_card.py and help.py line numbers re-pointed. Review: archive-branch and "at the time" main.py numbers kept as they were; bench path spelled out.

2026-10-21 (audit fixes): entry 10.30's query_ability citation re-pointed (:577-623) and its one-result-without-a-PNG hole noted as fixed.

2026-10-21 (titles): psd_parser.py, repo_traverser.py and card_models.py citations re-pointed (PARSER_VERSION 2 reads the printed title; `EXCESSIVE_STAT_EXCLUSIONS` deleted).

2026-10-21 (review): psd_parser.py citations re-pointed (+7 after `_printed_title`) after the WX-78 title fix.

2026-10-21 (S6/S7 gate): main.py, repo_traverser.py and help.py line numbers re-pointed after the /bake_cards block, the bake_lock in update_stats and the /bake_cards help entry; nothing else in this skill changed.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-22 (data build): the traverser's code moved to the uncompiled commands/card_data.py; the Entry 10 fix citations and the re-verification greps re-pointed (history is kept as it was).

2026-10-23 (unbatch): entry 10.30's /query_ability listing reverted to one message per card; the 10.29-10.30 re-verification row re-pointed.

2026-10-25 (step 10, work/pickles): Entry 10.32 (the pickles, /push, `CARD_SYNC`, the legacy /update_stats and their writers retired); the entries before it stay as history.

2026-10-25 (grid): entry 10.30: /query_ability grids.
