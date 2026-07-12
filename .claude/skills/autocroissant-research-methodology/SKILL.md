---
name: autocroissant-research-methodology
description: Load this skill whenever the question is "how do I know I'm right" in AutoCroissant — before promoting any hypothesis, mechanism, root cause, fix, or new idea to "accepted", and whenever a task involves the words hypothesis, experiment, evidence, root cause, investigate properly, when to stop, dead end, promote an idea, or retire an idea. It is the THINKING discipline calibrated to this hobby project (prevent expensive mistakes, not add ceremony) — the evidence bar (one mechanism must explain ALL observations including the negatives, worked removeprefix example), predict-the-numbers-BEFORE-running (the cca0aaf/637698b old_stats balloons as the cost of eyeballing), assigned adversarial refutation with the standard parser counter-case table (the Feb-8 whitespace saga as the cost of skipping it), the idea lifecycle from hunch through getattr-default experiment to adopted default or dated retirement (retirement is a success outcome; verified retirement list included), where good ideas historically came from in this repo, and hobby-calibrated stopping rules. The acceptance PROCEDURE (sandbox, goldens, pickle-push gate) lives in autocroissant-validation-and-qa; the extraction campaign's specific gates live in autocroissant-psd-extraction-campaign.
---

# AutoCroissant research methodology: from hunch to accepted result

Written 2026-07-11. All hashes, timestamps, diffs, and counts below were verified against this
repo (192 commits) on that date with read-only git and the diagnostics scripts.

**Calibration, up front.** This is a hobby project. The owner's stated doctrine (2026-07-11) is
"keep it fun — no heavy process." So this skill is NOT a process manual. The discipline below
exists for exactly one reason: to prevent EXPENSIVE mistakes — and this repo's expensive
mistakes are known and ranked by the owner: (1) pickle data corruption, (2) Discord API quirks.
Everything here is sized so it costs minutes, not evenings. When a rule below feels like
ceremony for the task at hand, skip it — EXCEPT where the step guards a pickle write or a push,
because that is where this repo has actually lost data (see the balloon story in section 2).
Every rule in this file carries the incident that earned it; a rule without an incident did not
make the cut.

Jargon used once and defined here:
- **mechanism** — a proposed cause-and-effect story ("X happens because Y does Z").
- **negative observation** — something that did NOT happen but would have if a candidate
  mechanism were true. The strongest evidence in this repo has been negative.
- **sandbox** — a scratch copy of this repo including pickles; experiments write there, never
  to the live pickles. Procedure owned by autocroissant-validation-and-qa.
- **golden card** — a PSD with certified expected parse output. Inventory owned by
  autocroissant-validation-and-qa; scripts by autocroissant-diagnostics-and-tooling.

## When NOT to use this skill

| You actually need | Go to |
|---|---|
| The extraction campaign's numbered phases, gates, and branch-on-observation instructions | autocroissant-psd-extraction-campaign |
| The acceptance PROCEDURE: sandbox setup, golden certification/nomination, diff_stats gate before a pickle push | autocroissant-validation-and-qa |
| Full incident write-ups (removeprefix bug story, balloon post-mortem, dead-end catalog) | autocroissant-failure-archaeology |
| Commit/push/revert gates, the PICKLE commit convention, never-commit list | autocroissant-change-control |
| How to run/interpret parse_one, gap_trace, diff_stats, inspect_pickle, dump_psd_layers | autocroissant-diagnostics-and-tooling |
| Step-by-step analysis recipes (layer dump → bbox → gap map, etc.) | autocroissant-analysis-toolkit |
| WHAT to research next (the two long-horizon ambitions) | autocroissant-research-frontier |

Division of labor in one line: **this skill is the thinking discipline ("how do I know I'm
right"); autocroissant-validation-and-qa is the checklist you execute once you think you are.**

## 1. The evidence bar: one mechanism must explain ALL observations, including the negatives

**Rule: enumerate observations first, mechanisms second. Include the negatives — what did NOT
break. A mechanism that needs one exception per observation is wrong. A mechanism that turns a
negative observation into a forced conclusion is probably right.**

### Worked example: the removeprefix root cause (2026-07-11)

Observations were listed before any mechanism was argued:

| # | Observation | Kind |
|---|---|---|
| O1 | `commands/psd_analyzer.py:1002` computes `relative_path = full_path.removeprefix("TTSCardMaker").strip('/')`, but `walk()` yields absolute paths (`/Users/.../Desktop/TTSCardMaker/...`), so `removeprefix` is a no-op and `classify()` sees top folder "Users" → every card UNKNOWN type, wrong stored path | code read |
| O2 | The line was introduced by f7c915c "cythonized the code" (2025-12-02) — the bug has existed for over seven months | git |
| O3 | **NEGATIVE:** stats.pkl today is healthy — 813 cards, **0** non-repo-relative paths, exactly 1 problem card (`inspect_pickle.py`, 2026-07-11) | measurement |
| O4 | The slash-command default is `use_local_repo=True` — the buggy path is the DEFAULT path | code read |

Candidate mechanisms, tested against the whole table:

- **"classify() is buggy"** — explains nothing cleanly: classify is shared by local and remote
  modes, so a buggy classifier corrupts every sweep. Fails O3 outright unless you invent an
  exception ("...but somehow the pickle stayed clean"). Rejected.
- **"local mode computes wrong paths"** — explains O1, O2. Alone it fails O3: seven months of
  default-local sweeps would have corrupted the pickle. O3 FORCES the completion: **production
  only ever ran remote mode** (`use_local_repo:False`), despite the local default. No exceptions
  needed anywhere in the table.

Confirmation by prediction (see section 2): if sweeps were remote, card timestamps are
last-COMMIT dates from the GitHub API, so cards touched in the same TTSCardMaker commit must
share second-exact identical timestamps — a pattern local file mtimes cannot produce. Measured
2026-07-11 and confirmed (second-exact clusters found); the executable fingerprint experiment
and its numbers live in autocroissant-analysis-toolkit Recipe 2. Mechanism accepted; full story
owned by autocroissant-failure-archaeology; the operating consequence (`use_local_repo:False`
until fixed) owned by autocroissant-run-and-operate.

### The corollary: finding one bug is not evidence there isn't a second

f7c915c introduced **two** unknown-type bugs in the same mechanical negative-index purge
(Cython `wraparound=False`):
- `relative_path.split('/')[:-1]` → `dirname(relative_path)` in `classify` — `dirname` returns
  a STRING, so `folders[0]` became the first *character* of the path → UNKNOWN for everything.
  Fixed six weeks later by fb47b5d (2026-01-15, "fix bug with unknown type...").
- `full_path.split("TTSCardMaker")[-1]` → `removeprefix("TTSCardMaker")` — still live as of
  2026-07-11 (psd_analyzer.py:1002).

Fixing the dirname bug made the UNKNOWN symptom disappear in the mode being tested, and the
second bug sat unnoticed for another six months. If the observation table had included "which
modes were exercised after the fix?", the gap would have been visible. **Do not stop at the
first mechanism that explains the symptom you happened to look at; stop when the table has no
unexplained rows.**

## 2. Predict the numbers BEFORE you run

**Rule: write down the expected number/output, then run, then compare. Anything unpredicted is
a finding — never rationalize it after the fact. If you cannot state an expected value, you do
not yet have a hypothesis; you have a hunch (go back to cheap read-only experiments).**

What this looks like here, concretely:

- **Golden-card expectations recorded before re-parses.** The certified expected values (Mini
  Doomer: minion, 2 stars, hp5/def2/atk5/spd5, ability ends "(BOT) Die."; The Freezer: ability
  byte-identical to its stats.pkl entry; The Lich King: 1 gap vs 1 type → "Minions inherit
  [undead].") exist precisely so a parser experiment starts from a written prediction. (Values
  here are snapshots of 2026-07-11 — the canonical current expected values, full inventory, and
  certification live in autocroissant-validation-and-qa §3; if one stops matching, check there
  before assuming a regression.) Scripts: autocroissant-diagnostics-and-tooling.
- **The campaign's Tier-A pattern** (autocroissant-psd-extraction-campaign): after the path fix,
  "a no-op local resweep must yield a diff_stats result where ONLY `timestamp` changes." That is
  a prediction of the ENTIRE diff, made before running; any non-timestamp field change falsifies
  the fix. Copy this pattern for any sweep-shaped experiment: predict the whole
  added/removed/modified triple, not just "it should be fine".
- **Sweep arithmetic.** Before any update sweep: "this run should touch N cards, so old_stats
  gains at most N archived versions and stats.pkl card count moves by exactly (new − deleted)."
  Cheap to state, and it is the exact prediction that would have caught the worst incident in
  the repo's history:

### The cautionary tale: eyeballing fails, and it fails SILENTLY

- **cca0aaf** (2025-11-10 00:26) pushed a PICKLE snapshot in which old_stats.pkl had ballooned
  **3196 → 12041 bytes** (duplicate-archiving bug family). Nothing about the push looked wrong
  to a human; it took until **eb9aa84** (same day, 20:02) to revert it. A written pre-push
  prediction ("old_stats grows by ≤ the number of changed cards") fails on sight against a 4x
  size jump.
- The part that proves the point: **637698b** (2025-11-25, fifteen days after the revert)
  committed a SECOND, nearly identical balloon — old_stats.pkl **3196 → 12039 bytes**. Nobody
  caught it. It was **never reverted** (verified: the next commit touching old_stats.pkl is
  76baa15, 2026-01-15, no revert in between). The first balloon was caught only because the
  symptom got noticed downstream; the second sailed through on eyeballs alone. `diff_stats.py`
  (autocroissant-diagnostics-and-tooling) exists to turn "the pickle looks fine" into numbers —
  but the numbers only protect you if you wrote the expected ones down FIRST.

### Mini-example: the same rule applied to code review

fb47b5d (2026-01-15 17:23) added the save gate `if (num_updated > 0): save` intending "only
write pickles when something changed". Predict-then-check question: *what value does
`num_updated` take on a no-change run?* Answer, from the code: it increments once per traversed
PSD (psd_analyzer.py:1046 — it is the progress counter), so the gate is always true — a no-op.
The owner caught it within 20 minutes: **4e03190** (17:43, message "am forehead") switched the
gate to `num_new` (today at psd_analyzer.py:1222). Two morals: (a) before shipping any
gate/threshold, state the value you expect the variable to hold in the boring case — if you
can't, you don't know what it counts; (b) commit messages here are sometimes jokes ("am
forehead") — verify diffs, not messages.

## 3. Adversarial refutation — assigned, not hoped for

**Rule: before a mechanism or fix is promoted, someone is explicitly ASSIGNED to break it — a
second person, or a second model session given the artifact and the instruction "find the card
or case that breaks this." "Nobody objected" is not refutation; refutation is a named attempt
that failed.** For a hobby repo this costs one message to a friend or one extra model session,
and roughly ten minutes with the scripts.

### The cost of skipping it: the 2026-02-08 whitespace saga (timestamps verified)

In brief (the full 8-commit chronology — 7 parser commits + the mid-saga PICKLE d31b948 — is
owned by autocroissant-failure-archaeology Entry 2): 04:34, 1c26747 adds `\s{2,}` → single-space
collapsing to fix 2 cards; 05:33-06:38, three iteration commits narrow and re-narrow it (first
exception within the hour) while a PICKLE snapshots data parsed by the doomed code; 16:23-19:12,
a collateral-damage fix lands (c7b3191), then 3bbaa2b "Fix the issue of collapsing spaces"
REMOVES collapsing entirely and 4bcee6b cleans up.

Seven code commits, ~15 hours, one day — and the approach was doomed at 04:34, because runs of
3+ spaces ARE the type-injection signal (`\s{3,}` gap pattern): collapsing whitespace destroys
the very thing the injector reads. Ten minutes of assigned refutation against the standard
counter-case list would have killed it before the first push. (The bug cards from this saga were
then turned into goldens — see section 5.)

### The standard counter-case list for parser changes

Run every row before promoting any change to commands/psd_analyzer.py parsing/injection.
Expected values and byte-comparison procedure: autocroissant-validation-and-qa. All commands are
safe/read-only and run from the repo root; `$S` = `.claude/skills/autocroissant-diagnostics-and-tooling/scripts`.

| # | Counter-case | What it refutes | How to run (verified 2026-07-11) |
|---|---|---|---|
| 1 | Mini Doomer (golden) | Multi-type injection + minion stats/stars | `python3 $S/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd` |
| 2 | The Freezer (golden) | Types at line START and multi-line gaps — the exact class 0490195 (2026-01-24, "types at beginning/end of ability not being injected") existed for; ability must stay byte-identical | `python3 $S/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"` |
| 3 | The Lich King (golden) | Single small gap + bbox prune threshold | `python3 $S/gap_trace.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"` |
| 4 | The 23 exclusion-list cards (18 ABILITY_EXCLUSIONS + 5 EXCESSIVE_STAT_EXCLUSIONS, psd_analyzer.py:642-668, counts as of 2026-07-11) | Each is a documented rule-breaker; "cleaner" mechanisms reliably re-break one | Sandbox sweep; problem output must not grow (procedure: validation-and-qa) |
| 5 | One rulebook page | Rulebook takes ANY text layer as ability (no layer named "ability"); also "20 Creature Types" (MISSPELT TYPE: tornado) must remain the ONLY *pickled* (parse-time) problem — the full-validation baseline is 5, per autocroissant-validation-and-qa §2's two-surface distinction | `python3 $S/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/00_Table_of_Contents.psd` — expect type rulebook, ability starting "TABLE OF CONTENTS" |
| 6 | One MDW card | MDW ability must stay `None` (psd_analyzer.py:282) — refutes any change that invents ability text (c7b3191's bug class). MDW is skipped by sweeps (EXCLUDE_FOLDERS, psd_analyzer.py:31), so only parse_one exercises it | `python3 $S/parse_one.py ~/Desktop/TTSCardMaker/MDW/cardback.psd` — expect type MDW, ability `<none>` (its NO ABILITY LAYER note is expected for a cardback) |

If a proposed mechanism survives all six rows AND a sandbox sweep whose diff_stats output
matches a written prediction (section 2), it has met the bar. Promotion itself (commit/push)
routes through autocroissant-change-control.

## 4. The idea lifecycle

Mapped to what this repo ACTUALLY has — no invented machinery:

```
hunch
  │  (free)
  ▼
cheap read-only experiment          — diagnostics scripts, git log/show, a scratch one-liner;
  │  (minutes, zero risk)             never writes a pickle, never runs the bot
  ▼
hypothesis with predicted numbers   — section 2: expected values written down first
  │
  ▼
sandbox experiment                  — scratch copy of repo + pickles; sweeps and parser edits
  │                                    happen THERE (procedure: autocroissant-validation-and-qa)
  ▼
config-field experiment             — new behavior behind a config.py field read as
  │                                    getattr(config, "field", old_default), so every machine
  │                                    without the field keeps old behavior
  │                                    (how-to + checklist: autocroissant-config-and-flags)
  ▼
adopted default                     — gates: autocroissant-change-control
  or
retirement                          — dated entry in autocroissant-failure-archaeology
```

**Honesty note:** this repo has NO feature-flag system, no experiment framework, no A/B
anything. The entire experiment mechanism is (a) a `getattr(config, "field", default)` read —
the live pattern at diffusion.py:24-29, music_player.py:37/54/61, psd_analyzer.py:34,
utils.py:21, query_card.py:24 — plus (b) runtime `/set_*` commands that do NOT persist across
restart (autocroissant-config-and-flags). That is enough for a hobby project. Do not add
ceremony beyond it.

### Retirement is a SUCCESS outcome

An idea that is tried, found wrong, and retired WITH A DATED WRITE-UP is a completed research
result. This repo already practices it (all verified in git):

| Retired thing | Lifespan | Evidence | Takeaway |
|---|---|---|---|
| `commands/Text_Export.jsx` (Photoshop-side text export) | 2023-11-17 → 2025-01-19 | added 2a5a799, deleted 7fc6f63 | Superseded by parsing PSDs directly in Python |
| descriptions.pkl era | → 2025-03-10 | deleted bb58387; schema later replaced wholesale by CardInfo dataclasses (366c8d9 "Big massive refactor") | Data schemas get retired too; pickles are code-coupled |
| requirements-linux*.txt (per-OS requirement files) | 2024-08 → 2024-08-18 | deleted 3f9a96d | Wrong split axis; the split that stuck is core-vs-AI (requirements.txt vs requirements2.txt, reorganized 2026-07-11 — autocroissant-build-and-env) |
| TYPE/WHITESPACE MISMATCH validator problem | 2026-01-24 → 2026-01-30 | added fed8a83, removed 081b1fd | Retired when injection improved; the campaign lists restoring it as a ranked candidate — retired ≠ forbidden forever, IF the retirement was documented |
| Whitespace collapsing in ability text | 2026-02-08 04:34 → 16:33 | 1c26747 → 3bbaa2b | Permanently fenced off: gaps are the injection signal |

The failure mode a dated retirement entry prevents: **re-invention**. The MISMATCH validator and
the collapsing idea are both natural things a fresh session would propose again; the archaeology
entry is what turns "let's try collapsing spaces" into a two-minute lookup instead of a repeat
of Feb 8.

## 5. Where good ideas have actually come from here

Verified provenance — useful because it tells you where to LOOK for the next one:

1. **Bug tails.** The type-injection engine matured almost entirely as the tail of named-card
   bugs: 0490195 (2026-01-24, types at line start/end), the 2026-01-30 placement batch
   (081b1fd, d0bb28c — TYPE_REGION_RATIO=0.5, dd800bc, 34be7e5), the Freezer/Mini-Doomer saga
   (1c26747→4bcee6b). And the bug cards then became the golden inventory — today's regression
   evidence IS yesterday's bug list. When you fix a card, always ask "what CLASS of cards is
   this?" — that question is where the feature came from.
2. **Friend-group requests.** `/react` arrived in d699b3f (2026-01-26, "added a new command for
   reacting to messages"); reminders that execute bot commands on schedule (the
   `slash_registry.get(cmd_name)` dispatch, analytics.py:209; hardened by 105198f, 2025-12-02,
   "reschedules reminders if they're past due"). The bot serves a friend group; their asks are a
   proven idea source.
3. **Operational pain.** Lazy torch/diffusers imports bf9478e (2026-01-26) — born from the bot
   paying AI startup cost when AI wasn't wanted; it grew into the whole AI-boundary doctrine
   (autocroissant-ai-boundary). Likewise the 2026-07-11 requirements split (core vs AI) came
   from two-machine install pain, not from a roadmap.

**Implication — keep a low-friction intake.** SUGGESTION (not an existing convention): a pinned
Discord channel in the friend group's server, or a NOTES.md scratch list, where one-line ideas
and bug-tail observations land the moment they occur. The bar for capturing an idea should be
one sentence; anything heavier (issue tracker, templates) is ceremony this project does not want.

## 6. When to STOP investigating (hobby calibration)

- **Time-box by default: one evening per mechanism.** The Feb-8 saga burned ~15 hours in a day
  on an approach that was refutable at minute ten. The tell was visible early: the FIRST
  exception (d6d1278 narrowing `\s{2,}` to `[ \t]{2,}` at 05:33, under an hour in) was the
  moment to step back, not push through.
- **"Documented dead end" is a complete, successful outcome.** Write the dated
  failure-archaeology entry (symptom → what was tried → why it's wrong → repro one-liner) and
  move on. The retirement table in section 4 is proof this repo works that way. Leave the repro
  one-liner so the next session starts at your frontier, not from zero.
- **One mechanism per session** (especially for cheaper/smaller model sessions): do not chase
  two hypotheses at once — interleaved experiments cross-contaminate the observation table and
  you end with two half-tested mechanisms, which per section 1 is zero tested mechanisms. Kill
  or confirm one, then start clean.
- **The >2-special-cases rule:** if your fix needs a third special case, the mechanism is
  probably wrong — return to section 1 and re-enumerate observations. Precedent: collapsing
  needed its first exception within an hour (d6d1278) and a second-order bug fix the same
  afternoon (c7b3191) before being abandoned (3bbaa2b). Standing warning: the 23-card exclusion
  list is the fossil record of accumulated special cases, and the owner's stated ambition
  ("perfect extraction") is to DELETE it — never grow an exclusion list just to keep a
  mechanism alive.
- **Stopping is not concluding.** A stopped investigation states what is known, what is ruled
  out, and what the next cheap experiment would be — it does not round "I ran out of evening"
  up to "it's probably X". Unproven ideas stay labeled open/candidate.

## Provenance and maintenance

Verified 2026-07-11 by direct git inspection and the read-only diagnostics scripts. Line
numbers drift; each volatile fact below carries its re-verification one-liner (run from
`/Users/michaelsrouji/Desktop/AutoCroissant`).

| Fact | Re-verify with |
|---|---|
| removeprefix live-bug line (was :1002) | `grep -n 'removeprefix("TTSCardMaker")' commands/psd_analyzer.py` |
| Slash default `use_local_repo=True` (O4; was main.py:361) | `grep -n 'use_local_repo' main.py` |
| f7c915c introduced BOTH the dirname-classify and removeprefix bugs (2025-12-02) | `git show f7c915c -- commands/psd_analyzer.py` — read the hunks touching `dirname` and `removeprefix` |
| dirname bug fixed by fb47b5d; save gate added there as `num_updated > 0` | `git show fb47b5d -- commands/psd_analyzer.py` — the `split('/')` hunk and the final gate hunk |
| Gate fixed to `num_new` 20 min later by 4e03190 "am forehead" (17:23:37 → 17:43:29) | `git show -s --format='%h %ci %s' fb47b5d 4e03190` |
| `num_updated` increments for EVERY traversed PSD (was :1046); gate now `num_new > 0` (was :1222) | `grep -n -e 'num_updated += 1' -e 'num_new > 0' commands/psd_analyzer.py` |
| stats.pkl healthy: 813 cards, 0 suspicious paths, 1 problem card, newest 2026-06-18 | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |
| Timestamp clustering confirms remote-mode attribution | the Counter one-liner and its expected numbers live in autocroissant-analysis-toolkit Recipe 2 |
| Balloon 1: cca0aaf old_stats 3196→12041; reverted same day by eb9aa84 | `git show --stat cca0aaf` then `git show -s --format='%ci %s' eb9aa84` |
| Balloon 2: 637698b old_stats 3196→12039 (2025-11-25), NEVER reverted | `git show --stat 637698b` then `git log --format='%h %ad %s' --date=short -- old_stats.pkl` (no revert after 637698b) |
| Feb-8 saga: 7 code commits 04:34:51→19:12:52, collapsing added 1c26747, removed 3bbaa2b | `git log --format='%h %ad %s' --date=iso 532b34f..4bcee6b` (exactly the Feb-8 commits) and `git show 3bbaa2b` |
| First collapsing exception (`\s{2,}`→`[ \t]{2,}`) in d6d1278 | `git show d6d1278 -- commands/psd_analyzer.py` |
| MISMATCH validator added fed8a83, removed 081b1fd | `git log -S 'WHITESPACE MISMATCH' --format='%h %ad %s' --date=short` (returns exactly those two commits) |
| Exclusion-list sizes (5 + 18 = 23 as of 2026-07-11) | `grep -n -A30 'EXCESSIVE_STAT_EXCLUSIONS' commands/psd_analyzer.py` (count the quoted names in each set) |
| MDW classify branch (`ability: None`, was :282) and EXCLUDE_FOLDERS (was :31) | `grep -n -e '"MDW"' -e 'EXCLUDE_FOLDERS =' commands/psd_analyzer.py` |
| getattr(config, ..., default) is the only flag mechanism | `grep -rn 'getattr(config' commands/ main.py` |
| Retirements: Text_Export.jsx (7fc6f63), descriptions.pkl (bb58387), requirements-linux* (3f9a96d) | `git log --all --diff-filter=D --format='%h %ad %s' --date=short -- '*Text_Export.jsx' descriptions.pkl 'requirements-linux*'` |
| Idea provenance commits: 0490195, d699b3f, bf9478e, 105198f | `git show -s --format='%h %ad %s' 0490195 d699b3f bf9478e 105198f` |
| requirements.txt is core-only (2026-07-11 reorg, intentionally uncommitted) | `grep -ic -e torch -e diffusers -e transformers requirements.txt` (prints 0; exit code 1 because no matches is expected) |
| Counter-case PSDs exist locally | `ls ~/Desktop/TTSCardMaker/MDW/cardback.psd ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/00_Table_of_Contents.psd` |

Maintenance rules:
- If the removeprefix bug gets fixed, update section 1's "still live" wording and defer the fix
  story to autocroissant-failure-archaeology and autocroissant-psd-extraction-campaign — do not
  grow the worked example here; its value is the reasoning shape, which stays true.
- If a golden or exclusion list changes, the numbers in the counter-case table (row 4) must be
  re-counted; certification itself belongs to autocroissant-validation-and-qa.
- New retirements go in section 4's table ONLY with a hash and date, and only after the
  failure-archaeology entry exists (one home per story; this table is the index, not the story).
- This file's incidents are frozen history (hashes don't drift); the volatile parts are line
  numbers, counts, and "still live" claims — re-verify those with the table above before citing.
