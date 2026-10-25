# Test suite cut and xdist

**Status:** planned 2026-10-24, owner's answers in; not built. Queue item 2 (after the parts editor). Measurements:
`~/Documents/Tools/wt/scratch/tests-plan/PLAN.md` (at main 164b4a4; `tools/cuts.py` there is the cut as data).

**Goal.** A suite that runs in about a minute in parallel and tests only what protects something (owner, 10-24: "We
have too many tests - stop bloating our test suite").

**Rulings that bind it**
- Run it with pytest-xdist; new work adds at most a few focused tests (memory `lean-tests-and-verification`).
- Site tests (owner, 10-24): keep about 15 site smoke tests in AutoCroissant (PLAN §3.4's list, one per site piece);
  the other site files go.
- `commands/discord_archive.py` and `commands/discord_match.py` are retired with their tests
  (`tests/test_discord_archive.py`, `tests/test_discord_match.py`). `commands/discord_edits.py created` reads the
  archive's files only and stays.
- Dead code is deleted, not archived (memory `fonts-and-dead-code`).

## Numbers

| | tests | `-n auto` wall (load ~45) |
|---|---:|---:|
| 10-24 (164b4a4) | 1,690 | 110 s; serial 40 min |
| 10-25 (e6a5c54), collected | 1,923 | not measured |
| After the cut (PLAN §5, re-derive) | about 620 + 15 smoke | about 40-50 s |

The suite grew after the plan (BUG 2, Python 3.14, split, retire). Re-run the plan's `tools/cuts.py` on a fresh junit file of
current main before cutting; the rules below decide new files.

## Steps (one commit each, straight to main)

1. **xdist and the parallel fixes:** `pytest-xdist>=3.6` in `requirements-dev.txt`; the run line in `tests/conftest.py`
   (today it still names `ac-py314-psd125`, where the bake tests skip) becomes
   `~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests`; PLAN §2.3 (`_state` compares object counts, not
   `size`) and §2.4 #1-6, #9-11 (`GIT_NO_LAZY_FETCH` in the clone readers, pool caps, monkeypatched globals, the
   SIGTERM handler, timeouts, `core.hooksPath=/dev/null` and `core.looseCompression=0` through `GIT_CONFIG_*`, stubbed
   retry sleeps). Also `pytest.ini` keeps only failed runs' temp dirs (70 GB of pytest temp on the Mac).
2. **Shared helpers to `tests/helpers.py`** (PLAN §2.4 #7) before any test file is deleted.
3. **The cut, per file or group** (PLAN §3.1-3.5): DEAD, FORK, OTHER REPO, DUP go; MERGE folds asserts into the
   survivor; SHRINK keeps 1-3 cases. Site: keep the 15 smoke tests, delete the rest. Delete with their tests:
   `commands/discord_archive.py`, `commands/discord_match.py`, `commands/png_recompress.py`, txt2_repair's
   `--check`/`--in-place`, the psd_details backfill CLI. `commands/history_backfill.py` goes with step 10
   (`work/pickles`).
4. **Speed:** the font-index stub and per-session template repos cloned with `--local` (PLAN §2.4 #12-13).
5. **The rule against re-growth**, one paragraph in `.claude/skills/autocroissant-validation-and-qa/SKILL.md` §1c: a
   change brings tests only for a gate, one happy path of a new command, or a pickle/data-safety property; never for
   TTSCardMaker's code, workflow YAML text, the fork's behaviour, finished migrations, real-card replays beyond the
   goldens, or more than 3 parametrize cases.

Kept on purpose: PLAN §4 (data save/diff gates, history recovery, the golden cards Mini Doomer, The Freezer, The Lich
King, one happy path per command, the bake and guard integrity gates). Rows for retired commands drop out.

## Verification

Two `-n auto` runs on the fork venv with 0 failures and no silent skips of the bake tests; every PLAN §4 test still
collected. One light review of the deletions.

## Defaults (PLAN §7 items 3-6, not asked)

The 7 recovery-rule tests go (the recovery ran); goldens 39 -> 11; the two CLIs above are deleted. test_stars_types'
hold is moot (BUG 2 landed).
