---
name: autocroissant-validation-and-qa
description: "Load this skill BEFORE judging whether any AutoCroissant change is safe or finished — whenever the task sounds like \"is this change safe\", \"test this\", \"validate\", \"verify\", \"acceptance\", \"QA\", \"golden\", \"regression\", \"how do I know it works\", \"did the parser break anything\", or before pushing/committing pickles (/push, /update, PICKLE commits) or touching commands/psd_analyzer.py (parser, CardValidator, exclusion lists). This repo has NO CI and only an offline pytest suite (tests/); this skill defines what counts as evidence — the evidence hierarchy (golden-card byte-equality > full-sweep diff_stats deltas > validator problem-card counts > exclusion-list sizes > eyeballing Discord output), the certified golden-card inventory (Mini Doomer, The Freezer, The Lich King) with expected values and the nomination procedure, the sandboxed parser-change acceptance procedure with the exact update_stats invocation and its GitHub-API cost, the two 2026-10-19 regression methods (old-vs-new parser over every PSD with no pickles or API; sandboxed forced LOCAL sweep compared field-by-field with the stored remote data, with that day's numbers: 813 cards, 0 unknown, 59 ability + 2 star diffs explained as upstream PSD edits), the pickle-push diff gate, how to add a CardValidator check without repeating the TYPE/WHITESPACE MISMATCH mistake, and the concrete definition of \"regression\". Baselines certified 2026-07-11; regression methods re-measured 2026-10-19."
---

# AutoCroissant validation & QA: what counts as evidence

Certified 2026-07-11; section 2's problem list, section 4 and the new section 4.1 updated 2026-10-19
(icon-font parsing, the fixed local-path bug, the old-vs-new parser method). This repo has **no
CI and no README**, and only one offline test suite (row 1c below). The pickles
(`stats.pkl`, `old_stats.pkl`) ARE the production database and are committed to git. The
owner's two costliest failure classes are (1) pickle data corruption and (2) Discord API
quirks. So QA here is not just "run the tests" — the suite is one row; QA is a small set of **measured numbers with
certified baselines**. A change is safe when the numbers say so, and unsafe when any number
moved without an explanation.

**"Looks right" is not evidence; numbers are.** Every acceptance decision in this project
reduces to comparing today's numbers against the certified baselines below.

Jargon used throughout (defined once):

- **golden card** — a specific PSD with certified expected parse output, chosen because it
  once exercised a real bug or algorithm edge.
- **sweep** — one full run of `update_stats()` over all ~904 PSDs in the TTSCardMaker repo.
- **sandbox** — a full local copy of this repo (pickles included) in a scratch directory,
  where a sweep can overwrite pickles harmlessly.
- **problem card** — a card for which the validator reports at least one problem string.
  There are TWO distinct problem surfaces (pickled vs computed) — see below; conflating
  them produces wrong baselines.
- **exclusion list** — a per-card allowlist inside `CardValidator` that silences a check
  for named cards. Registered debt against the owner's "perfect extraction" goal.

All commands assume `cwd = /Users/michaelsrouji/Desktop/AutoCroissant` (repo root) and use:

```bash
SCRIPTS=.claude/skills/autocroissant-diagnostics-and-tooling/scripts
```

Safety notes that govern everything below: never run `update_stats()` against the live
repo's pickles (sandbox only — procedure below); the diagnostics scripts in `$SCRIPTS` are
read-only and safe; `config.py` holds secrets — never print or commit it.

## 1. The evidence hierarchy (strongest → weakest)

| # | Evidence | How to measure | Baseline (2026-07-11) | What it proves |
|---|---|---|---|---|
| 1 | **Byte-identical golden-card parses** | `parse_one.py` / `gap_trace.py` output diff before vs after; ability text byte-compared to the stats.pkl entry | Golden inventory table in section 3 | The parser produces exactly the same output on cards known to exercise the hard paths. Byte-level: even one changed space in ability text is a real change (whitespace is load-bearing — the gaps are the type-injection signal; see the 2026-02-08 saga, 1c26747 through 3bbaa2b/4bcee6b) |
| 1b | **Old-vs-new parser equality over EVERY PSD** (added 2026-10-19) | section 4.1 method A: import the git-HEAD `psd_analyzer` and the working-tree one side by side, parse all card PSDs with both, compare every field | 2026-10-19: 813 PSDs, **0 differences** (the icon-font change); 2026-10-21: 813 PSDs, **159 differences, every one intended** (157 `display_name` from the printed title, `STATS TOO HIGH` on Sonic and Royal Eradicator Main Cannon) | The parser change is behavior-neutral on today's corpus, with no pickles, no network, no API budget. Stronger than a sweep diff for "did my code change anything", because it isolates the code from upstream PSD edits |
| 1c | **Offline pytest suite** (added 2026-10-20) | `/Users/michaelsrouji/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (about 13 s; no network, no config.py needed, never writes the pickles: `conftest.py` hashes them around the run) | 2026-10-21: **352 passed**, 18 fork-only skips (310 earlier that day, 238 on 2026-10-20, 163 before that day's optimization pass) | 38 parser goldens in `tests/goldens/` (the 3 certified ones included; a golden whose PSD changed upstream skips; re-pin with `tests/make_goldens.py` only after an accepted parser change), plus CardEdits, duplicates, the creature split, update_stats on a fake GitHub, repo_history, type emojis, the TTS save, reminder times and message splitting all still behave as today; since the optimization pass also the dispatch/edit loops and help coverage (`test_robustness.py`), the facade and pickle contract of the psd_analyzer split (`test_split_contract.py`) and the search/commit-lookup/ETag speedups against brute force (`test_network_speed.py`). Run it before every push (autocroissant-change-control class (a)). **Not a parser gate on its own:** 38 goldens cover part of the parser, so any change to the parser or a card module (`psd_parser.py`, `psd_analyzer.py`, `card_models.py`, `card_edits.py`, `repo_traverser.py`) still needs 1b (until Irithyll, 16_The_Field and Termination_Shock were added, `row_threshold` 40->60 and `STAR_CORE_ALPHA` 0.5->0.2 changed 16 of 813 cards and passed the suite) |
| 2 | **diff_stats deltas on a full sweep** | sandbox sweep, then `diff_stats.py before.pkl after.pkl` | HEAD vs working today: `813/813, added 0 removed 0 modified 0, exit 0` | Whole-database impact. Every added/removed/modified line must have an explanation; "removed" rows you didn't intend and mass path/type changes are corruption signatures |
| 3 | **Validator problem-card count** (must not increase) | pickled surface: `inspect_pickle.py --problems`; full surface: re-validation one-liner (section 2) | pickled: **exactly 1** ("20 Creature Types" — `MISSPELT TYPE: tornado`); full validation: **exactly 5** (the 1 above + 4 computed-only, listed in section 2) | Extraction quality did not regress anywhere the validator can see |
| 4 | **Exclusion-list size** | count `CardValidator.ABILITY_EXCLUSIONS` (`EXCESSIVE_STAT_EXCLUSIONS` was deleted on 2026-10-21: a stat above 10 is `STATS TOO HIGH` unless the card's Card Edits.json entry sets `allow_high_stats`, as Sonic's and Royal Eradicator Main Cannon's do) | **18** ABILITY_EXCLUSIONS (psd_parser.py:950 on 2026-10-21), no stat list | Shrinking them is progress toward "perfect extraction" (owner doctrine); growing them is new registered debt |
| 5 | **Eyeballing Discord output** | run a command, look at the message | none | Weakest. Catches only gross breakage; never sufficient alone. Acceptable as a final smoke check AFTER 1–4 pass, never as a substitute |

If evidence levels conflict, the stronger level wins. A change that "looks great" in
Discord but produces one unexplained `diff_stats` row is NOT accepted.

## 2. The two problem surfaces (read this before counting problems)

Verified against `commands/psd_analyzer.py` on 2026-07-11; Channel A re-verified 2026-10-19.
Problems travel through two distinct channels, and they have different baselines:

**Channel A — parse-time problems, PERSISTED in the pickle.**
`CardInfo.problems` is a dataclass field (card_models.py:94). During parsing these problem
kinds are appended to it (line numbers as of 2026-10-19):
- `NO ABILITY LAYER` (psd_parser.py:414), when no ability text layer exists.
- `MISSPELT TYPE: <name>` (psd_parser.py:549), for visible pixel layers named in
  `MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']` (psd_parser.py:24).
- Since 2026-10-19, four icon-font problems, added once each through `_add_problem`
  (psd_parser.py:623): `UNKNOWN ICON TOKEN: [x]`, `UNKNOWN ICON CHARACTER: U+XXXX`,
  `ICON TOKEN NOT IN STOOGEDINGS FONT: [x]` and `ICON TOKEN PARTLY IN STOOGEDINGS FONT: [x]`
  (psd_parser.py:656-688; meanings in impossibility-cards-reference §4.1). On 2026-10-19 no card
  PSD uses the font, and none fires on the corpus (method A in section 4.1), so the baseline below is
  unchanged. Expect them to appear as cards migrate; each firing is a real authoring mistake.

`StatsDatabase.save()` pickles the CardInfo objects themselves, so these problems persist into
`stats.pkl` and surface as the `"problem"` key in `to_dict()` (card_models.py:140).
`inspect_pickle.py --problems` counts exactly this channel.

**Channel B — computed validation problems, OUTPUT-ONLY, never pickled.**
`CardValidator.validate(card)` (psd_parser.py:975-1011 on 2026-10-21) builds a fresh local list from
computed checks — `UNKNOWN TYPE`, `ABILITY TEXT NOT FOUND`, `HP/DEF/ATK/SPD NOT FOUND`,
`STATS TOO HIGH` (via `_validate_stats`, psd_parser.py:1014) — then extends it with
the card's Channel-A problems and returns it. The return value is **never stored on the
card**: during a sweep it flows into `_format_problems` (repo_traverser.py:466, called :410)
and out to Discord/console (the archived local traversal on branch `archive/local-update-stats` has its
own copy of this flow). Since 2026-10-21 the same `validate()` runs in a second place, `commands/card_check.py`
(the push guard's card check, TTSCardMaker card-check.yml): per pushed PSD, after `CardEdits.apply` as in the sweep
(repo_traverser.py:387-388), into a report, a commit comment and a Discord line; `tests/test_card_check.py` pins the clone's
cards to the sweep's output, so a validator change shows there first (autocroissant-run-and-operate §7).
So a sweep's printed problem list = Channel A + Channel B; the pickle stores only Channel A.

**One side effect to know:** `validate()` MUTATES `card.problems` — it removes
`NO ABILITY LAYER` when the card is in `ABILITY_EXCLUSIONS` (psd_parser.py:1005-1008).
And `validate()` only runs during a sweep when `output_problematic=True`
(repo_traverser.py:391-392). Consequence: a sweep run with `output_problematic=False`
would leave `NO ABILITY LAYER` in the pickled problems of the 18 excluded cards,
inflating the Channel-A count by up to 18 with zero real regression. **Always sweep with
`output_problematic=True` (the default) when measuring.**

Certified baselines (both measured 2026-07-11):

- **Channel A (pickled): exactly 1 problem card.** (The PSD was fixed upstream on 2026-10-22, TTSCardMaker 100511d3:
  its `tornado` layer is now `wind`; the next sweep drops this to 0 and adds `wind` to the page's types.)
  ```bash
  python3 $SCRIPTS/inspect_pickle.py --problems
  # 20 Creature Types [Auxiliary/Rulebook/20_Creature_Types.psd]: ['MISSPELT TYPE: tornado']
  ```
- **Full validation (A+B over the whole DB): exactly 5 problem cards.** Read-only; no
  sweep needed; safe to run anytime:
  ```bash
  python3 -c "
  import sys; sys.path.insert(0, '.')
  from commands.psd_analyzer import stats_db, CardValidator
  stats_db.load()
  bad = {n: p for n, p in ((n, CardValidator.validate(c)) for n, c in stats_db.stats.items()) if p}
  print('cards failing full validation:', len(bad))
  for n, p in sorted(bad.items()): print(' ', n, p)
  "
  ```
  Output today: `20 Creature Types` (MISSPELT TYPE: tornado), `Computer Virus`
  (HP/DEF/ATK/SPD NOT FOUND), and `Anubisath Guardian`, `Qiraji Soldier`, `Silithid`
  (ABILITY TEXT NOT FOUND — all three are 2026-06 World of Warcraft additions whose
  pickled `ability` is the empty string: an ability layer exists, so `NO ABILITY LAYER`
  did not fire, but the extracted text is empty). These 4 computed-only cases are
  known-open extraction issues, NOT noise — a fix that clears them is measurable progress.

**The acceptance gate is: neither count may increase.** A decrease is progress and must be
explained (which card, which fix), then the baselines in this file get updated.

## 3. The golden inventory (certified 2026-07-11; all three re-pinned 2026-10-22)

Goldens are the strongest evidence because each was chosen from a battle scar: The Freezer
and Mini Doomer WERE the bug cards of the 2026-02-08 whitespace saga (8 commits in one
day — 7 parser commits plus a mid-saga PICKLE — 1c26747 through 3bbaa2b "Fix the issue of
collapsing spaces"/4bcee6b; chronology in autocroissant-failure-archaeology Entry 2); The
Lich King pinned the single-gap + prune-threshold path until 2026-10-22, when the stoogedings
batch put both cards' inline types in the text as tokens (gap injection is still exercised by the
Twin Emperors, Irithyll and 00_Table_of_Contents goldens). All PSD paths are relative to the
TTSCardMaker clone at `~/Desktop/TTSCardMaker`.

**This table is the CANONICAL home of the golden expected values.** Other skills quote
them only as dated snapshots that defer here — when a golden legitimately changes, update
THIS table (and the date-stamps) first; the snapshots point back to this section.

| Golden | PSD path | What it certifies | Expected values (certified 2026-07-11) |
|---|---|---|---|
| **Mini Doomer** | `Auxiliary/Minions/Mini_Doomer.psd` | Minion classification; stars counted from visible pixel layers (two visible "Warpstar" smartobjects under a "Stars" group); 3 creature types; summed stat digits | `type=minion`, `stars=2`, `types=['wind', 'defensive_minion', 'light']`, `hp=5 def=2 atk=5 spd=5`, ability ends `(BOT) Die.`, validator problems `NONE` (re-pinned 2026-10-22: blob e0b11bab -> 197c3e76, TTSCardMaker 38c55bd3 (queue A gradient badges); every parsed field and the validator unchanged. The `(BOT)` is now a stoogedings badge run in Text/Ability with an `Ability FX: (BOT)` pixel layer (2165-2251) over it, so psd_details `icons` is `mixed` and the Ability layer's bbox bottom moved 2245 -> 2251 (psd_edit's padded region 2258 -> 2264), 96 layer records) |
| **The Freezer** | `Field/1 Stars/The_Freezer.psd` | Field classification; inline types at line START and line END (since 2026-10-22 as stoogedings `[token]` runs, icon layers hidden: 0 gaps, 0 below-midline icons; before, 4 gaps filled from 4 icon layers); parse output equals the stats.pkl entry byte-for-byte | `type=field`, `stars=1`, `types=['field']`, problems `NONE`, ability EXACTLY:<br>`For friendly creatures:`<br>`[water] Turns into [ice]. [ice]: +1 DEF`<br>`(1GCD) Choose a friendly creature to turn into a [ice] and lower its speed by 1.`<br>(re-certified 2026-10-22: equal to the stats.pkl entry character-for-character from PICKLE 3888392 on; the line read `(1PGT)` before the 2026-10-21 cooldown update, impossibility-simulator-rules §3; re-pinned 2026-10-22: blob 90fa61f4 -> 8505c4f0, TTSCardMaker 91fcda38, every parsed field and the validator unchanged) |
| **The Lich King** | `Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd` | Above-midline creature types beside an inline `[undead]` token (until 2026-10-22: single gap + single below-midline type, prune-threshold arithmetic) | `gap_trace.py` (2026-10-22, blob c69f7c24, TTSCardMaker 1c24dd48): above-midline creature types `['ice', 'undead']`; 1 text layer (y=1767) reading `Minions inherit [undead].`; 0 gaps, 0 below-midline icons (gap count 0 vs kept types 0, match); final line `Minions inherit [undead].`; `hp=7 def=6 atk=7 spd=5`, problems `NONE`. Through blob 1732d9bf it was: text layer y=1773, `Minions inherit<GAP:7>.`, icon `undead @ (x=734, y=2239)`, prune `max(746, 1200) = 1200`, kept `['undead']`; the parse was the same |

Commands to reproduce (each takes a few seconds; read-only):

```bash
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"
python3 $SCRIPTS/gap_trace.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"
```

**Equality rules.** A fresh `parse_one.py` run has `timestamp: 0.0` and no `author` —
those two fields come from traversal metadata (git commit dates / DB preservation,
psd_parser.py:355-361, which also keeps `series` and `subseries`; repo_traverser.py:387), not from the PSD. So:

- *Golden vs stats.pkl*: all fields must match EXCEPT `timestamp` and `author`; ability
  text is compared byte-for-byte.
- *Golden before-vs-after a change*: save both `parse_one` outputs to files and `diff`
  them — must be empty (both are fresh parses, so timestamp/author cancel out). This is
  the comparison the acceptance procedure uses.

### How to NOMINATE a new golden

1. **Qualify it.** A golden must have exercised a real bug or a distinct algorithm edge
   (precedent above: the Feb-8 bug cards became the goldens). Do not add cards that merely
   duplicate an existing golden's mechanism — keep the set small and orthogonal.
2. **Record it.** Run `parse_one.py` (and `gap_trace.py` if injection is involved) and
   save the full outputs with today's date and the repo-relative PSD path.
3. **Certify it.** Cross-check the parse against the current `stats.pkl` entry
   (`python3 $SCRIPTS/inspect_pickle.py "Card Name"`); confirm ability byte-equality and
   note any known deviations. If the card is currently mis-parsed, it is a CANDIDATE
   golden — record the current (wrong) output AND the intended output, clearly labeled;
   it graduates to certified when the fix lands.
4. **Register it.** Add a row to the table above with the certification date and what
   mechanism it pins. Update the frontmatter/baseline dates.

## 4. Parser-change acceptance procedure (decision-gated)

Use this for ANY change to parsing/classification/validation (`commands/psd_parser.py`, or any card module:
`psd_analyzer.py`, `card_models.py`, `card_edits.py`, `repo_traverser.py`)
behavior, and for anything else that could alter what a sweep writes. Per-iteration
debugging uses goldens and `gap_trace.py` (cheap); the full sweep is the **expensive
promotion gate you run ONCE at the end**, not a per-iteration tool.

**Step 0 — SANDBOX. Never experiment against live pickles.**
"Sandbox" means: copy the entire repo (pickles included) to a scratch directory and run
everything there:

```bash
cp -r /Users/michaelsrouji/Desktop/AutoCroissant /tmp/ac_sandbox
cd /tmp/ac_sandbox
```

Why cwd is load-bearing: `STATS_PKL = "stats.pkl"` and `OLD_STATS_PKL = "old_stats.pkl"`
are cwd-relative (global_config.py:7-9) — `update_stats()` loads and saves whichever
pickles sit in the current working directory. Run the sweep with `cwd=/tmp/ac_sandbox`
and only the sandbox pickles change.

- The TTSCardMaker clone (`~/Desktop/TTSCardMaker`) can be SHARED read-only — nothing in
  this procedure writes to it, and the remote-mode sweep doesn't read it at all.
- **`config.py` (secret tokens) copies along with `cp -r`.** Keep the sandbox strictly
  local, never commit/upload anything from it, and `rm -rf /tmp/ac_sandbox` when done.
- Footgun: `parse_one.py` / `inspect_pickle.py` / `gap_trace.py` derive the repo root
  from their own file location and `chdir` there. Inside the sandbox, run the SANDBOX's
  copies (`/tmp/ac_sandbox/.claude/skills/.../scripts/...`) or you will silently read the
  LIVE repo's pickles.

Gate: sandbox exists; ALL experimentation (code edits, sweeps) happens in the sandbox
only. The live repo gets the change at step 5, through autocroissant-change-control.

**Step 1 — Golden parses, BEFORE and AFTER.**

```bash
cd /tmp/ac_sandbox
S=.claude/skills/autocroissant-diagnostics-and-tooling/scripts
mkdir -p golden_before golden_after
# BEFORE applying your change (or run from a pristine second copy):
python3 $S/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd > golden_before/mini_doomer.txt
python3 $S/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd" > golden_before/freezer.txt
python3 $S/parse_one.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" > golden_before/lich_king.txt
python3 $S/gap_trace.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" > golden_before/lich_king_trace.txt
# ... apply your change, then repeat into golden_after/ ...
diff -r golden_before golden_after
```

Gate: the diff is empty, OR every differing line is exactly the intended change (e.g. your
fix targets a golden — then the AFTER output must match the intended expected values, and
this skill's inventory table gets updated at promotion time). Any other byte of drift =
stop, you changed more than you think.

**Step 2 — Full sweep IN THE SANDBOX (the expensive gate).**

Exact signature, verified 2026-10-20 at repo_traverser.py:486 (`cards` is a comma-separated list of
names to re-parse anyway):

```python
def update_stats(interaction=None, output_problematic=True, force_update=False, verbose=True, cards="")
```

(`force_update` defaulted to True in Python until 2026-10-19; it now matches the slash command's
False, so a sweep must pass `force_update=True` explicitly, as below. `use_local_repo` and
`use_local_timestamp` were removed on 2026-10-20 when local mode was archived; passing them now raises
TypeError. Flag ownership: autocroissant-config-and-flags.)

Safe invocation (GitHub, the only mode), from the sandbox root:

```bash
cd /tmp/ac_sandbox
cp stats.pkl stats_before.pkl && cp old_stats.pkl old_stats_before.pkl   # pre-sweep snapshots
python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import update_stats; update_stats(interaction=None, force_update=True)"
```

- **This is the reference sweep.** The archived local traversal (`update_stats_local()`, kept on branch
  `archive/local-update-stats`, section 4.1 method B) walks the clone instead: no PSD downloads and,
  with its default `use_local_timestamp=True`, no per-card API calls. It became path-correct on
  2026-10-19 (archaeology Entry 1). Read its diff differently: every card's `timestamp` changes to a
  local mtime and every card is archived into old_stats, so compare fields other than `timestamp`, and
  ignore old_stats growth. Never promote its pickles to the live repo (docs/BUGS_AND_IMPROVEMENTS.md B6).
- **API budget warning:** change detection compares each PSD's blob sha in the tree with
  `card.blob_sha` (`_psd_changed`, repo_traverser.py:77); an unchanged PSD costs no requests. Each
  re-parsed PSD costs one `/git/blobs` download (`_download`, DOWNLOAD_WORKERS=4) plus one commit
  page (`_commit_page`, repo_traverser.py:112-115; per_page=100, LOOKUP_WORKERS=2 beside the downloads)
  that gives both the newest commit and, for a new or authorless card, the first author (a file with
  100+ commits costs one more request, `_get_remote_timestamp`, repo_traverser.py:409-444). So
  `force_update=True` costs ≈ 2 × 813 ≈ **1,630+ calls** (3 × 813 before 2026-10-20; about a third of the 5000/hr
  GIT_TOKEN limit; impossible at 60/hr without a token), plus the tree
  request and the small-file blobs; a default run costs only the changed PSDs. A forced run, or
  the first after a restart (the tree ETag is in memory), always gets the full tree. Minutes of
  wall time. Run it once per promotion, never in a loop.
- Headless quirk (harmless): at the end `update_stats` queues `card_repo.populate_files` /
  `prep_dataframes` onto the bot's command queue (repo_traverser.py:558-559); with no bot
  running nothing drains it and the process just exits.
- With `verbose=True` (default) the sweep prints progress, a summary
  (`N cards changed or new, N unchanged, N moved.`), and the full problem output (Channel A+B).

Gate: the sweep completes and saved (`stats.pkl` mtime changed). Proceed to steps 3–4;
the sweep's own printed problem list previews step 4.

**Step 3 — diff_stats: sandbox result vs pre-sweep snapshot.**

```bash
cd /tmp/ac_sandbox
python3 $S/diff_stats.py stats_before.pkl stats.pkl            # exit 0 = identical, 1 = changes, 2 = load error
python3 $S/diff_stats.py stats_before.pkl stats.pkl --verbose  # per-field old -> new for every modified card
```

Gate: **every added/removed/modified row is explained by the intended change** (the
`--verbose` field diffs and the aggregated field-change counts make mass effects obvious).
Expected explainable classes: cards your change intentionally re-parses differently;
cards genuinely edited upstream in TTSCardMaker since the pickle was last updated (separate them
with method A in section 4.1). Red flags = reject: unexplained `removed` rows (traversal missed
files), mass `path` changes (the old removeprefix bug signature), mass `type → unknown`
(classification broke), timestamp-only churn you can't account for. Also compare `old_stats` growth:
`python3 $S/inspect_pickle.py` summary in the sandbox. A sweep archives a card only when its PSD
blob changed or it moved (a forced re-parse of an unchanged PSD adds no history), so every
archived version above the pre-sweep count (run the same summary before step 2) must be one of
those (unexplained ballooning is the eb9aa84 duplicate-archiving family; `diff_stats.py` cannot
diff old_stats.pkl — it only handles stats-shaped pickles).

**Step 4 — Problem counts must not increase (both surfaces).**

```bash
cd /tmp/ac_sandbox
python3 $S/inspect_pickle.py --problems      # Channel A: baseline exactly 1 (20 Creature Types / MISSPELT TYPE: tornado)
# Channel A+B: run the full-validation one-liner from section 2 — baseline exactly 5
```

Gate: pickled count ≤ 1 and full-validation count ≤ 5 (2026-07-11 baselines), and any
DECREASE is attributed to your change on purpose. An increase = regression, full stop —
either your change broke extraction on some card, or it surfaced a real pre-existing
problem; in the second case the finding is valuable but the change still doesn't promote
until the new problem is understood and either fixed or explicitly accepted (with the
baseline in this file updated in the same breath).

### 4.1 The two regression methods used on 2026-10-19

Both ran for the icon-font change and the local-path fix. Method A was scripted on 2026-10-20 and kept:
`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/parse_all_ns.py <out.json> <sandbox root>` (every card PSD,
every field plus the validator's list, `timestamp`/`author` dropped) and `compare.py a.json b.json`
(field-by-field; expect `813 cards, 0 differences`). Build each sandbox with `git archive` (HEAD, or the
working tree copied) in /tmp; never check out HEAD in the live repo. Method B's scripts were not kept.

**Method A — old parser vs new parser on every PSD (no pickles, no API).**
1. Put HEAD's code next to the working one: a `git archive` of HEAD in a scratch dir (since the
   2026-10-20 split the parser spans psd_parser.py, card_models.py and the facade, so copy the whole
   tree, not one file).
2. In one script, build a `PSDParser` from each module with the same `all_types` (and, for the new
   one, the same `IconVocabulary`), then parse every `.psd` under `~/Desktop/TTSCardMaker` outside
   `MDW/` and `Markers/` with both. Each parser needs a `StatsDatabase`-like object because
   `parse` reads the existing card's author; give both the same loaded copy.
3. Compare `to_dict()` of the two results field by field, including `problems` and the exact
   `ability` bytes.
4. Gate: 0 differences, or every difference is the intended change.

Result on 2026-10-19: all 813 card PSDs parsed identically by the git-HEAD parser and the new one.
Layers that don't use the stoogedings font go through the old code path byte-for-byte. A
Photoshop-built test card exercising every token case also parsed as designed. Why this method
matters: it separates "my code changed output" from "the PSDs changed upstream", which a sweep
diff cannot do.

**Method B — sandboxed forced LOCAL sweep vs the stored remote data.**
1. Sandbox per Step 0 (copy of the repo and pickles in /tmp), checked out at 795fd23: the archived
   traversal calls `RepositoryTraverser._should_update_card`, which bacab06 removed, and `db.dirty_files`
   (renamed `seen_cards` 2026-10-20), so on current main it raises AttributeError.
2. In the sandbox (local mode was archived on 2026-10-20; on 2026-10-19 this step was
   `update_stats(interaction=None, use_local_repo=True, force_update=True)`):
   the archived traversal is not in main, so first copy it out of its branch (the sandbox is a `cp -r`,
   so its `.git` knows the branch):
   `mkdir -p archive && git show archive/local-update-stats:archive/local_update_stats.py > archive/local_update_stats.py`
   then `python3 -c "import sys; sys.path.insert(0,'.'); from archive.local_update_stats import update_stats_local; update_stats_local(force_update=True)"`
3. Diff the sandbox `stats.pkl` against the pre-sweep copy, field by field, ignoring `timestamp`
   (local mtimes replace commit dates on every card in a local-timestamp run).
4. Gate: 0 cards typed `unknown`, summary line "0 cards changed location", `type`/`types`/`path`/
   `stats` identical to the stored data; every ability or star difference explained by method A
   (if the old parser also produces it, it is an upstream PSD edit, not a regression).

Result on 2026-10-19: 813 cards, 0 UNKNOWN-typed, 0 path changes; types, paths and stats identical
to the stored remote-sweep data; 59 ability and 2 star differences, all reproduced by the HEAD
parser, so they are PSD edits newer than the bot's last sweep. This is the measurement that retired
"local mode corrupts every path" (autocroissant-failure-archaeology Entry 1).

**The `PARSER_VERSION` rule (2026-10-21).** `/update_stats` re-parses only PSDs whose blob changed, so a parser change
never reaches the stored cards by itself. When it must (a new field, a changed reading of the same bytes), bump
`PARSER_VERSION` in `commands/psd_parser.py` (:22; every parse stamps `CardInfo.parser_version`): the next `/update_stats`
sees every stored card stamped older (or unstamped, for older pickles) and re-parses them all once (~813 PSD downloads plus
commit pages, no history: the blobs are unchanged; it also skips the tree ETag while any card is stale), then goes
quiet again. Version 2 (the printed title) is the first bump. The gate for such a change is this section's method A with
the intended differences written down beforehand; on 2026-10-21 that was exactly 157 `display_name` additions and 2
`STATS TOO HIGH` (Sonic spd 20, Royal Eradicator Main Cannon atk 12, both allowed by `allow_high_stats` in the edits
file, so the live validation stays at the section 2 baselines).

**Step 5 — Promote via change-control.**

Only after gates 1–4 pass: apply the code change to the live repo through
autocroissant-change-control (its change classes and pickle-commit discipline own
committing/pushing). Whether to also promote the sandbox's re-swept pickles or re-run the
sweep on the live side is a change-control decision — either way the pickle-push gate
below applies before any push. Then `rm -rf /tmp/ac_sandbox`.

## 5. The pickle-push gate (this skill owns the POLICY; diagnostics owns the tool)

Before any `/push`, `/update`, or manual commit that includes pickle files, run the
HEAD-vs-working diff and account for every line; **a push with unexplained removed or
modified rows is forbidden** — a bad pickle snapshot has been pushed and reverted before
(eb9aa84, old_stats ballooned 3196→12041 bytes; story in autocroissant-failure-archaeology):

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 $SCRIPTS/diff_stats.py /tmp/stats_head.pkl stats.pkl   # every line must be explainable; exit 0 today
```

## 6. How to add a CardValidator check

**Where.** Computed checks go in `CardValidator.validate` (psd_parser.py:975) or, for
creature/minion stat rules, `CardValidator._validate_stats` (psd_parser.py:1014). This is
the DEFAULT home for new checks: return-value-only, visible in sweeps and `parse_one.py`,
zero database footprint, trivially removable. Parse-time checks
(`card.problems.append(...)` inside `PSDParser`, like MISSPELT TYPE at
psd_parser.py:549) PERSIST into stats.pkl — every firing is written into the committed
database and needs a sweep to add AND to scrub. Only make a check parse-time if it needs
information that exists solely during layer traversal, and expect a higher precision bar.

**Problem-string convention.** SCREAMING short phrases, stable and grep-able, optional
lowercase detail after a colon — existing vocabulary: `UNKNOWN TYPE`,
`ABILITY TEXT NOT FOUND`, `HP NOT FOUND`, `STATS TOO HIGH`, `NO ABILITY LAYER`,
`MISSPELT TYPE: tornado`. Match it.

**Exclusion-set etiquette.** `ABILITY_EXCLUSIONS` (21 names since 2026-10-21) exists to silence TRUE positives — cards that
legitimately have no ability text (`EXCESSIVE_STAT_EXCLUSIONS` went on 2026-10-21: a stat above 10 is allowed per card
by `allow_high_stats` in Card Edits.json, data rather than code). Adding a name requires a stated
reason (commit message at minimum: which card, why it's legitimate) and is **registered
debt** against the owner's perfect-extraction goal (zero per-card exclusion lists).
Never add an exclusion to shut up a FALSE positive — that's a parser or check bug; fix it.
Every new check that wants its own exclusion list should make you question the check.
Shrinking these lists is a headline win; growing them goes in the acceptance explanation.

**The lesson of the removed TYPE/WHITESPACE MISMATCH check.** fed8a83 (2026-01-24,
"output a problem if num types does not match num gaps in ability") appended
`TYPE / WHITESPACE MISMATCH (N types, M gaps)` to `card.problems` whenever the counts
differed. Six days later 081b1fd removed it while reworking injection: with line-start/
line-end type placement (0490195) and leftover-types-appended-to-last-line, a count
mismatch became NORMAL on correctly-parsed cards — the check fired on legitimate cards.
Three lessons: (a) **checks must be high-precision** — a check that fires on legitimate
cards trains the operator to skim past ALL problem output, which is worse than no check;
(b) it was a parse-time check, so its noise was being written into the pickles, not just
the console; (c) a re-scoped, precise version remains a live candidate — that belongs to
autocroissant-psd-extraction-campaign, and the incident detail to
autocroissant-failure-archaeology.

**Adding a check IS a parser change**: run the full acceptance procedure (section 4). The
check's firing list after a sandbox sweep is its precision measurement — every card it
flags should be a card a human agrees is wrong.

## 7. What "regression" means here, concretely

Any ONE of the following, with no accepted explanation, means the change is rejected:

- any golden-card byte-diff (`parse_one`/`gap_trace` before-vs-after; timestamp/author
  excluded per section 3);
- any NEW problem card on either surface (pickled Channel A count > 1, or full-validation
  count > 5, vs the 2026-07-11 baselines);
- any unexplained `diff_stats` row after a sweep (added, removed, or modified), or
  unexplained `old_stats` version growth beyond 218 names / 223 versions;
- any exclusion-list growth (ABILITY_EXCLUSIONS > 18; no stat list exists since 2026-10-21).

The corresponding improvements (goldens fixed toward intended values, problem counts
down, exclusion lists shorter) are how progress is measured — update the baselines in
this file whenever one legitimately moves, with date and commit.

## 8. When NOT to use this skill

- **Tool invocation details** (script flags, output-field meanings, all five diagnostics
  scripts incl. `dump_psd_layers.py`) → **autocroissant-diagnostics-and-tooling** (the
  scripts live in its `scripts/` dir; this skill only states which numbers gate what).
- **The extraction campaign's specific phases/gates and the solution menu for the
  type-injection problem** → **autocroissant-psd-extraction-campaign**.
- **Shipping**: committing, pushing, PICKLE-commit convention, what must never be
  committed → **autocroissant-change-control**; the /push//pull//update flow anatomy →
  **autocroissant-run-and-operate** §5.
- **Incident history** (removeprefix bug and its 2026-10-19 fix, eb9aa84 revert, whitespace saga
  details) → **autocroissant-failure-archaeology**.
- **TTSCardMaker-side checks** (font rebuild byte-identity, migrating PSDs to tokens, the website
  build) → **ttscardmaker-companion**.
- **Card-domain semantics** (classification table, layer rules, injection algorithm,
  schemas) → **impossibility-cards-reference**.
- **Debugging a live symptom** (bot misbehaving now) → **autocroissant-debugging-playbook**.

## 9. Provenance and maintenance

All facts verified 2026-07-11 directly against the working tree and by running the
read-only diagnostics; the rows marked (2026-10-19) or (2026-10-20) were re-verified on that day against HEAD a7a4bb3 plus
the uncommitted working tree. The section 4.1 numbers are as reported by the 2026-10-19 runs (their
sandbox was not kept). Line numbers drift — re-find with the greps below. If any
re-verification below disagrees with this file, the repo wins; update this file
(baselines, counts, line numbers, dates) in the same change.

| Volatile fact | Re-verify with (cwd = repo root) |
|---|---|
| `update_stats` signature & line (477; force_update=False; cards=""; no local options since 2026-10-20) (2026-10-20) | `grep -n -A4 "def update_stats" commands/repo_traverser.py` |
| Exclusion-list size (18; no stat list) | `python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import CardValidator as V; print(len(V.ABILITY_EXCLUSIONS), hasattr(V, 'EXCESSIVE_STAT_EXCLUSIONS'))"` (18 False) |
| Exclusion-list definition & line (950) (2026-10-21) | `grep -n "ABILITY_EXCLUSIONS: set" commands/psd_parser.py` |
| `PARSER_VERSION` (2) and its stamp, re-parse and ETag bypass (psd_parser.py:22, :352; repo_traverser.py:80, :166) (2026-10-21) | `grep -n "PARSER_VERSION" commands/psd_parser.py commands/repo_traverser.py commands/card_models.py` |
| Printed title read into `display_name` (psd_parser.py:409, `_printed_title` :422) (2026-10-21) | `grep -n "_printed_title\|^def title_text" commands/psd_parser.py` |
| Parse-time problem sites (404, 486; icon problems 593-625) (2026-10-20) | `grep -n "card.problems.append\|_add_problem(card" commands/psd_parser.py` |
| validate() mutation of card.problems (947-949) (2026-10-20) | `grep -n "NO ABILITY LAYER" commands/psd_parser.py` |
| validate() gated on output_problematic (384; the archive has its own) (2026-10-20) | `grep -n "if output_problematic" commands/repo_traverser.py` |
| Pickled problems key in to_dict (113) (2026-10-20) | `grep -n '"problem"' commands/card_models.py` |
| Local-path fix present (method B precondition) (2026-10-20: moved to branch `archive/local-update-stats`) | `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py` (:115) |
| Pickle paths cwd-relative | `grep -n "STATS_PKL" global_config.py` |
| Channel-A baseline (exactly 1) | `python3 $SCRIPTS/inspect_pickle.py --problems` |
| Full-validation baseline (exactly 5) | one-liner in section 2 |
| DB shape (813 cards; 218/223 old_stats; newest entry) | `python3 $SCRIPTS/inspect_pickle.py` |
| Golden expected values | the three `parse_one.py` + one `gap_trace.py` commands in section 3 |
| Freezer byte-equality vs stats.pkl | compare `parse_one` ability block to `python3 $SCRIPTS/inspect_pickle.py "The Freezer"` |
| HEAD-vs-working diff clean (0/0/0) | `git show HEAD:stats.pkl > /tmp/stats_head.pkl && python3 $SCRIPTS/diff_stats.py /tmp/stats_head.pkl stats.pkl` |
| Slash-command force_update default (main.py:383) (2026-10-20) | `grep -n "force_update" main.py` |
| MISMATCH check history | `git show -s --oneline fed8a83 081b1fd` and `git show fed8a83 -- commands/psd_analyzer.py` |
| eb9aa84 revert exists | `git show -s --oneline eb9aa84` |
| Remote-sweep API cost (blob + one commit page per re-parsed PSD, post-gate) | `grep -n "_get_remote_timestamp\|_commit_page\|COMMITS_PAGE =\|_blob\|_psd_changed" commands/repo_traverser.py` |
| Offline suite | `~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` (704 passed, 30 skipped, about 140 s, 2026-10-22, TTSCardMaker 07589cc7) |
| Method A scripts kept | `ls .claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/parse_all_ns.py .claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/compare.py` |
| PSD count (~904) | `find ~/Desktop/TTSCardMaker -name "*.psd" \| wc -l` |

Maintenance triggers: any accepted parser/validator change (update baselines + golden
table + date-stamps); a new golden nomination (section 3 procedure); the first
card migrated to stoogedings tokens (expect Channel-A icon problems to become possible; add a
token golden); exclusion-list membership changes (update counts here AND the acceptance
explanation); TTSCardMaker growth (PSD count and API-cost estimate drift).

2026-10-20 (later): method A's scripts moved from `~/Documents/ac-optimization/scripts/` into
`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` (byte-identical).

2026-10-21: suite count refreshed (310 passed; `tests/test_pickle_saves.py`, `tests/test_query_output.py` added); main.py and card_models.py line numbers re-pointed. Parser gate re-run: 813 cards, 0 differences.

2026-10-21 (later): PARSER_VERSION 2 reads the printed title into `display_name` (157 cards) and `EXCESSIVE_STAT_EXCLUSIONS` is gone (rows 1b, 1c, 4; §4 rule; §6, §7, §9); 8 goldens re-pinned (display_name; Royal Eradicator's validator now lists STATS TOO HIGH); suite 352 passed, 18 skipped; psd_parser.py and repo_traverser.py line numbers re-pointed.

2026-10-21 (review): the gate is 157 `display_name` + 2 `STATS TOO HIGH` (159): WX-78's title "wx-" beside an image layer named "78" now reads as the file name (new golden `Creatures/DST/3 Stars/WX-78.psd`; suite 356 passed, 18 skipped); psd_parser.py citations re-pointed; repo_traverser.py `_format_problems` :451 (called :404), `update_stats` :472, the end-of-run queue_commands :526-527. Gate re-run: the re-verify row for `ABILITY_EXCLUSIONS` now says :950.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-21 (card check): §2 names `commands/card_check.py` as the second place `validate()` runs (per pushed PSD, after the edits, into TTSCardMaker's card-check.yml report); validate :975-1011, `_validate_stats` :1014, the mutation :1005-1008, the sweep's gate :391-392 and `_format_problems` :466 (called :410) re-pointed; suite row 417 passed, 24 skipped (tests/test_card_check.py: 22, among them an API-failure exit-0 test and a readers-parity test against repo_traverser).

2026-10-22 (stoogedings batch): The Freezer and The Lich King re-pinned with `tests/make_goldens.py` (blobs 90fa61f4 -> 8505c4f0, 1732d9bf -> c69f7c24); every parsed field, alt_names and the validator are unchanged, only the blob moved. Their inline types are now tokens, so §3 lists the new `gap_trace` facts. `tests/test_psd_details.py` follows Royal Eradicator's new layers (icons `mixed`, the Blazing overlay now an `Ability FX: Blazing` layer). Suite 688 passed, 38 skipped against TTSCardMaker 5034c4cf.

2026-10-22 (queue A, L11, L14): 14 goldens re-pinned with `tests/make_goldens.py` after proving each against its old blob from the clone's history: 12 parse identically (Mini Doomer, Tainted Lazarus, Pix, Twin Emperors, Royal Eradicator, Wham Bam Bone, Sniffle Thwomp, Little Guy, P.A.N.D.A., The Radiance, Poison Circle, Poison); God Tamer stars 0 -> 5 (L14 Stars group, NO STARS FOUND gone); 20_Creature_Types `tornado` -> `wind` (types gain `wind`, MISSPELT TYPE gone). No clone card shows NO STARS FOUND or MISSPELT TYPE now (card_check over 812 PSDs: only Computer Virus), so tests/test_card_check.py reads God Tamer, Bubble and 20_Creature_Types from the clone's history (OLD); tests/test_rulebook_hidden.py compares the four H1 pages before and after 100511d3 (to_dict equal); the JSX-FX comparison reads Gasoline and Royal Eradicator from before the JSX ran on them. Fork-only: four psd_edit FX tests on Mini Doomer fail because an ability edit deletes the `Ability FX: (BOT)` badge layer (FX layers are rebuilt only for chart status words); open. Suite 704 passed, 30 skipped against TTSCardMaker 07589cc7.
