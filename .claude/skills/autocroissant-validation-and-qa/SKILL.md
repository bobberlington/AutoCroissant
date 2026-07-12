---
name: autocroissant-validation-and-qa
description: Load this skill BEFORE judging whether any AutoCroissant change is safe or finished — whenever the task sounds like "is this change safe", "test this", "validate", "verify", "acceptance", "QA", "golden", "regression", "how do I know it works", "did the parser break anything", or before pushing/committing pickles (/push, /update, PICKLE commits) or touching commands/psd_analyzer.py (parser, CardValidator, exclusion lists). This repo has NO tests and NO CI; this skill defines what counts as evidence instead — the evidence hierarchy (golden-card byte-equality > full-sweep diff_stats deltas > validator problem-card counts > exclusion-list sizes > eyeballing Discord output), the certified golden-card inventory (Mini Doomer, The Freezer, The Lich King) with expected values and the nomination procedure, the sandboxed parser-change acceptance procedure with the exact update_stats invocation and its GitHub-API cost, the pickle-push diff gate, how to add a CardValidator check without repeating the TYPE/WHITESPACE MISMATCH mistake, and the concrete definition of "regression". Baselines certified 2026-07-11.
---

# AutoCroissant validation & QA: what counts as evidence

Certified 2026-07-11. This repo has **no test suite, no CI, no README**. The pickles
(`stats.pkl`, `old_stats.pkl`) ARE the production database and are committed to git. The
owner's two costliest failure classes are (1) pickle data corruption and (2) Discord API
quirks. So QA here is not "run the tests" — it is a small set of **measured numbers with
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
| 2 | **diff_stats deltas on a full sweep** | sandbox sweep, then `diff_stats.py before.pkl after.pkl` | HEAD vs working today: `813/813, added 0 removed 0 modified 0, exit 0` | Whole-database impact. Every added/removed/modified line must have an explanation; "removed" rows you didn't intend and mass path/type changes are corruption signatures |
| 3 | **Validator problem-card count** (must not increase) | pickled surface: `inspect_pickle.py --problems`; full surface: re-validation one-liner (section 2) | pickled: **exactly 1** ("20 Creature Types" — `MISSPELT TYPE: tornado`); full validation: **exactly 5** (the 1 above + 4 computed-only, listed in section 2) | Extraction quality did not regress anywhere the validator can see |
| 4 | **Exclusion-list sizes** | count `CardValidator.ABILITY_EXCLUSIONS` and `EXCESSIVE_STAT_EXCLUSIONS` | **18** ABILITY_EXCLUSIONS (psd_analyzer.py:649) + **5** EXCESSIVE_STAT_EXCLUSIONS (psd_analyzer.py:642) | Shrinking them is progress toward "perfect extraction" (owner doctrine); growing them is new registered debt |
| 5 | **Eyeballing Discord output** | run a command, look at the message | none | Weakest. Catches only gross breakage; never sufficient alone. Acceptable as a final smoke check AFTER 1–4 pass, never as a substitute |

If evidence levels conflict, the stronger level wins. A change that "looks great" in
Discord but produces one unexplained `diff_stats` row is NOT accepted.

## 2. The two problem surfaces (read this before counting problems)

Verified against `commands/psd_analyzer.py` on 2026-07-11. Problems travel through two
distinct channels, and they have different baselines:

**Channel A — parse-time problems, PERSISTED in the pickle.**
`CardInfo.problems` is a dataclass field (psd_analyzer.py:135). During parsing exactly two
problem kinds are appended to it: `NO ABILITY LAYER` (psd_analyzer.py:422, when no ability
text layer exists) and `MISSPELT TYPE: <name>` (psd_analyzer.py:456, for visible pixel
layers named in `MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']`,
psd_analyzer.py:42). `StatsDatabase.save()` pickles the CardInfo objects themselves
(psd_analyzer.py:210-215), so these problems persist into `stats.pkl` and surface as the
`"problem"` key in `to_dict()` (psd_analyzer.py:165-166). `inspect_pickle.py --problems`
counts exactly this channel.

**Channel B — computed validation problems, OUTPUT-ONLY, never pickled.**
`CardValidator.validate(card)` (psd_analyzer.py:670-705) builds a fresh local list from
computed checks — `UNKNOWN TYPE`, `ABILITY TEXT NOT FOUND`, `HP/DEF/ATK/SPD NOT FOUND`,
`STATS TOO HIGH` (via `_validate_stats`, psd_analyzer.py:707-733) — then extends it with
the card's Channel-A problems and returns it. The return value is **never stored on the
card**: during a sweep it flows only into `_format_problems` (psd_analyzer.py:1098-1106)
and out to Discord/console (called at psd_analyzer.py:965-968 remote, 1041-1044 local).
So a sweep's printed problem list = Channel A + Channel B; the pickle stores only Channel A.

**One side effect to know:** `validate()` MUTATES `card.problems` — it removes
`NO ABILITY LAYER` when the card is in `ABILITY_EXCLUSIONS` (psd_analyzer.py:699-702).
And `validate()` only runs during a sweep when `output_problematic=True`
(psd_analyzer.py:965/1041). Consequence: a sweep run with `output_problematic=False`
would leave `NO ABILITY LAYER` in the pickled problems of the 18 excluded cards,
inflating the Channel-A count by up to 18 with zero real regression. **Always sweep with
`output_problematic=True` (the default) when measuring.**

Certified baselines (both measured 2026-07-11):

- **Channel A (pickled): exactly 1 problem card.**
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

## 3. The golden inventory (certified 2026-07-11)

Goldens are the strongest evidence because each was chosen from a battle scar: The Freezer
and Mini Doomer WERE the bug cards of the 2026-02-08 whitespace saga (8 commits in one
day — 7 parser commits plus a mid-saga PICKLE — 1c26747 through 3bbaa2b "Fix the issue of
collapsing spaces"/4bcee6b; chronology in autocroissant-failure-archaeology Entry 2); The
Lich King pins the single-gap + prune-threshold path. All PSD paths are relative to the
TTSCardMaker clone at `~/Desktop/TTSCardMaker`.

**This table is the CANONICAL home of the golden expected values.** Other skills quote
them only as dated snapshots that defer here — when a golden legitimately changes, update
THIS table (and the date-stamps) first; the snapshots point back to this section.

| Golden | PSD path | What it certifies | Expected values (certified 2026-07-11) |
|---|---|---|---|
| **Mini Doomer** | `Auxiliary/Minions/Mini_Doomer.psd` | Minion classification; stars counted from visible pixel layers (two visible "Warpstar" smartobjects under a "Stars" group); 3 creature types; summed stat digits | `type=minion`, `stars=2`, `types=['wind', 'defensive_minion', 'light']`, `hp=5 def=2 atk=5 spd=5`, ability ends `(BOT) Die.`, validator problems `NONE` |
| **The Freezer** | `Field/1 Stars/The_Freezer.psd` | Field classification; multi-type inline injection including line-START and line-END type positions; parse output equals the stats.pkl entry byte-for-byte | `type=field`, `stars=1`, `types=['field']`, problems `NONE`, ability EXACTLY:<br>`For friendly creatures:`<br>`[water] Turns into [ice]. [ice]: +1 DEF`<br>`(1PGT) Choose a friendly creature to turn into a [ice] and lower its speed by 1.`<br>(re-verified equal to the stats.pkl entry character-for-character today) |
| **The Lich King** | `Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd` | Single gap + single below-midline type (exact match, no leftovers); prune-threshold arithmetic | `gap_trace.py`: above-midline creature types `['ice', 'undead']`; 1 text layer (y=1773); gap `<GAP:7>` in `Minions inherit<GAP:7>.`; 1 below-midline icon `undead @ (x=734, y=2239)`; prune threshold `max(last_y//3=746, card_mid_y=1200) = 1200`; kept `['undead']`, dropped none; final line `Minions inherit [undead].` |

Commands to reproduce (each takes a few seconds; read-only):

```bash
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"
python3 $SCRIPTS/parse_one.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"
python3 $SCRIPTS/gap_trace.py ~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"
```

**Equality rules.** A fresh `parse_one.py` run has `timestamp: 0.0` and no `author` —
those two fields come from traversal metadata (git commit dates / DB preservation,
psd_analyzer.py:379-380, 929), not from the PSD. So:

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

Use this for ANY change to `commands/psd_analyzer.py` parsing/classification/validation
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
are cwd-relative (global_config.py:6-7) — `update_stats()` loads and saves whichever
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

Exact signature, verified 2026-07-11 at psd_analyzer.py:1173:

```python
def update_stats(interaction=None, output_problematic=True, use_local_repo=True,
                 use_local_timestamp=True, force_update=True, verbose=True)
```

(The Python defaults are NOT what the Discord slash command sends — main.py:363 passes
`force_update=False` by default. Flag ownership: autocroissant-config-and-flags.)

Safe invocation — remote mode, from the sandbox root:

```bash
cd /tmp/ac_sandbox
cp stats.pkl stats_before.pkl && cp old_stats.pkl old_stats_before.pkl   # pre-sweep snapshots
python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import update_stats; update_stats(interaction=None, use_local_repo=False, force_update=True)"
```

- **Remote mode (`use_local_repo=False`) is mandatory as of 2026-07-11**, even in a
  sandbox: local mode's relative-path computation is broken (the `removeprefix` live bug,
  psd_analyzer.py:1002, introduced by f7c915c) and classifies every card as UNKNOWN under
  a "Users" top folder — its output is garbage, not a baseline. Full story:
  autocroissant-failure-archaeology. (Exception: if the change under test IS the local-path
  fix, local-mode behavior is the thing you're measuring — the campaign skill owns that.)
- **API budget warning:** remote mode makes 2 fixed calls plus one `repo.get_commits(path=...)`
  call per NON-EXCLUDED PSD — timestamps are fetched *before* the force_update check
  (psd_analyzer.py:929, 1066), but AFTER the `EXCLUDE_FOLDERS` skip (psd_analyzer.py:922), so a
  sweep costs ≈ 2 + 813 ≈ **815 API calls regardless of flags** (~16% of the 5000/hr GIT_TOKEN
  limit; 60/hr without a token — impossible). Derivation owned by autocroissant-analysis-toolkit
  Recipe 4. `force_update=True` additionally downloads every PSD via `urlretrieve`. Minutes of
  wall time. Run it once per promotion, never in a loop.
- Headless quirk (harmless): at the end `update_stats` queues `card_repo.populate_files` /
  `prep_dataframes` onto the bot's command queue (psd_analyzer.py:1241-1242); with no bot
  running nothing drains it and the process just exits.
- With `verbose=True` (default) the sweep prints progress, a summary
  (`N had newer timestamps or were new / N did not / N changed location`), and the full
  problem output (Channel A+B).

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
cards genuinely edited upstream in TTSCardMaker since the pickle was last updated (today:
none — newest entry Anubisath Guardian @ 2026-06-18 matches TTSCardMaker's latest commit).
Red flags = reject: unexplained `removed` rows (traversal missed files), mass `path`
changes (the removeprefix bug signature), mass `type → unknown` (classification broke),
timestamp-only churn you can't account for. Also compare `old_stats` growth:
`python3 $S/inspect_pickle.py` summary in the sandbox — archived-version count above the
baseline (218 names / 223 versions) must be explained (unexplained ballooning is the
eb9aa84 duplicate-archiving family; `diff_stats.py` cannot diff old_stats.pkl — it only
handles stats-shaped pickles).

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

**Where.** Computed checks go in `CardValidator.validate` (psd_analyzer.py:670) or, for
creature/minion stat rules, `CardValidator._validate_stats` (psd_analyzer.py:707). This is
the DEFAULT home for new checks: return-value-only, visible in sweeps and `parse_one.py`,
zero database footprint, trivially removable. Parse-time checks
(`card.problems.append(...)` inside `PSDParser`, like MISSPELT TYPE at
psd_analyzer.py:456) PERSIST into stats.pkl — every firing is written into the committed
database and needs a sweep to add AND to scrub. Only make a check parse-time if it needs
information that exists solely during layer traversal, and expect a higher precision bar.

**Problem-string convention.** SCREAMING short phrases, stable and grep-able, optional
lowercase detail after a colon — existing vocabulary: `UNKNOWN TYPE`,
`ABILITY TEXT NOT FOUND`, `HP NOT FOUND`, `STATS TOO HIGH`, `NO ABILITY LAYER`,
`MISSPELT TYPE: tornado`. Match it.

**Exclusion-set etiquette.** `ABILITY_EXCLUSIONS` (18 names) and
`EXCESSIVE_STAT_EXCLUSIONS` (5 names) exist to silence TRUE positives — cards that
legitimately have no ability text or stats above 10. Adding a name requires a stated
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
- any exclusion-list growth (ABILITY_EXCLUSIONS > 18 or EXCESSIVE_STAT_EXCLUSIONS > 5).

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
- **Incident history** (removeprefix live bug, eb9aa84 revert, whitespace saga details) →
  **autocroissant-failure-archaeology**.
- **Card-domain semantics** (classification table, layer rules, injection algorithm,
  schemas) → **impossibility-cards-reference**.
- **Debugging a live symptom** (bot misbehaving now) → **autocroissant-debugging-playbook**.

## 9. Provenance and maintenance

All facts verified 2026-07-11 directly against the working tree and by running the
read-only diagnostics. Line numbers drift — re-find with the greps below. If any
re-verification below disagrees with this file, the repo wins; update this file
(baselines, counts, line numbers, dates) in the same change.

| Volatile fact | Re-verify with (cwd = repo root) |
|---|---|
| `update_stats` signature & line (1173) | `grep -n "def update_stats" commands/psd_analyzer.py` |
| Exclusion-list sizes (18 / 5) | `python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import CardValidator as V; print(len(V.ABILITY_EXCLUSIONS), len(V.EXCESSIVE_STAT_EXCLUSIONS))"` |
| Exclusion-list definitions & lines (642, 649) | `grep -n "EXCESSIVE_STAT_EXCLUSIONS\|ABILITY_EXCLUSIONS: set" commands/psd_analyzer.py` |
| Parse-time problem sites (422, 456) | `grep -n "card.problems.append" commands/psd_analyzer.py` |
| validate() mutation of card.problems (699-702) | `grep -n "NO ABILITY LAYER" commands/psd_analyzer.py` |
| validate() gated on output_problematic (965, 1041) | `grep -n "if output_problematic" commands/psd_analyzer.py` |
| Pickled problems key in to_dict (165-166) | `grep -n '"problem"' commands/psd_analyzer.py` |
| Pickle paths cwd-relative | `grep -n "STATS_PKL" global_config.py` |
| Channel-A baseline (exactly 1) | `python3 $SCRIPTS/inspect_pickle.py --problems` |
| Full-validation baseline (exactly 5) | one-liner in section 2 |
| DB shape (813 cards; 218/223 old_stats; newest entry) | `python3 $SCRIPTS/inspect_pickle.py` |
| Golden expected values | the three `parse_one.py` + one `gap_trace.py` commands in section 3 |
| Freezer byte-equality vs stats.pkl | compare `parse_one` ability block to `python3 $SCRIPTS/inspect_pickle.py "The Freezer"` |
| HEAD-vs-working diff clean (0/0/0) | `git show HEAD:stats.pkl > /tmp/stats_head.pkl && python3 $SCRIPTS/diff_stats.py /tmp/stats_head.pkl stats.pkl` |
| Slash-command force_update default (main.py:363) | `grep -n "force_update" main.py` |
| MISMATCH check history | `git show -s --oneline fed8a83 081b1fd` and `git show fed8a83 -- commands/psd_analyzer.py` |
| eb9aa84 revert exists | `git show -s --oneline eb9aa84` |
| Remote-sweep API cost (get_commits per PSD, pre-gate) | `grep -n "_get_remote_timestamp\|get_commits" commands/psd_analyzer.py` |
| PSD count (~904) | `find ~/Desktop/TTSCardMaker -name "*.psd" \| wc -l` |

Maintenance triggers: any accepted parser/validator change (update baselines + golden
table + date-stamps); a new golden nomination (section 3 procedure); the removeprefix bug
getting fixed (retire the remote-mode-only rule in step 2 and re-certify a local-mode
sweep); exclusion-list membership changes (update counts here AND the acceptance
explanation); TTSCardMaker growth (PSD count and API-cost estimate drift).
