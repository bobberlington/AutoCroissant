---
name: autocroissant-psd-extraction-campaign
description: Load this skill to RUN the improvement campaign for AutoCroissant's hardest live problem - PSD ability/type/stat extraction. Trigger phrases and symptoms include "improve extraction", "perfect extraction", "parser campaign", "type injection wrong" / "[type] landed in the wrong place", "gaps vs types", "exclusion list" (ABILITY_EXCLUSIONS / EXCESSIVE_STAT_EXCLUSIONS / burn-down), "problem cards", "ABILITY TEXT NOT FOUND" / "STATS TOO HIGH" / "MISSPELT TYPE", "fix update_stats local mode" / removeprefix / relative-path bug (fixed 2026-10-19; Phase 0 done), "migrate cards to stoogedings tokens", and any plan to change parsing heuristics in commands/psd_analyzer.py. It contains the owner's measurable goal (100 percent correct extraction, zero per-card exclusion lists), a decision-gated numbered campaign (Phase 0 removeprefix foundation -> Phase 1 baseline scoreboard -> Phase 2 gap_trace instrumentation -> Phase 3 ranked solution menu with predicted numbers -> Phase 4 validation and promotion), exact commands with verified expected outputs (2026-07-11 golden numbers), live suspect lists, fenced-off known wrong paths with their incident commits, and scoreboard re-measurement one-liners. Read-only phases are safe against the live repo; anything that writes pickles runs only in a sandbox copy.
---

# The PSD Extraction Campaign

This is the executable campaign for the project's hardest live problem (owner-stated 2026-07-11):
extracting ability text, type icons, and stats from the ~904 card PSDs in `~/Desktop/TTSCardMaker`.

**The goal ("perfect extraction")**: 100% correct ability/type/stat extraction, with **zero
per-card exclusion lists**. Today the parser is good but propped up by 18 hardcoded card names
(`ABILITY_EXCLUSIONS`; the 5 `EXCESSIVE_STAT_EXCLUSIONS` went on 2026-10-21, replaced by the website's per-card
`allow_high_stats`). The traversal bug that made local-mode
`/update_stats` dangerous was fixed on 2026-10-19 (Phase 0). The campaign removes the remaining
props without breaking the 813 cards that parse correctly today.

**New lever since 2026-10-19: the stoogedings icon font.** A card whose inline icons are typed as
`[fire]` tokens in the font needs no gap guessing at all; the parser reads the tokens directly
(impossibility-cards-reference §4.1). No card uses it yet (0 of 813); about 45 PSDs (81 smart-object
icon layers) are the migration backlog. Migrating a card is card-content work in TTSCardMaker
(prototype and caveats: ttscardmaker-companion §5), but it is the most direct route to perfect
inline types for those cards, and Options 2-3 below then matter only for unmigrated cards.

**Success is MEASURED, never judged by eye.** Every phase ends at a gate with an expected number.
If your run shows a different number, take the branch — do not rationalize and continue.

Rules of engagement:

- All Phase 0–2 commands are read-only and safe to run from the repo root
  (`/Users/michaelsrouji/Desktop/AutoCroissant`). They never write a pickle.
- Anything that calls `update_stats()` or saves pickles runs ONLY in a sandbox copy
  (procedure owned by **autocroissant-validation-and-qa**). Never against the live repo.
- Code changes ship through **autocroissant-change-control**. This skill decides WHAT to try and
  what numbers prove it; those skills own HOW to land it.
- Since 2026-10-20 module imports print no token-presence line (only main.py does); the
  `2>&1 | grep -v "Git token"` filters below are harmless leftovers.

Glossary (used throughout):

| Term | Meaning |
|---|---|
| gap | A run of 3+ spaces in ability text (`_gap_pattern`, psd_parser.py:324 as of 2026-10-20) where a type icon visually sits |
| icon | Any visible layer with pixels whose lowercased name is in `all_types` (psd_parser.py:536; `all_types` is populated from TTSCardMaker `Types/` — pixel or smartobject kind both count) |
| token | Since 2026-10-19: a `[name]` typed in the stoogedings font, read directly by the parser — no gap needed (impossibility-cards-reference §4.1) |
| midline | `card_mid_y = height * TYPE_REGION_RATIO(0.5)` (psd_parser.py:23,375). Icons above it = creature-type tags (`card.types`); below it = candidates for inline injection into ability text |
| injection | Replacing gaps with ` [typename] ` line by line (`_inject_type_names`, psd_parser.py:901) |
| leftover-append | When kept icons outnumber gaps, the extras are appended to the last line (psd_parser.py:939-943) |
| sweep | One `update_stats()` traversal (GitHub trees API; the local dir walk is archived) that reparses changed cards |
| sandbox | A scratch copy of this repo + pickles where sweeps may write; see autocroissant-validation-and-qa |
| golden card | A card whose full expected parse is pinned in this skill and byte-checked after every change |

## The scoreboard

Append a dated row whenever you re-measure (commands per metric in "Provenance and maintenance").
This table IS the campaign's progress record.

| Date | stats.pkl cards | Stored problem cards | Re-validation problems | ABILITY_EXCLUSIONS | STAT_EXCLUSIONS | Empty-string abilities | Residual-gap cards | Goldens byte-equal |
|---|---|---|---|---|---|---|---|---|
| 2026-07-11 | 813 | 1 | 5 | 18 | 5 | 20 | 10 | 3/3 |
| (target) | tracks TTSCardMaker | true content errors only | == stored problems | 0 | 0 | tracked, explained | legit layouts only | N/N, N growing |

Metric definitions (all verified 2026-07-11):

- **Stored problem cards**: entries in stats.pkl whose `problem` field is non-empty. Today exactly 1:
  `20 Creature Types` (`MISSPELT TYPE: tornado`) — a content flag in the source PSD, not a parser bug.
- **Re-validation problems**: cards that fail `CardValidator.validate` when run over today's stored
  data. Today 5 — the stored count undercounts because the pickle stores only parse-time problems
  (`NO ABILITY LAYER`, `MISSPELT TYPE`); validator-computed problems (`ABILITY TEXT NOT FOUND`,
  `HP/DEF/ATK/SPD NOT FOUND`, `STATS TOO HIGH`) are reported to Discord during a sweep but never
  stored. The 5: `20 Creature Types`, `Computer Virus` (all four stats NOT FOUND),
  `Anubisath Guardian`, `Qiraji Soldier`, `Silithid` (ABILITY TEXT NOT FOUND — blank minions not in
  the exclusion list).
- **Empty-string abilities**: cards with `ability == ""` (an Ability text layer exists but is blank).
  Today 20 = 17 of the 18 ABILITY_EXCLUSIONS + the 3 WoW minions above. The 18th exclusion,
  `Shadow Duelist`, now HAS ability text — a stale exclusion.
- **Residual-gap cards**: cards whose STORED ability still contains a 3+ space run (a gap that
  received no icon). Today 10: 4 rulebook table pages (legit layout) + `Michael's Blessing`,
  `Miracle Matter`, `Pile Of Swords`, `Twin Emperors`, `Warcraftian Druid`, `Warcraftian Hunter`.
  These 6 are the standing Phase 2 suspect list.
- **Goldens**: Mini Doomer, The Freezer, The Lich King (inventory owned by
  autocroissant-validation-and-qa; expected values pinned below).

---

## Phase 0 — Restore a trustworthy foundation: DONE (2026-10-19)

**What it was.** `_process_local_files` computed
`relative_path = full_path.removeprefix("TTSCardMaker").strip('/')`, a no-op on the absolute paths
`walk()` yields, so a local-mode `/update_stats` classified every card from a top folder of `Users`
(UNKNOWN type, wrong paths, mass archiving). Introduced by cythonization commit f7c915c
(2025-12-02). Full story: **autocroissant-failure-archaeology** Entry 1.

**What landed** (2026-10-19, committed in 53db8ee):
exactly the Gate 0b candidate this phase specified, plus the `relpath` import:

```python
relative_path = relpath(full_path, local_path).replace('\\', '/')   # psd_analyzer.py:1226 on 2026-10-19
```

(Since 2026-10-20 this line lives in `archive/local_update_stats.py` on branch `archive/local-update-stats`,
not in main: local mode was archived out of the bot, see below.)

**Measured results (2026-10-19)** — procedure in autocroissant-validation-and-qa §4.1:
- Sandboxed forced LOCAL sweep: 813 cards, 0 typed UNKNOWN, "0 cards changed location", and
  types, paths and stats identical to the stored remote-sweep data.
- 59 ability and 2 star differences, all reproduced by the git-HEAD parser, so they are upstream PSD
  edits newer than the bot's last sweep (docs/BUGS_AND_IMPROVEMENTS.md D5), not effects of the fix.
- Old parser vs new parser on all 813 PSDs: 0 differences.

**Prediction vs measurement.** Option 1 below predicted Tier A would change only `timestamp`. The 61
extra rows broke that prediction, and the explanation (stale stats.pkl, proven by the old parser
reproducing them) was found before promotion — the methodology working as intended. Lesson: Tier A's
precondition "stats.pkl matches the clone" must be checked first, or an old-vs-new parser run
(method A) used to separate code effects from upstream edits.

**What is still true about local mode.** The fix makes local mode path-correct, not
timestamp-compatible: with `use_local_timestamp=True`, every card's mtime is newer than its stored
commit date (813/813 on 2026-10-19), so a local run with local timestamps re-archives everything
(option 1's own Tier A prediction said so; docs/BUGS_AND_IMPROVEMENTS.md B6). On 2026-10-20 `/update_stats` first
defaulted to remote mode, then local mode was ARCHIVED: the options are gone from the command, and the
traversal survives as `update_stats_local()` on branch `archive/local-update-stats`, for sandbox sweeps only
(operating rule owned by autocroissant-run-and-operate §7; story: autocroissant-failure-archaeology
Entry 10.12).

### Gate 0a — Regression check (keep running it after parser refactors)

```bash
python3 - <<'EOF'
from os import walk
from os.path import expanduser, join as path_join, relpath
local_path = expanduser('~/Desktop/TTSCardMaker')
for folder, _, files in walk(local_path):
    for file in files:
        if file.endswith('.psd'):
            full_path = path_join(folder, file)
            print('full_path :', full_path)
            print('relpath   :', relpath(full_path, local_path))
            raise SystemExit
EOF
git grep -n "relative_path = " archive/local-update-stats -- archive/local_update_stats.py   # :115
```

**EXPECTED (2026-10-19; file path updated 2026-10-20):** `relpath` prints a clean repo-relative path
(first walked PSD on this machine: `MDW/cardback_item.psd`), and the grep shows the
`relpath(full_path, local_path)` line in `_process_local_files`. If the grep shows a string operation (`removeprefix`, `split`, slicing)
instead, the bug has been reintroduced — stop and read the fenced-off fix below.

### FENCED-OFF wrong fix — do not do this (kept as a lesson)

`relative_path = full_path.removeprefix(local_path).strip('/')` (string-prefix stripping with the
absolute clone path) LOOKS equivalent and is not. It breaks when `local_path` carries a trailing
slash, when the walk yields a symlink-resolved or differently-normalized prefix, and it silently
degrades to the same no-op failure mode when the prefix mismatches by one character — the exact
shape of the original bug. `os.path.relpath(full_path, local_path)` normalizes both sides and
fails loudly (`ValueError`/`..`-prefixed result) instead of silently mangling. The whole incident
happened because a string operation was assumed to be a path operation; do not re-assume it.

---

## Phase 1 — Baseline measurement (all read-only)

Run all of these from the repo root before changing anything, and again after every landed change.
Record results as a new scoreboard row.

### Gate 1a — Pickle summary

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py 2>&1 | grep -v "Git token\|Trying to open\|Loaded existing"
```

**EXPECTED (2026-07-11):** 813 cards; old_stats 218 names / 223 archived versions; distribution
creature 297, item 182, minion 91, debuff 52, field 50, aux item 48, rulebook 48, nme 26, buff 12,
creatures 7; `cards with recorded parse problems: 1`; 0 orphans; newest `Anubisath Guardian @
2026-06-18`; `entries with suspicious (non repo-relative) paths: 0`.

- Higher card count / newer newest-timestamp → TTSCardMaker gained cards and a sweep ran; fine,
  update the scoreboard.
- Suspicious paths > 0 → a local-mode sweep ran with the bug (since 2026-10-20 only the archived
  traversal on branch `archive/local-update-stats` walks the clone). STOP the campaign; this is now a
  data-recovery incident: autocroissant-debugging-playbook ("stats look wrong") +
  autocroissant-change-control (pickle discipline). Do not proceed on corrupted baseline.
- Problem count > 1 → run Gate 1b and classify each new entry before continuing.

### Gate 1b — Stored problems

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --problems 2>&1 | grep -v "Git token\|Trying to open\|Loaded existing"
```

**EXPECTED (2026-07-11):** exactly one line:
`20 Creature Types [Auxiliary/Rulebook/20_Creature_Types.psd]: ['MISSPELT TYPE: tornado']`

### Gate 1c — Exclusion-list sizes (the numbers this campaign exists to drive to zero)

```bash
awk '/ABILITY_EXCLUSIONS: set/{f=1;next} f&&/}/{exit} f&&/",/{n++} END{print "ABILITY_EXCLUSIONS:", n}' commands/psd_parser.py
```

**EXPECTED (2026-10-21):** 18 (the stat list no longer exists). (Cross-check that imports agree with source:
`python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import CardValidator as V; print(len(V.ABILITY_EXCLUSIONS))"`
→ `18 5`. If source and import disagree, a stale compiled `.so` is shadowing the `.py` — see
autocroissant-build-and-env, run `python3 setup.py clean`.)

### Gate 1d — Golden parses, byte-checked against stats.pkl

```bash
python3 - <<'EOF' 2>&1 | grep -v "Git token\|Trying to open\|Loaded existing"
import sys; sys.path.insert(0, '.')
from os.path import expanduser
from commands.psd_analyzer import stats_db, PSDParser, RepositoryTraverser, StatsDatabase
stats_db.load()
clone = expanduser('~/Desktop/TTSCardMaker')
fresh = StatsDatabase()
RepositoryTraverser(fresh)._populate_types_from_local(clone)
parser = PSDParser(fresh.all_types, fresh)
goldens = [
    ('Mini Doomer',   'Auxiliary/Minions/Mini_Doomer.psd'),
    ('The Freezer',   'Field/1 Stars/The_Freezer.psd'),
    ('The Lich King', 'Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd'),
]
for name, rel in goldens:
    card = parser.parse(clone + '/' + rel, rel)
    db = stats_db.stats[name]
    print(f'{name:15s} ability_byte_equal={card.ability == db.ability} '
          f'types={card.types == db.types} card_type={card.card_type == db.card_type} '
          f'stats={card.stats.to_dict() == db.stats.to_dict()}')
EOF
```

**EXPECTED (2026-07-11):** three lines, every flag `True`. Pinned golden values (snapshot of
2026-07-11 from `parse_one.py`, re-runnable — see Provenance; canonical current expected values
live in autocroissant-validation-and-qa §3 — if these stop matching, check there before assuming
a regression):

- **Mini Doomer** → type=minion, stars=2, types=['wind','defensive_minion','light'],
  hp=5 def=2 atk=5 spd=5, ability ends `(BOT) Die.`, problems NONE.
- **The Freezer** → type=field, stars=1, types=['field'], problems NONE, ability
  character-for-character:
  `For friendly creatures:\n[water] Turns into [ice]. [ice]: +1 DEF\n(1PGT) Choose a friendly creature to turn into a [ice] and lower its speed by 1.`
  (Note: the PSD reads `(1GCD)` since 2026-10-21 (cooldown update, impossibility-simulator-rules §3); stats.pkl keeps `(1PGT)` until the next `/update_stats`, after which this golden's line reads `(1GCD) Choose …`.)
- **The Lich King** → ability's last line `Minions inherit [undead].` (full trace in Phase 2).

Any `False` → the parser in your working tree no longer reproduces the database. Either you changed
parsing code (intended? then this is your regression signal) or a stale `.so` shadows the source
(autocroissant-build-and-env). Diagnose with Phase 2 on the failing card before proceeding.

### Gate 1e — Deep censuses (the metrics the summary hides)

```bash
python3 - <<'EOF' 2>&1 | grep -v "Git token\|Trying to open\|Loaded existing"
import sys, re; sys.path.insert(0, '.')
from commands.psd_analyzer import stats_db, CardValidator
stats_db.load()
val = [(n, CardValidator.validate(c)) for n, c in sorted(stats_db.stats.items())]
val = [(n, p) for n, p in val if p]
print(f're-validation problems: {len(val)}')
for n, p in val: print('  ', n, p)
gap = re.compile(r'\s{3,}')
res = sorted(n for n, c in stats_db.stats.items() if c.ability and gap.search(c.ability))
print(f'residual-gap cards: {len(res)}'); print('  ', res)
empty = sorted(n for n, c in stats_db.stats.items() if c.ability == '')
print(f'empty-string abilities: {len(empty)}'); print('  ', empty)
EOF
```

**EXPECTED (2026-07-11):** `re-validation problems: 5` (the 5 named in the scoreboard section);
`residual-gap cards: 10` (4 `Auxiliary/Rulebook` pages + the 6 suspects); `empty-string abilities:
20` (17 exclusion-list names + Anubisath Guardian, Qiraji Soldier, Silithid).

- New names in re-validation → new problem cards arrived from TTSCardMaker; add them to the
  Phase 3 option-4 target table and classify with the burn-down recipe.
- Residual-gap count grew → a card with unfilled gaps was committed upstream OR an injection
  regression landed; gap_trace it (Phase 2).

### Gate 1f — No-op diff check (proves your working pickles match HEAD)

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
```

**EXPECTED (2026-07-11):** `added: 0   removed: 0   modified: 0`, exit code 0. Anything else means
uncommitted pickle changes exist — resolve per autocroissant-change-control before campaigning.

---

## Phase 2 — Instrumented understanding

For ANY suspect card, run the injection tracer before forming a theory:

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py "<path/to/card.psd>" 2>&1 | grep -v "Git token"
```

It re-drives the real `PSDParser` methods and prints 6 numbered steps. What each step can reveal:

| Step | Shows | A wrong parse looks like |
|---|---|---|
| [1] creature-types above midline | which icons became `card.types` tags | an inline icon stolen into tags (icon y just above `card_mid_y`) → its gap goes unfilled |
| [2] raw ability text layers + bbox | text layer presence and y-position | 0 layers → NO ABILITY LAYER; text present but blank (`'\r'`) → empty ability; an extra text layer joining in wrong order |
| [3] joined text with `<GAP:n>` markers | every 3+ space run the injector will target | a gap that is really table formatting; a missing gap where an icon sits (author used ≤2 spaces) |
| [4] below-midline icons, sorted + prune decision | injection candidates, row grouping (40px), `max(last_y//3, card_mid_y)` threshold | a legitimate icon in `dropped:` (prune too aggressive); row grouping merging two visual rows |
| [5] gap count vs kept types | the ordinal balance | MISMATCH → leftovers appended to last line, or gaps left unfilled |
| [6] final text after injection + punctuation cleanup | what would be stored | `[type]` in the wrong slot; punctuation cleanup (`\s+([:;,\.\?!])`→`\1`, psd_parser.py:841 on 2026-10-20) having eaten a legitimate gap before punctuation |

### Worked example — The Lich King (verified 2026-07-11)

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" 2>&1 | grep -v "Git token"
```

**EXPECTED:** PSD 1200x2400, card_mid_y=1200. [1] tags `['ice', 'undead']` (note: `undead` appears
BOTH as a tag above the midline and as an inline icon below — this is why the midline exists).
[2] one text layer at (x=23, y=1773). [3] one gap: `Minions inherit<GAP:7>.` [4] one icon:
`undead` at (x=734, y=2239); prune threshold `max(2239//3=746, 1200)=1200`, kept `['undead']`,
dropped `[]`. [5] `gap count = 1 vs kept types = 1 (match)`. [6] final last line
`Minions inherit [undead].` — the punctuation cleanup collapsed the space before `.`.

### Counterexample — Warcraftian Hunter (why gap-counting alone cannot be a validator)

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/4 Stars/Warcraftian_Hunter.psd" 2>&1 | grep -v "Git token"
```

**EXPECTED (2026-07-11):** [3] shows FOUR gaps — `<GAP:23>Turtle<GAP:17>Cat` /
`<GAP:23>Hawk<GAP:16>Wolf` — which are a two-column visual table of pet names, not icon slots.
[4] shows ZERO below-midline icons. [5] `gap count = 4 vs kept types = 0` — mismatch — and the
final text is CORRECT precisely because nothing was injected. This card is in the residual-gap
census by design. Any validator or "fix" that treats every gap as an icon slot breaks this card;
any whitespace normalization destroys its columns (see Known wrong paths).

Classify every suspect into exactly one bucket before proposing a fix: missing/blank text layer |
icon misrouted by midline | icon wrongly pruned | ordinal gap/type mismatch | leftover-append
surprise | punctuation-cleanup artifact | legitimate layout (no fix; certify). If a card fits no
bucket, the pipeline model in **impossibility-cards-reference** is incomplete — update understanding
before code.

---

## Phase 3 — The solution menu, ranked

Discipline (owned by **autocroissant-research-methodology**): every option states its theory
obligations and its predicted, measurable effect BEFORE you run anything. A change whose numbers
you cannot predict is not ready to be coded. Unproven ideas below are labeled CANDIDATE.

### Option 1 — The relative-path fix (DONE 2026-10-19) + a local-mode regression harness (open)

**Status:** the fix landed on 2026-10-19 (Phase 0, committed in 53db8ee). Local mode left the bot on
2026-10-20; the sweep below now runs through `update_stats_local()`, copied into the sandbox from branch
`archive/local-update-stats` (`mkdir -p archive && git show archive/local-update-stats:archive/local_update_stats.py > archive/local_update_stats.py`).
What remains open is the HARNESS: scripting the two checks from autocroissant-validation-and-qa §4.1 (old-vs-new parser over
every PSD; sandboxed local sweep diffed field-by-field) so every future parser change can run them
in minutes. The predictions below are kept because they are the harness's expected outputs.

**Theory obligation (met):** nothing else in local mode is path-sensitive. `_populate_types_from_local`
walks `Types/` directly; the `EXCLUDE_FOLDERS` skip splits the absolute folder into parts; file access
uses the absolute path and only classification/storage uses the relative one. **Why it was ranked
first:** it unblocks fast local iteration for everything below — remote sweeps then cost 1+ API call per
card and minutes of wall time (since 2026-10-20 only changed PSDs cost requests); local-timestamp sweeps make no API calls.

**Predicted numbers, stated before running** (all in a SANDBOX; procedure owned by
autocroissant-validation-and-qa):

- Tier A (fast, local timestamps): sandbox force sweep with the fix
  (`update_stats_local(force_update=True)` from the archive branch, copied in as above, headless; on 2026-10-19
  this was `update_stats(None, True, True, True, True)`) then
  `diff_stats.py <pre.pkl> <post.pkl>` shows `added: 0 removed: 0`, and `field change counts`
  contains ONLY `{'timestamp': 813}` — local mtimes replace remote commit dates; every other field
  byte-identical. old_stats grows by one archived copy per card whose mtime is newer than the
  stored timestamp (≤813; on a fresh clone, all). Summary line reads `813 had newer timestamps or
  were new ... 0 cards changed location.`
- Tier B (full equivalence, `update_stats_local(use_local_timestamp=False, force_update=True)` — costs ~813+ GitHub API calls, needs
  GIT_TOKEN's 5000/hr): `diff_stats.py` reports `0/0/0`, exit 0, and old_stats is unchanged —
  local traversal becomes bit-identical to remote traversal on an unchanged clone. Precondition:
  clone at the same commit as upstream main.
- Both tiers: the sweep's problem report lists exactly today's 5 re-validation cards.
- Measured 2026-10-19 (Tier A, forced): `path`/`type`/`types`/`stats` unchanged on all 813, 0 location
  changes; plus 59 ability and 2 star rows from upstream PSD edits (Phase 0). Until D5's refresh lands,
  expect those 61 rows on top of the timestamp-only prediction. Tier B has not been run.
- With the pre-2026-10-19 code, Tier A showed mass `path` changes and `type` → unknown; to see that
  failure shape, run HEAD~ code in a sandbox, never against live pickles.

**If Tier A shows any non-timestamp field changes** → the parser is nondeterministic or
environment-sensitive for those cards; gap_trace each before shipping the fix. **If Tier B is not
0/0/0** → path or timestamp semantics still differ between modes; the fix is incomplete; do not
promote.

### Option 2 — Restore gap/type mismatch detection as a HIGH-PRECISION validator problem (CANDIDATE)

**History (verify the diffs, not the messages):** fed8a83 (2026-01-24, "output a problem if num
types does not match num gaps in ability") added
`TYPE / WHITESPACE MISMATCH (N types, M gaps)` whenever whole-text gap count != total icon count
(both > 0). 081b1fd (2026-01-30) rewrote injection per-line with leftover-append and REMOVED the
check. It thrashed because the equality is false for legitimate layouts:

- Pure-formatting gaps with zero icons — Warcraftian Hunter (4 gaps, 0 icons) is in the corpus
  today and correct.
- Icons that sit at the start/end of lines without a 3+ space gap in the extracted text — the
  leftover-append path exists precisely for them, and it makes types > gaps normal.

**Derived precision conditions (obligations before coding):**

- C1 (types side): fire only when leftover-append fires for ≥2 icons AFTER per-line filling —
  a single trailing icon is a common legitimate layout; multiple homeless icons usually mean gaps
  were missed or icons misrouted. The threshold must be tuned by a corpus census, not intuition.
- C2 (gaps side): fire only when kept_types > 0 AND at least one gap on an icon-bearing card
  remains unfilled after injection — and measure this BEFORE the punctuation cleanup at
  psd_parser.py:841 (2026-10-20), which eats a gap that sits directly before punctuation (Lich King's gap
  would vanish from the final text).
- C3 (guard): kept_types == 0 → never fire (exempts Warcraftian Hunter and the 4 rulebook tables).

**Predicted numbers, stated before running:** an instrumented sandbox census over all 904 PSDs
(per card: gaps, kept icons, leftover count, unfilled-gap count pre-cleanup) shows C1(≥2)+C2+C3
firing on 0 cards, or only on cards independently confirmed defective by gap_trace. True-positive
proof: seed a synthetic fault (in the sandbox parser, drop one icon from The Freezer's
`type_bboxes`) → C2 fires on The Freezer. If the census shows the conditions firing on >~5
legitimate cards, the validator is not precise; refine or retire it in
autocroissant-failure-archaeology like its predecessor.

### Option 3 — Bbox-anchored injection (CANDIDATE; hardest, highest ceiling)

**Theory:** stop assigning icons to gaps by global ordinal order; assign each icon to a text LINE
using its (x, y), then fill that line's gaps left-to-right. Localizes any mistake to one line.

**Derivation obligations — stated honestly:** psd-tools exposes the LAYER bbox only
(`layer.bbox` = x1,y1,x2,y2 for the whole Ability text layer), NOT per-line text bboxes. Line
y-positions must be estimated: `line_height = (y2 - y1) / num_lines`,
`line_index = clamp(floor((icon_center_y - y1) / line_height), 0, num_lines - 1)`.

Worked numbers (The Lich King, verified 2026-07-11 via dump_psd_layers): Ability layer bbox
(23, 1773, 1176, 2285) → height 512px over 5 lines → 102.4 px/line. Undead icon bbox
(734, 2239, 808, 2299) → center_y=2269 → (2269-1773)/102.4 = 4.84 → line index 4 = last line
`Minions inherit .` — correct.

**Labeled hard sub-problems (do not hand-wave these):** (a) `num_lines` from `'\n'` in extracted
text vs RENDERED lines differs when text soft-wraps — the estimate breaks on wrapped paragraphs;
(b) mixed font sizes/leading break constant line_height; (c) mapping icon x to a character column
inside a line needs font metrics psd-tools does not provide — per-line ordinal filling (line's
icons sorted by x fill the line's gaps in order) is the tractable compromise; (d) last-line icons
can center below the layer bbox (clamp needed — Lich King's icon bottom, 2299, already exceeds
layer y2=2285).

**Predicted numbers, stated before running:** a sandbox sweep with bbox-anchored injection
produces stored abilities byte-identical to today for every card EXCEPT ones independently
diagnosed as mis-injected (today that set is: none confirmed; the 6 residual-gap suspects are
unfilled-gap cases, not mis-placement cases). Goldens stay 3/3. Any golden change = defect in the
new logic, by definition. If the byte-identical set is smaller than ~807/813, the line-estimation
model is too crude — stop and revisit.

### Option 4 — Exclusion-list burn-down (start anytime; pairs with any option)

**The repeatable per-card recipe:**

```bash
# 1. find the PSD
find ~/Desktop/TTSCardMaker -iname "*<card_name>*" -name "*.psd"
# 2. what does the parser extract?
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py "<psd>" 2>&1 | grep -v "Git token"
# 3. what is physically in the file?
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py "<psd>" --text | head -60
# 4. classify: genuinely-blank card vs parser miss  ->  5. fix (Phase 3/4) or re-certify (record here)
```

**Worked example — Crystal Bits (run 2026-07-11):**
`find` → `~/Desktop/TTSCardMaker/Auxiliary/Items/Crystal_Bits.psd`. `parse_one` → type=aux item,
stars=1, types=['active'], hp=3, def=0, ability is an EMPTY string, `validator problems: NONE`
(suppressed only by its ABILITY_EXCLUSIONS entry). `dump_psd_layers --text` → the `Ability` text
layer exists with bbox (0, 0, 0, 0) and raw text `"'\r'"` — a blank placeholder. **Classification:
genuinely no ability text on the card.** The exclusion is truthful but structural: the parser
cannot currently say "blank on purpose" without a hardcoded name.

**Live target table (verified 2026-07-11):**

| Target | Finding | Action |
|---|---|---|
| Shadow Duelist | In ABILITY_EXCLUSIONS but now HAS ability text (`Heals the creature this is summoned on by 2 HP...`) | Stale entry. Remove it; predicted effect: exclusions 18→17, re-validation problems unchanged at 5 |
| Crystal Bits, Panda + 15 more empty-ability exclusions | Blank `Ability` layer (`'\r'`, bbox (0,0,0,0)); Panda additionally has full stats 6/2/7/3 | Genuinely blank. Re-certify individually, or take the structural candidate below |
| Anubisath Guardian, Qiraji Soldier, Silithid | Same blank pattern (verified on Silithid), NOT in the list → they fire `ABILITY TEXT NOT FOUND` in every force-sweep report | The exclusion treadmill live: adding 3 names is the anti-goal. Take the structural candidate |
| Computer Virus | Creature with ability but hp/def/atk/spd all -1 → fires 4 NOT FOUND problems per force sweep | Unclassified — run the recipe (step 3 on its `Darks`/`Bars` groups) before theorizing |
| The 5 EXCESSIVE_STAT_EXCLUSIONS (DONE 2026-10-21) | Cards whose summed stat digits exceed 10 | The list is gone: a stat above 10 is STATS TOO HIGH unless the card's Card Edits.json entry sets `allow_high_stats` (Sonic spd 20, Royal Eradicator Main Cannon atk 12: data, not code); the canonical encoding above 10 is a top row 10 plus a bottom row N-10 |

**Structural CANDIDATE (the single highest-leverage exclusion killer):** the validator conflates
`ability == ""` with `ability is None` (`not card.ability`, psd_parser.py:993 as of 2026-10-20). The parser
already distinguishes them: a present-but-blank `Ability` layer yields `""`, a MISSING layer yields
`None` + parse-time `NO ABILITY LAYER`. Candidate rule: blank-but-present = legitimate no-ability
card; only `None` (or absence of the layer) is a problem. **Predicted effect:**
ABILITY_EXCLUSIONS 18 → 0 (17 blank + 1 stale), the 3 WoW minions stop firing, re-validation
problems 5 → 2 (`20 Creature Types`, `Computer Virus`), empty-string abilities stays 20 and becomes
the watched metric. **Stated risk:** a card whose ability text is accidentally deleted (layer kept,
text emptied) goes silent. Mitigation is measurement, not a name list: the scoreboard tracks the
empty-ability count, and diff_stats flags any card whose ability transitions non-empty → empty at
sweep time — a reviewer gate in Phase 4. Route through change-control like everything else.

---

## Phase 4 — Validation and promotion

The full acceptance procedure (sandbox construction, golden nomination, pickle-push gate) is OWNED
by **autocroissant-validation-and-qa**; the change classes and commit discipline by
**autocroissant-change-control**. This section pins only the campaign-specific expected numbers for
those gates, as of 2026-07-11:

1. **Golden gate:** Gate 1d script → 3/3 all-True. Any change to a golden's bytes fails the change,
   full stop. Nominate newly certified cards (e.g., a burned-down exclusion card) as new goldens so
   N grows.
2. **Sandbox sweep gate (remote mode, force_update=True, unchanged upstream):** diff_stats pre vs
   post → `0/0/0`; old_stats unchanged (since 2026-10-20 history is appended only when a PSD's blob
   changed or the card moved); problem REPORT = exactly the current re-validation set (5 today; fewer only if your
   change predicted the specific disappearances).
3. **Sandbox sweep gate (local traversal from branch `archive/local-update-stats`, copied into the sandbox; the option 1 fix is in since 2026-10-19):** Tier A/B numbers as
   predicted in option 1, after subtracting rows that the unchanged (HEAD) parser also produces
   (upstream PSD edits; 61 on 2026-10-19). Any other surprise field in `field change counts` = not
   ready.
4. **Problem-count non-increase:** re-validation census (Gate 1e) after the sandbox sweep must not
   contain any NEW name. New names = regression, regardless of how good the diff looks.
5. Then and only then: change-control for the code; PICKLE commit convention + diff-before-push for
   any data snapshot. Never /push pickles produced inside a sandbox.

---

## Known wrong paths — fenced off, with their incidents

These were all tried. Re-proposing one without new evidence wastes a campaign cycle.

| Wrong path | Incident | Why it is wrong |
|---|---|---|
| Collapse multiple spaces in ability text (`\s{2,}` → `' '`) | The 2026-02-08 whitespace saga: 8 commits in one day (7 parser commits + a mid-saga PICKLE), `1c26747` → … → `3bbaa2b` "Fix the issue of collapsing spaces"/`4bcee6b` — collapsing finally REMOVED (chronology: autocroissant-failure-archaeology Entry 2) | Gaps ARE the injection signal, and wide spacing is real layout — Warcraftian Hunter's pet-name columns (Phase 2) would be mashed into one line. Whitespace in ability text is load-bearing |
| Treat above-midline icons as inline types (or lower/remove the midline) | `d0bb28c` (2026-01-30) replaced a top-400px rule with `TYPE_REGION_RATIO=0.5` because creature-type tags live in the top region | The Lich King carries `undead` twice: a tag at y=21 and an inline icon at y=2239. Erase the midline and every tag becomes a bogus injection candidate |
| "Clean up" `x[len(x)-1]` into `x[-1]` in commands/psd_analyzer.py | `dd800bc` (2026-01-30) changed `bboxes[-1]` → `bboxes[len(bboxes)-1]`; the negative-indexing ban dates from cythonization `f7c915c` | This file compiles with Cython `wraparound=False`; negative indices are undefined in compiled form. The idiom is deliberate (autocroissant-change-control non-negotiables) |
| Trust commit messages over diffs | `dd800bc` is titled just "Update psd_analyzer.py" yet carries the wraparound fix; `081b1fd` is titled "fix bug..." yet also silently deleted the mismatch validator | Always `git show <hash> -- commands/psd_analyzer.py` and read the hunks. History messages in this repo are sometimes jokes or vague |
| Hand-edit stats.pkl to "fix" one card | The cca0aaf → eb9aa84 bad-pickle commit/revert (2025-11-10) is what mistrusted pickle surgery earns | Use `/update_metadata` (main.py:436 → `manual_metadata_entry`, psd_analyzer.py:169; 2026-10-20 lines). And know its verified semantics, below |
| Rename or delete a Types/ icon PNG to "fix" a type name | Not an incident yet; the 2026-10-19 icon-font work made it a cross-repo hazard | Image-layer type names, font tokens, the font manifest (which keeps old entries), emoji names and the website all key off the file stem. Add an alias instead (ttscardmaker-companion §1.7) |

**What the code shows about manual entries vs the next sweep** (re-read 2026-10-20 after the blob-sha
change — state no more than this): `manual_metadata_entry` creates a missing card as
`CardInfo(name, UNKNOWN, path=f"{path}.psd")` (psd_analyzer.py:221) with `timestamp` 0.0 and no
`blob_sha`; the command cannot set either. `_psd_changed` (repo_traverser.py:77) re-parses a card that
is new, moved, or whose blob sha differs; a card with no sha is re-parsed only if its PSD was committed
since the newest stored timestamp, otherwise it adopts the current sha and is kept. Manual EDITS to an
existing entry survive until the PSD's blob changes, `cards:` names the card, or `force_update:True`; a
re-parse of an unchanged blob does not archive the manual version to old_stats. Conclusion the code
supports: `/update_metadata` is a between-sweeps correction tool, not a durable fix. Durable fixes live
in the parser, the source PSD, or `site/Card Edits.json` (re-applied after every re-parse).

---

## When NOT to use this skill

- Understanding the parser/domain without changing it (folder classification, layer semantics,
  query DSL, card counts) → **impossibility-cards-reference**.
- How to run/interpret the five diagnostic scripts themselves → **autocroissant-diagnostics-and-tooling**
  (this skill only consumes them).
- A live symptom that is not extraction (bot won't start, commands missing, messages not sending,
  music) → **autocroissant-debugging-playbook**.
- The removeprefix bug's full story, evidence, and fix → **autocroissant-failure-archaeology** Entry 1.
- Migrating cards to stoogedings tokens (the font, the JSX prototype, Photoshop pitfalls) →
  **ttscardmaker-companion**; how the parser reads tokens → **impossibility-cards-reference** §4.1.
- Operating a sweep on the live bot (`/update_stats` etiquette, multi-machine handoff) →
  **autocroissant-run-and-operate**.
- Sandbox construction and the general acceptance procedure → **autocroissant-validation-and-qa**;
  commit/push/pickle discipline → **autocroissant-change-control**.
- The evidence bar and idea lifecycle in the abstract → **autocroissant-research-methodology**;
  the two long-horizon ambitions → **autocroissant-research-frontier**.

## Provenance and maintenance

Everything dated 2026-07-11 was measured that day on the owner's Mac against stats.pkl at commit
`284d13c` and the local TTSCardMaker clone (upstream latest 2026-06-18). Phase 0, Option 1's status,
the glossary line numbers and the items marked 2026-10-19 were updated that day (HEAD a7a4bb3 plus
the working tree, since committed in 53db8ee; the TTSCardMaker clone is now shallow at 3a268ca). Line numbers drift; each
volatile fact below has a one-line re-verification. After any landed parser change, re-run Phase 1
and append a scoreboard row; if a Phase 3 option ships or retires, move its entry (retirements are
documented in autocroissant-failure-archaeology).

- Phase 0 fix present (psd_analyzer.py:1226 until 2026-10-20; now archive/local_update_stats.py:115 on branch
  archive/local-update-stats) and the old bug gone:
  `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py; grep -c removeprefix commands/psd_analyzer.py` (0)
- Bug introduction in f7c915c (history):
  `git show f7c915c -- commands/psd_analyzer.py | grep -B3 -A3 removeprefix`
- Local mode archived 2026-10-20 (no `use_local_*` options; slash force_update=False at main.py:383):
  `grep -rn "use_local" main.py commands/` (only `RETIRED_OPTIONS` in analytics.py); `grep -n "force_update: Optional" main.py`
- Python-side `update_stats` force_update=False default since 2026-10-19 (repo_traverser.py:473 on 2026-10-20):
  `grep -n "force_update: bool = " commands/repo_traverser.py`
- `_populate_types_from_local` (repo_traverser.py:202 on 2026-10-20, skips dotfiles since 2026-10-19; kept for the diagnostics scripts):
  `grep -n -A9 "def _populate_types_from_local" commands/repo_traverser.py`
- Pipeline line numbers (psd_parser.py, 2026-10-20) — gap regex :318, TYPE_REGION_RATIO :22, card_mid_y :375, prune :831,
  inject :838, leftover-append :876-880, punctuation cleanup :778, `not card.ability` :937:
  `grep -n "_gap_pattern\|TYPE_REGION_RATIO\|card_mid_y = int\|_prune_type_bboxes\|_inject_type_names\|not card.ability" commands/psd_parser.py`
  and for the cleanup line `grep -n "ability_text = sub" commands/psd_parser.py`
- Exclusion sizes 18 + 5: the two awk commands in Gate 1c.
- Scoreboard 813 cards / 1 stored problem / distribution / newest card: Gate 1a command.
- Scoreboard 5 re-validation / 10 residual-gap / 20 empty: Gate 1e command.
- Goldens 3/3 byte-equal: Gate 1d command. Per-card golden output:
  `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py <psd>` on each
  of the three golden paths.
- Lich King and Warcraftian Hunter traces: the two gap_trace commands in Phase 2.
- Crystal Bits / Silithid blank-layer evidence:
  `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py <psd> --text | grep -B1 "TEXT:"`
- Lich King bboxes used in option 3 — Ability layer (23, 1773, 1176, 2285), inline undead icon
  (734, 2239, 808, 2299):
  `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" | grep -i "ability\|undead"`
- Mismatch-validator history:
  `git show fed8a83 -- commands/psd_analyzer.py | grep -B4 -A10 MISMATCH` (added) and
  `git show 081b1fd -- commands/psd_analyzer.py | grep -B2 -A8 MISMATCH` (removed).
- Incident commits (3bbaa2b, d0bb28c, dd800bc, f7c915c, cca0aaf/eb9aa84):
  `git log --oneline -1 <hash>` then read the diff, never just the message.
- Manual-entry semantics (2026-10-20: CardInfo timestamp default card_models.py:87, `_psd_changed`
  repo_traverser.py:77, `manual_metadata_entry` psd_analyzer.py:169, its CardInfo creation
  psd_analyzer.py:221; /update_metadata at main.py:436):
  `grep -n "timestamp: float = \|def _psd_changed\|def manual_metadata_entry" commands/card_models.py commands/repo_traverser.py commands/psd_analyzer.py; grep -n "update_metadata" main.py`
- `/update_stats` operating rule (GitHub only since 2026-10-20; the archived local traversal is for
  sandbox sweeps): owned by autocroissant-run-and-operate §7.
- Font adoption count (0 of 813 PSDs on 2026-10-19) and the migration backlog (45 PSDs / 81
  smart-object icon layers): ttscardmaker-companion §5.

2026-10-21: main.py and card_models.py line numbers re-pointed (atomic pickle saves, 0.1 s queue tick); no parser change, gate 813 cards / 0 differences.

2026-10-21 (titles): `EXCESSIVE_STAT_EXCLUSIONS` deleted (goal paragraph, the count commands, the suspect table); the parser now also reads the printed title into `display_name` (PARSER_VERSION 2; the gate showed 157 + 2 intended differences).

2026-10-21 (review): goal paragraph says 18 hardcoded names; psd_analyzer.py citations +2 (`manual_metadata_entry` :169, its CardInfo :221); psd_parser.py citations re-pointed after the WX-78 title fix (a title that is part of the file name beside an image layer printing the rest reads as the file name); the gate shows 157 + 2.

2026-10-21 (S6/S7 gate): main.py, repo_traverser.py and help.py line numbers re-pointed after the /bake_cards block, the bake_lock in update_stats and the /bake_cards help entry; nothing else in this skill changed.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.
