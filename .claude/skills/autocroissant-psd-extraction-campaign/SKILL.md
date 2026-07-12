---
name: autocroissant-psd-extraction-campaign
description: Load this skill to RUN the improvement campaign for AutoCroissant's hardest live problem - PSD ability/type/stat extraction. Trigger phrases and symptoms include "improve extraction", "perfect extraction", "parser campaign", "type injection wrong" / "[type] landed in the wrong place", "gaps vs types", "exclusion list" (ABILITY_EXCLUSIONS / EXCESSIVE_STAT_EXCLUSIONS / burn-down), "problem cards", "ABILITY TEXT NOT FOUND" / "STATS TOO HIGH" / "MISSPELT TYPE", "fix update_stats local mode" / removeprefix / relative-path bug, and any plan to change parsing heuristics in commands/psd_analyzer.py. It contains the owner's measurable goal (100 percent correct extraction, zero per-card exclusion lists), a decision-gated numbered campaign (Phase 0 removeprefix foundation -> Phase 1 baseline scoreboard -> Phase 2 gap_trace instrumentation -> Phase 3 ranked solution menu with predicted numbers -> Phase 4 validation and promotion), exact commands with verified expected outputs (2026-07-11 golden numbers), live suspect lists, fenced-off known wrong paths with their incident commits, and scoreboard re-measurement one-liners. Read-only phases are safe against the live repo; anything that writes pickles runs only in a sandbox copy.
---

# The PSD Extraction Campaign

This is the executable campaign for the project's hardest live problem (owner-stated 2026-07-11):
extracting ability text, type icons, and stats from the ~904 card PSDs in `~/Desktop/TTSCardMaker`.

**The goal ("perfect extraction")**: 100% correct ability/type/stat extraction, with **zero
per-card exclusion lists**. Today the parser is good but propped up by 23 hardcoded card names
(18 `ABILITY_EXCLUSIONS` + 5 `EXCESSIVE_STAT_EXCLUSIONS`) and one live traversal bug that makes
local-mode `/update_stats` dangerous. The campaign removes the props without breaking the 813 cards
that parse correctly today.

**Success is MEASURED, never judged by eye.** Every phase ends at a gate with an expected number.
If your run shows a different number, take the branch — do not rationalize and continue.

Rules of engagement:

- All Phase 0–2 commands are read-only and safe to run from the repo root
  (`/Users/michaelsrouji/Desktop/AutoCroissant`). They never write a pickle.
- Anything that calls `update_stats()` or saves pickles runs ONLY in a sandbox copy
  (procedure owned by **autocroissant-validation-and-qa**). Never against the live repo.
- Code changes ship through **autocroissant-change-control**. This skill decides WHAT to try and
  what numbers prove it; those skills own HOW to land it.
- Module imports print token-presence noise. Append `2>&1 | grep -v "Git token"` to any command
  below if you want clean output; expected outputs shown here already have it filtered.

Glossary (used throughout):

| Term | Meaning |
|---|---|
| gap | A run of 3+ spaces in ability text (`_gap_pattern`, psd_analyzer.py:348) where a type icon visually sits |
| icon | Any visible layer with pixels whose lowercased name is in `all_types` (psd_analyzer.py:442; `all_types` is populated from TTSCardMaker `Types/` — pixel or smartobject kind both count) |
| midline | `card_mid_y = height * TYPE_REGION_RATIO(0.5)` (psd_analyzer.py:30,405). Icons above it = creature-type tags (`card.types`); below it = candidates for inline injection into ability text |
| injection | Replacing gaps with ` [typename] ` line by line (`_inject_type_names`, psd_analyzer.py:593) |
| leftover-append | When kept icons outnumber gaps, the extras are appended to the last line (psd_analyzer.py:632-635) |
| sweep | One `update_stats()` traversal (local dir walk or GitHub trees API) that reparses changed cards |
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

## Phase 0 — Restore a trustworthy foundation

You cannot iterate on extraction quality while the local traversal mangles every path. Fix or
route around this first.

**The bug in one paragraph** (full story with evidence is OWNED by
**autocroissant-failure-archaeology** — read it before touching this code):
`_process_local_files` computes `relative_path = full_path.removeprefix("TTSCardMaker").strip('/')`
(commands/psd_analyzer.py:1002). `walk()` yields ABSOLUTE paths under
`expanduser("~/Desktop/TTSCardMaker")`, so `removeprefix("TTSCardMaker")` is a no-op and
`relative_path` keeps the absolute prefix. `CardClassifier.classify` then sees top folder `Users`
→ every card gets UNKNOWN type, wrong stored paths, and mass path-change archiving. One local-mode
`/update_stats` run would corrupt stats.pkl at scale — and `use_local_repo` defaults to True on the
slash command (main.py:361). Introduced by cythonization commit f7c915c (2025-12-02), which replaced
the previously correct `full_path.split("TTSCardMaker")[-1]` in a mechanical negative-index purge
(Cython `wraparound=False`). Current data is fine because real sweeps have run in remote mode.
**Operating rule until fixed (owned by autocroissant-run-and-operate): `/update_stats` must be run
with `use_local_repo:False`.**

### Gate 0a — Reproduce it, read-only

```bash
python3 - <<'EOF'
from os import walk
from os.path import expanduser, join as path_join, relpath
local_path = expanduser('~/Desktop/TTSCardMaker')
for folder, _, files in walk(local_path):
    folder = folder.replace('\\', '/')
    for file in files:
        if file.endswith('.psd'):
            full_path = path_join(folder, file)
            print('full_path        :', full_path)
            print('line-1002 result :', full_path.removeprefix('TTSCardMaker').strip('/'))
            print('CANDIDATE relpath:', relpath(full_path, local_path))
            raise SystemExit
EOF
```

**EXPECTED (2026-07-11):** `line-1002 result` starts with `Users/...` (the absolute path minus its
leading slash — NOT repo-relative), while `CANDIDATE relpath` prints a clean repo-relative path
(first walked PSD on this machine: `MDW/cardback_item.psd`).

- If you see this → bug still live. Continue to Gate 0b; keep the remote-mode operating rule.
- **If instead** `line-1002 result` is already repo-relative → the fix has landed. Confirm with
  `grep -n "relative_path = " commands/psd_analyzer.py`, update this Phase to "historical", skip to
  Phase 1, and re-run the full baseline (Phase 1) before trusting anything else here.
- **If instead** the script prints nothing → the TTSCardMaker clone is missing or empty; nothing
  else in this campaign will work. Fix the clone first (`LOCAL_DIR_LOC` in global_config.py).

### Gate 0b — The candidate fix (NOT applied; label stays CANDIDATE until Phase 4 passes)

The principled fix is:

```python
# commands/psd_analyzer.py:1002  (CANDIDATE - do not apply outside a change-control branch)
relative_path = relpath(full_path, local_path).replace('\\', '/')
```

Note the import: psd_analyzer.py:9 (`from os.path import getmtime, basename, expanduser, splitext,
join as path_join`) does NOT currently include `relpath`; the candidate fix must add it. Evidence
the construction is right: the diagnostics scripts compute the relative path exactly this way
(`relpath(psd_path, clone_root)`), and every `parse_one.py` run below classifies correctly with it.

Blast-radius check — is anything else in local mode path-sensitive?

1. `_populate_types_from_local` (psd_analyzer.py:883-890): **unaffected**. Verify by reading it:
   it walks `path_join(local_path, "Types")` directly and filters with
   `if folder.endswith("Types")` on the walked ABSOLUTE folder — the absolute path
   `.../TTSCardMaker/Types` ends with `Types`, so the check passes, and `Types/Stars` is correctly
   skipped (ends with `Stars`). Empirical proof: every diagnostic run below prints
   `known types (36)` populated via this exact function in local mode.
   `sed -n '883,890p' commands/psd_analyzer.py` to confirm.
2. The `EXCLUDE_FOLDERS` skip (psd_analyzer.py:994) splits the absolute folder path into parts —
   `Markers`/`MDW` still match as parts. Unaffected.
3. `getmtime(full_path)` and `parser.parse(full_path, relative_path)` use the absolute path for
   file access (correct) and the relative path only for classification/storage. The fix changes
   exactly one behavior: what gets classified and stored.

### FENCED-OFF wrong fix — do not do this

`relative_path = full_path.removeprefix(local_path).strip('/')` (string-prefix stripping with the
absolute clone path) LOOKS equivalent and is not. It breaks when `local_path` carries a trailing
slash, when the walk yields a symlink-resolved or differently-normalized prefix, and it silently
degrades to the same no-op failure mode when the prefix mismatches by one character — the exact
shape of the bug you are fixing. `os.path.relpath(full_path, local_path)` normalizes both sides and
fails loudly (`ValueError`/`..`-prefixed result) instead of silently mangling. The whole incident
happened because a string operation was assumed to be a path operation; do not re-assume it.

### Gate 0c — Promotion route

The fix is a code change classified and gated by **autocroissant-change-control** (branch, review,
the non-negotiables) and must pass the **autocroissant-validation-and-qa** sandbox sweep gates with
the campaign-specific expected numbers in Phase 4 below (option 1 gives the predicted diffs). Do not
merge on "the one-liner is obviously right" — f7c915c was also obviously right.

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
- Suspicious paths > 0 → a local-mode sweep ran with the bug. STOP the campaign; this is now a
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
awk '/ABILITY_EXCLUSIONS: set/{f=1;next} f&&/}/{exit} f&&/",/{n++} END{print "ABILITY_EXCLUSIONS:", n}' commands/psd_analyzer.py
awk '/EXCESSIVE_STAT_EXCLUSIONS: set/{f=1;next} f&&/}/{exit} f&&/",/{n++} END{print "EXCESSIVE_STAT_EXCLUSIONS:", n}' commands/psd_analyzer.py
```

**EXPECTED (2026-07-11):** 18 and 5. (Cross-check that imports agree with source:
`python3 -c "import sys; sys.path.insert(0,'.'); from commands.psd_analyzer import CardValidator as V; print(len(V.ABILITY_EXCLUSIONS), len(V.EXCESSIVE_STAT_EXCLUSIONS))"`
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
| [6] final text after injection + punctuation cleanup | what would be stored | `[type]` in the wrong slot; punctuation cleanup (`\s+([:;,\.\?!])`→`\1`, psd_analyzer.py:538) having eaten a legitimate gap before punctuation |

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

### Option 1 — The relative-path fix + a local-mode regression harness (do this first)

**Theory obligation:** Gate 0a/0b evidence; nothing else in local mode is path-sensitive (verified
Gate 0b). **Why ranked first:** it unblocks fast local iteration for everything below — remote
sweeps cost 1+ API call per changed card and minutes of wall time; local sweeps are free.

**Predicted numbers, stated before running** (all in a SANDBOX; procedure owned by
autocroissant-validation-and-qa):

- Tier A (fast, local timestamps): sandbox force sweep with the fix
  (`update_stats(None, True, True, True, True)` headless) then
  `diff_stats.py <pre.pkl> <post.pkl>` shows `added: 0 removed: 0`, and `field change counts`
  contains ONLY `{'timestamp': 813}` — local mtimes replace remote commit dates; every other field
  byte-identical. old_stats grows by one archived copy per card whose mtime is newer than the
  stored timestamp (≤813; on a fresh clone, all). Summary line reads `813 had newer timestamps or
  were new ... 0 cards changed location.`
- Tier B (full equivalence, `use_local_timestamp=False` — costs ~813+ GitHub API calls, needs
  GIT_TOKEN's 5000/hr): `diff_stats.py` reports `0/0/0`, exit 0, and old_stats is unchanged —
  local traversal becomes bit-identical to remote traversal on an unchanged clone. Precondition:
  clone at the same commit as upstream main.
- Both tiers: the sweep's problem report lists exactly today's 5 re-validation cards.
- With the UNFIXED code, Tier A instead shows mass `path` changes and `type` → unknown — run it
  once in the sandbox if you want to see the failure shape; never against live pickles.

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
  psd_analyzer.py:538, which eats a gap that sits directly before punctuation (Lich King's gap
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
| The 5 EXCESSIVE_STAT_EXCLUSIONS | Cards whose summed stat digits exceed 10 | Verify each with parse_one against the visible bars; if the sum is faithful, the >10 rule (not the card) is what needs redesign — the list encodes game content in code. CANDIDATE: data-driven expected-max instead of a name list |

**Structural CANDIDATE (the single highest-leverage exclusion killer):** the validator conflates
`ability == ""` with `ability is None` (`not card.ability`, psd_analyzer.py:690). The parser
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
   post → `0/0/0`; old_stats unchanged (the e7befd5 guard skips archiving when timestamps are
   equal); problem REPORT = exactly the current re-validation set (5 today; fewer only if your
   change predicted the specific disappearances).
3. **Sandbox sweep gate (local mode, after option 1):** Tier A/B numbers exactly as predicted in
   option 1. Any surprise field in `field change counts` = not ready.
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
| Hand-edit stats.pkl to "fix" one card | The cca0aaf → eb9aa84 bad-pickle commit/revert (2025-11-10) is what mistrusted pickle surgery earns | Use `/update_metadata` (main.py:389 → `manual_metadata_entry`, psd_analyzer.py:1300). And know its verified semantics, below |

**What the code shows about manual entries vs the next sweep** (verified 2026-07-11 — state no
more than this): `manual_metadata_entry` creates a missing card as
`CardInfo(name, UNKNOWN, path=f"{path}.psd")` (psd_analyzer.py:1349-1353) whose `timestamp`
defaults to `0.0` (CardInfo field default, psd_analyzer.py:128); the command cannot set a
timestamp. `_should_update_card` (psd_analyzer.py:775) reparses when
`stored_timestamp < new_timestamp` or the path changed — so a manually CREATED entry (timestamp
0.0) is overwritten by the next sweep of any kind, its manual version archived to old_stats first.
Manual EDITS to an existing entry leave its real timestamp untouched: they survive non-force sweeps
while the PSD is unchanged, and are overwritten by any `force_update:True` sweep or once the PSD's
mtime/commit date advances. Conclusion the code supports: `/update_metadata` is a between-sweeps
correction tool, not a durable fix. Durable fixes live in the parser or in the source PSD.

---

## When NOT to use this skill

- Understanding the parser/domain without changing it (folder classification, layer semantics,
  query DSL, card counts) → **impossibility-cards-reference**.
- How to run/interpret the five diagnostic scripts themselves → **autocroissant-diagnostics-and-tooling**
  (this skill only consumes them).
- A live symptom that is not extraction (bot won't start, commands missing, messages not sending,
  music, diffusion) → **autocroissant-debugging-playbook**.
- The removeprefix bug's full story, evidence, and status → **autocroissant-failure-archaeology**.
- Operating a sweep on the live bot (`/update_stats` etiquette, multi-machine handoff) →
  **autocroissant-run-and-operate**.
- Sandbox construction and the general acceptance procedure → **autocroissant-validation-and-qa**;
  commit/push/pickle discipline → **autocroissant-change-control**.
- The evidence bar and idea lifecycle in the abstract → **autocroissant-research-methodology**;
  the two long-horizon ambitions → **autocroissant-research-frontier**.

## Provenance and maintenance

Everything dated 2026-07-11 was measured that day on the owner's Mac against stats.pkl at commit
`284d13c` and the local TTSCardMaker clone (upstream latest 2026-06-18). Line numbers drift; each
volatile fact below has a one-line re-verification. After any landed parser change, re-run Phase 1
and append a scoreboard row; if a Phase 3 option ships or retires, move its entry (retirements are
documented in autocroissant-failure-archaeology).

- Bug line (was psd_analyzer.py:1002) and its content:
  `grep -n 'removeprefix("TTSCardMaker")' commands/psd_analyzer.py`
- Bug introduction in f7c915c:
  `git show f7c915c -- commands/psd_analyzer.py | grep -B3 -A3 removeprefix`
- Slash defaults use_local_repo=True (main.py:361), force_update=False (main.py:363):
  `grep -n "use_local_repo: Optional" main.py; grep -n "force_update: Optional" main.py`
- Python-side `update_stats` force_update=True default (psd_analyzer.py:1177):
  `grep -n "force_update: bool = True" commands/psd_analyzer.py`
- `_populate_types_from_local` unaffected (psd_analyzer.py:883-890):
  `grep -n "_populate_types_from_local" commands/psd_analyzer.py` then read those 8 lines.
- Pipeline line numbers — gap regex :348, TYPE_REGION_RATIO :30, card_mid_y :405, prune :586,
  inject :593, leftover-append :632-635, punctuation cleanup :538, `not card.ability` :690:
  `grep -n "_gap_pattern\|TYPE_REGION_RATIO\|card_mid_y = int\|_prune_type_bboxes\|_inject_type_names\|not card.ability" commands/psd_analyzer.py`
  and for the cleanup line `grep -n "ability_text = sub" commands/psd_analyzer.py`
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
- Manual-entry semantics (CardInfo timestamp default :128, `_should_update_card` :775,
  `manual_metadata_entry` :1300, CardInfo creation :1349-1353; /update_metadata at main.py:389):
  `grep -n "timestamp: float = \|def _should_update_card\|def manual_metadata_entry" commands/psd_analyzer.py; grep -n "update_metadata" main.py`
- `/update_stats` operating rule (`use_local_repo:False` until the fix ships): owned by
  autocroissant-run-and-operate; confirm the bug is still live with Gate 0a before relying on it.
