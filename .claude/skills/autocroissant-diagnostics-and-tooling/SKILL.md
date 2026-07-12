---
name: autocroissant-diagnostics-and-tooling
description: Load this skill whenever you need to MEASURE something in AutoCroissant instead of eyeballing it — trigger phrases include "diagnose", "inspect pickle", "dump layers", "trace injection", "diff stats", "is the database healthy", "measure why ONE specific card parsed the way it did", "what changed between two runs", "run the pre-push pickle diff (diff_stats)", "measure", "verify the parse", or any symptom like wrong [type] placement in ability text, missing stats, a card with the wrong type, or a suspicious stats.pkl before a commit. It documents the five read-only diagnostic scripts in .claude/skills/autocroissant-diagnostics-and-tooling/scripts/ (dump_psd_layers.py, parse_one.py, gap_trace.py, inspect_pickle.py, diff_stats.py): exact invocations, annotated golden outputs, failure modes, the pre-pickle-push diff gate, the single-card investigation pipeline, the DB health-check cadence, and the conventions for adding a new diagnostic script.
---

# AutoCroissant diagnostics and tooling

Measure, don't eyeball. This repo has no tests and no CI; the pickles ARE the database and the
costliest historical failures were corrupted pickle commits and silent parse regressions. The five
scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/` turn "it looks fine"
into numbers you can compare, diff, and re-run. All five are read-only: they never write a pickle,
never touch the network, never run the bot.

All commands below assume cwd = repo root `/Users/michaelsrouji/Desktop/AutoCroissant`.
Three of the five (`parse_one.py`, `gap_trace.py`, `inspect_pickle.py`) also work from anywhere —
they derive the repo root from their own location and `chdir` there at import. `diff_stats.py`
does NOT chdir: relative `.pkl` arguments resolve against your SHELL's cwd, so run the push gate
from the repo root (or pass absolute pickle paths). `dump_psd_layers.py` has no repo-root logic
at all and runs from anywhere — it needs only psd-tools and an absolute/`~` PSD path.

**Noise lines**: every script that imports bot code prints one to three
`Git token found, API limited to 5000 requests/hour.` lines from the config import chain
(plus `Trying to open stats.pkl` / `Loaded existing stats...` progress lines where the DB is
loaded). Ignore them or filter with `| grep -v "Git token"`. `dump_psd_layers.py` imports no
bot code and prints no noise. WARNING: do not pipe `diff_stats.py` through grep when you need
its exit code — the pipeline exit is grep's, not the script's (see the push gate below).

Golden outputs quoted below are snapshots of 2026-07-11 — the canonical current expected values
live in **autocroissant-validation-and-qa §3**; if a golden output stops matching, check there
before assuming a regression.

## Which tool answers which question

| Question | Tool | Invocation shape |
|---|---|---|
| Is the DB healthy right now? | `inspect_pickle.py` summary | no args |
| Which cards have recorded parse problems? | `inspect_pickle.py --problems` | one flag |
| What is stored for card X, and its history? | `inspect_pickle.py "Card Name"` | name (case-insensitive) |
| Why did THIS card parse wrong? | `dump_psd_layers.py` → `gap_trace.py` → `parse_one.py`, in that order | see the pipeline workflow |
| What does the parser physically see in this PSD? | `dump_psd_layers.py` (`--text` for raw text) | psd path |
| Why is a `[type]` injected in the wrong place / missing? | `gap_trace.py` | psd path |
| What would /update_stats extract for one card? | `parse_one.py` (add `--with-db` to match a real run) | psd path |
| Is this pickle safe to push? | `diff_stats.py` HEAD vs working (the push gate) | two pkl paths |
| What changed between two runs / two snapshots? | `diff_stats.py` old vs new | two pkl paths |
| Did old_stats.pkl balloon? | `inspect_pickle.py` summary (archived-versions count) + `git diff --stat HEAD -- old_stats.pkl` | — |

What these tools do NOT answer: whether a measured value is *acceptable*
(autocroissant-validation-and-qa), what a card-domain value *means*
(impossibility-cards-reference), or how to *fix* what you found
(autocroissant-psd-extraction-campaign via autocroissant-change-control).

## Shared mechanics (read once)

- Scripts live in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`. Repo-root
  handling differs per script (verified 2026-07-11): `parse_one.py`, `gap_trace.py`, and
  `inspect_pickle.py` each compute `REPO_ROOT = Path(__file__).resolve().parents[4]`, insert it
  on `sys.path`, and `os.chdir(REPO_ROOT)` at import (pickle paths in `global_config.py` are
  repo-relative). `diff_stats.py` computes REPO_ROOT and inserts it on `sys.path` but does NOT
  chdir — its relative pickle arguments resolve against the shell's cwd (from a non-repo cwd the
  documented push-gate invocation exits 2 with `ERROR loading pickles: ... 'stats.pkl'`).
  `dump_psd_layers.py` imports only `sys` + psd-tools: no repo-root logic, no `config.py` needed.
- Every script except `dump_psd_layers.py` needs `config.py` present in the repo root (secret;
  the bot module import chain reads it). Without it the import chain fails.
- `parse_one.py` and `gap_trace.py` additionally need the TTSCardMaker clone at
  `~/Desktop/TTSCardMaker` (`LOCAL_DIR_LOC`, global_config.py:4) — both for the PSD itself and
  to populate the known-types list from the clone's `Types/` folder.
- Because `parse_one.py` and `gap_trace.py` `chdir` to the repo root at import, a **relative**
  PSD argument resolves against the repo root, not your shell's cwd and not the clone. Always
  pass absolute or `~/` PSD paths — to every script. Verified 2026-07-11:
  `parse_one.py Auxiliary/Minions/Mini_Doomer.psd` →
  `ERROR: file not found: /Users/michaelsrouji/Desktop/AutoCroissant/Auxiliary/...`.
- Stale-`.so` trap: `parse_one.py`, `gap_trace.py`, `inspect_pickle.py` import
  `commands/psd_analyzer.py`. If a compiled Cython `.so` exists it shadows the `.py`, so your
  parser edits silently won't be what these tools exercise. None exist as of 2026-07-11; if in
  doubt run `python3 setup.py clean` (see autocroissant-build-and-env).

---

## 1. dump_psd_layers.py — what is physically in the PSD

**Purpose**: print the layer tree exactly as the parser will see it — name, kind, visibility,
pixel presence, bbox — plus (with `--text`) the raw engine text of every text layer. This is
step 1 of any "card parsed wrong" investigation: before blaming the parser, confirm what the
file contains. Needs only `psd-tools`; no `config.py`, no bot imports, no noise lines, works on
any PSD anywhere.

**Invocation** (from repo root):

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" --text
```

**Annotated output** (The Lich King, verified 2026-07-11; excerpted):

```
PSD: /Users/michaelsrouji/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd
size: 1200 x 2400  (mid-y at ratio 0.5 = 1200)          <- midline: icons above = creature types,
                                                            below = inline-injection candidates
layer (indent = depth)                    kind        vis  pix  bbox
Types                                     group       True False (18, 21, 159, 308)
  Ice                                     pixel       True True (21, 158, 156, 308)   <- above mid-y
  Undead                                  pixel       True True (18, 21, 159, 135)    <- above mid-y
Stats Bars                                group       True False (360, 1200, 1173, 1662)
  Darks                                   group       True False (0, 0, 0, 0)
    SPD Dark                              group       True False (0, 0, 0, 0)
      5                                   exposure    True False (0, 0, 0, 0)   <- the ONE visible
      6                                   exposure    False False (0, 0, 0, 0)     digit => spd=5
...
  Ability                                 type        True True (23, 1773, 1176, 2285)
     TEXT: "'(OFS) Equip Frostmourne.\\r...These minions die at the end of their turn.\\rMinions inherit       .\\r'"
...
Undead                                    smartobject True True (734, 2239, 808, 2299) <- BELOW mid-y:
                                                                     inline-injection candidate
```

**Interpretation guide**:

| Column / line | Meaning |
|---|---|
| `size` + `mid-y` | The midline that splits type icons into creature-types (above) vs inline candidates (below). NOTE: this script hardcodes ratio 0.5 in the printout; `gap_trace.py` imports the real `TYPE_REGION_RATIO`. If the constant ever changes, trust gap_trace's value. |
| `kind` | psd-tools layer kind. `type` means a TEXT layer (yes, confusingly). `group`, `pixel`, `smartobject`, `exposure`, `shape` are the others you'll see. |
| `vis` | `is_visible()`. Visibility drives stat digits (only VISIBLE digit layers are summed) and star counting. A stat "missing" is often a digit layer someone toggled off. |
| `pix` | `has_pixels()`. Star counting and type icons need pixel presence. |
| `bbox` | `(x1, y1, x2, y2)`. Compare y1 against mid-y to predict which side of the midline the parser puts an icon on. |
| `TEXT:` (with `--text`) | Raw engine text before cleanup. Line breaks are `\r` (the parser converts to `\n`). Runs of 3+ literal spaces are the injection gaps — count them here to predict gap_trace's `<GAP:n>` markers. |

Note the two "Undead" layers above: the pixel layer inside `Types` at y=21 (above midline →
goes to the card's `types` list) and the smartobject at y=2239 (below midline → candidate for
inline `[undead]` injection). Same name, two different roles, distinguished purely by position.
That distinction is the heart of most mis-parse investigations.

**Failure modes**: file not found → prints usage/traceback; PSD outside the clone is fine
(this tool is location-agnostic). No exit-code contract beyond 0/1-on-missing-args.

**Questions it answers**: Is the layer named what the parser expects? Is it visible? Which side
of the midline is it on? What does the raw text (and its gap spacing) actually contain?

---

## 2. parse_one.py — what /update_stats would extract for one card

**Purpose**: run ONE PSD through the real bot parser (`PSDParser.parse` + `CardValidator.validate`)
without Discord, without the network, and without touching any pickle. Fastest way to see the
full extracted `CardInfo` and validator problems for a single card.

**Invocation** (from repo root; absolute or `~/` path — relative paths resolve against the repo
root and will fail, see Shared mechanics):

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py \
  ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd
```

**Annotated output** (Mini Doomer, fresh mode, verified 2026-07-11):

```
known types (36): ['.ds_store', 'active', 'attack', ... 'water', 'wind']
                       ^- macOS .DS_Store artifact leaked into all_types when populated
                          locally; harmless (no layer is named '.ds_store') but explains
                          the count being 36, not 35. Remote populate would not include it.
relative path: Auxiliary/Minions/Mini_Doomer.psd     <- computed CORRECTLY by this script
--------------------------------------------------------------------------------
type        : minion
path        : Auxiliary/Minions/Mini_Doomer.psd
timestamp   : 0.0          <- always 0.0 here: timestamps are set by the traverser,
                              not the parser. Never read timestamp from this output.
stars       : 2
types       : ['wind', 'defensive_minion', 'light']
hp          : 5
def         : 2
atk         : 5
spd         : 5
ability:
========================================
Has stats equal to the base stats -2 of the creature that summoned it. 
(BOT) Die.
========================================
validator problems: NONE
```

**Interpretation guide**:

- `known types (N)` — the type vocabulary populated from the clone's `Types/` folder. If a type
  icon "isn't detected", first check its lowercased layer name is in this list.
- `relative path` — this script computes the repo-relative path CORRECTLY (`relpath` against the
  clone root). The bot's own local traversal currently computes it WRONG for absolute clone
  paths (`removeprefix` no-op, psd_analyzer.py:1002) — so parse_one shows the *intended* parse,
  which is NOT what a local-mode `/update_stats` would store today. Story and status: see
  autocroissant-failure-archaeology ("removeprefix bug", commit f7c915c).
- Field lines — `CardInfo.to_dict()` output. Empty fields are OMITTED (to_dict drops
  ability/stars/subtype/series/types/author/problems when empty and stats when invalid), so a
  fresh parse showing no `author:`/`series:` line is normal, not a bug.
- `ability` block between `===` fences — byte-for-byte what would be stored. Trailing spaces and
  blank runs are LOAD-BEARING (they are the injection signal; see the whitespace saga in
  autocroissant-failure-archaeology). Golden check: The Freezer's ability from
  `parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"` matches its stats.pkl
  entry character-for-character (verified 2026-07-11).
- `validator problems` — exactly what `/update_stats` would record in `card.problems`.

**`--with-db` vs fresh** (verified 2026-07-11): `PSDParser.parse` preserves `author` and
`series` from an existing DB entry when they are non-empty (psd_analyzer.py, the
"Preserve author and series" block in `parse()` — grep below). Default (fresh) parses against an
empty DB: no preservation, so Mini Doomer fresh shows neither field, while
`--with-db` adds `series: Kirby` and `author: Chestnut` and prints
`(loaded stats.pkl: 813 cards -- author/series preservation active)`.
Rule of thumb: use **fresh** to test pure extraction; use **`--with-db`** when comparing
field-for-field against the stored entry (otherwise author/series will look like regressions).

**Failure modes**: needs `config.py` in the repo root; needs the clone at `~/Desktop/TTSCardMaker`;
relative argv paths fail (resolve against repo root); a PSD outside the clone prints a WARNING
and classifies as UNKNOWN (classification is folder-based).

**Questions it answers**: What exactly would the parser extract? Would the validator flag it?
Did my parser edit change this card's output (run before/after and diff the two outputs)?

---

## 3. gap_trace.py — why a [type] landed where it did

**Purpose**: trace the type-injection pipeline (the project's hardest recurring problem) step by
step: raw text layers, sorted order, gaps, below-midline icon bboxes, the prune decision, the
gap-vs-type count, and the final injected text. It re-drives the REAL `PSDParser` methods
(`_process_layer`, `_sort_by_position`, `_prune_type_bboxes`, `_process_abilities`) — only the
orchestration mirrors `PSDParser._extract_from_layers`.

**Invocation**:

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"
```

**Annotated output** (The Lich King, verified 2026-07-11; abbreviated):

```
PSD 1200x2400; TYPE_REGION_RATIO=0.5 -> card_mid_y=1200   <- the real imported constant

[1] creature-types collected ABOVE midline: ['ice', 'undead']
        ^- these go to card.types; they are NOT injection candidates

[2] ability text layers found: 1
    at (x=23, y=1773): "'(OFS) Equip Frostmourne.\n...Minions inherit       .\n'"
        ^- raw text + bbox of each text layer, pre-cleanup

[3] joined ability text with gaps marked (3+ spaces = injection slots):
...
Minions inherit<GAP:7>.
        ^- every run of 3+ spaces becomes <GAP:n>. n=7 here: seven literal spaces.
           No <GAP> where you expect one => the text layer lacks 3+ spaces there.

[4] type icons BELOW midline (candidate inline types): 1
    undead          at (x=734, y=2239)
    prune threshold = max(last_y//3=746, card_mid_y=1200)
    kept after prune: ['undead']   dropped: []
        ^- icons with y above the threshold are dropped as "too high to be inline".
           threshold = max(last-icon-y // 3, card_mid_y); here max(746,1200)=1200.

[5] gap count = 1 vs kept types = 1   (match)
        ^- MISMATCH here is the classic wrong-injection signature: more types than
           gaps => leftovers get APPENDED to the last line; more gaps than types
           => some gaps stay as raw runs of spaces.

[6] FINAL ability text after injection + punctuation cleanup:
========================================
...
Minions inherit [undead].
========================================
```

**Interpretation guide, section by section**:

| Section | What to check |
|---|---|
| header | `card_mid_y` — is the icon you care about really below it? (compare bboxes from dump_psd_layers) |
| [1] | Types that ended up as creature types instead of inline candidates. An icon "missing" from injection is often sitting above the midline. |
| [2] | Multiple text layers? `_sort_by_position` orders them top-to-bottom, then left-to-right with 40px row grouping — a mis-ordered join scrambles which line each gap is on. |
| [3] | The `<GAP:n>` markers are the injection slots, per line. Whitespace is load-bearing: never "clean up" multiple spaces in card text (incident 3bbaa2b). |
| [4] | The prune step. An icon dropped here (listed under `dropped:`) was judged too high; an icon kept that shouldn't be usually means the PSD has a stray icon layer below midline. |
| [5] | The count comparison. `(match)` is what you want. On MISMATCH the script prints an explicit warning that leftovers append to the last line. |
| [6] | Byte-exact final text — compare against the stored entry (`inspect_pickle.py "Name"`). |

**MAINTENANCE NOTE** (from the script's docstring — keep it true): the extraction loop mirrors
`PSDParser._extract_from_layers` (commands/psd_analyzer.py:398 as of 2026-07-11). If that method
changes shape, update gap_trace to match; the one-line check is to compare against that method's
body.

**Failure modes**: needs `config.py` and the clone at `~/Desktop/TTSCardMaker`; same
relative-path caveat as parse_one (no explicit warning branch here — a PSD outside the clone
just classifies from a `../...`-style path, i.e. garbage; keep inputs inside the clone).

**Questions it answers**: Which gap got which type and why? Why was an icon pruned? Why did a
type append to the last line? Is the failure in the text (gaps), the icons (position), or the
counts (mismatch)?

---

## 4. inspect_pickle.py — DB health and single-card lookup

**Purpose**: read-only view of `stats.pkl` / `old_stats.pkl`: a health summary, the list of
cards with recorded problems, or one card in full detail with its archived history.

**Invocations**:

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py              # summary
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --problems   # problem cards
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "the lich king"  # one card
```

**Annotated summary output** (golden baseline, verified 2026-07-11):

```
stats.pkl     : 813 cards
old_stats.pkl : 218 cards with history, 223 archived versions total
                    ^- watch this number across updates: sudden ballooning =
                       duplicate-archiving regression (the cca0aaf/eb9aa84 incident)
card_type distribution:
  creature     297
  item         182
  minion       91
  debuff       52
  field        50
  aux item     48
  rulebook     48
  nme          26
  buff         12
  creatures    7      <- yes, both 'creature' and 'creatures' exist; a known
                         classification anomaly, see impossibility-cards-reference
cards with recorded parse problems: 1
cards without author (orphans)    : 0
rulebook entries                  : 48
newest timestamp: Anubisath Guardian @ 2026-06-18 02:47:02
                    ^- should match the latest TTSCardMaker activity; a stale value
                       means updates haven't run
entries with suspicious (non repo-relative) paths: 0
                    ^- MUST be 0. See heuristic note below.
```

`--problems` (verified 2026-07-11) lists exactly one card:
`20 Creature Types [Auxiliary/Rulebook/20_Creature_Types.psd]: ['MISSPELT TYPE: tornado']` —
the known baseline. Anything beyond this line is new.

**Card lookup** (`"the lich king"`): resolution order is exact name → case-insensitive match →
substring candidates printed as `did you mean: ...` (up to 10) with exit 1. Verified: lowercase
`the lich king` resolves to `=== The Lich King ===` and prints every stored field (timestamp
rendered human-readable) followed by the archive:

```
old versions: 2
  [0] ts=2026-01-30 18:58:46 type=creature ability[:60]='(OFS) Equip Frostmourne.\n(1PG)...'
  [1] ts=2026-05-28 04:53:59 type=creature ability[:60]='...'
```

Each `[i]` line is one archived CardInfo (timestamp, type, first 60 ability chars) — use it to
see when a card's stored text last changed.

**The "suspicious paths" heuristic** (read the code before trusting it blindly): it flags any
entry whose path's top folder is not in
`("Creatures", "Items", "Field", "Auxiliary", "N.M.E", "MDW", "Rulebook", "Types")`.
It exists to catch exactly the removeprefix corruption signature (paths starting `Users/...`).
Two caveats: (a) a legitimately NEW top-level folder in TTSCardMaker would false-positive —
update the tuple in the script if the card repo grows one; (b) 0 here does not prove every path
is *correct*, only that top folders look sane.

**Failure modes**: needs `config.py`; loads the committed pickles from the repo root (always
present in a checkout). Missing card name → `not found` + suggestions, exit 1.

**Questions it answers**: Is the DB healthy? What is stored for card X and when did it last
change? Which cards are flagged? Did old_stats balloon?

---

## 5. diff_stats.py — the numbers between two stats.pkl snapshots

**Purpose**: diff two stats.pkl-format snapshots into added/removed/modified with per-field
counts. THE gate before every pickle push — it turns "the pickle looks fine" into numbers.
Handles both modern CardInfo pickles and legacy plain-dict pickles (pre-366c8d9), so it can diff
across the Oct-2025 schema change. It does NOT diff `old_stats.pkl` (dict-of-lists shape →
load error, exit 2).

**Exit codes** (all verified 2026-07-11): `0` = no changes; `1` = changes found; `2` = usage or
load error. Do not pipe through grep/head when you consume the exit code.

**Run it from the repo root (or pass absolute paths).** Unlike the other pickle-reading scripts,
diff_stats.py does not chdir — a relative argument like `stats.pkl` resolves against your shell's
cwd, and from anywhere else the gate exits 2 with `ERROR loading pickles: ... 'stats.pkl'`
(verified 2026-07-11).

**Invocation A — the pre-push gate** (HEAD vs working copy):

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl && \
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
echo $?
```

Golden output on a clean tree (verified 2026-07-11):

```
old: 813 cards   new: 813 cards
added: 0   removed: 0   modified: 0
```
exit 0 — the working pickle is identical to HEAD; there is no data change to push.

**Invocation B — what changed between two runs/snapshots** (any two pickles; here the two most
recent PICKLE commits, verified 2026-07-11):

```bash
git show 8753489:stats.pkl > /tmp/stats_old.pkl && git show HEAD:stats.pkl > /tmp/stats_new.pkl && \
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_old.pkl /tmp/stats_new.pkl
```

Annotated real output (8753489 → 284d13c):

```
old: 808 cards   new: 813 cards
added: 6   removed: 1   modified: 114
-- added --
  + Anubisath Guardian [Auxiliary/Minions/Anubisath_Guardian.psd]
  ...
  + Z!!!!! [Creatures/Other/5 Stars/Z!!!!!.psd]
-- removed --  (unexpected removals = traversal problem!)
  - ZOEY!!!!! [Creatures/Other/5 Stars/ZOEY!!!!!.psd]
        ^- before panicking, look for a matching "added" entry: ZOEY!!!!! -> Z!!!!!
           is a RENAME (same folder), which is expected, not data loss.
-- modified --
  field change counts: {'series': 111, 'timestamp': 3, 'ability': 1, 'types': 1}
        ^- READ THIS LINE FIRST. It is the mass-change detector. 111 'series'
           changes across 114 modified cards = one deliberate bulk series edit,
           not 111 independent regressions. Counts can exceed the modified-card
           count sum-wise: one card may change several fields.
  * 20 Creature Types: ['ability', 'timestamp']
  * A Snack: ['series']
  ...
```
exit 1 (changes found). Add `--verbose` to print `old -> new` values per field per card.

**Red flags** (from the script's docstring; each maps to a known incident):

| Signal | Meaning | Where the story lives |
|---|---|---|
| `removed` cards you did not delete (and no matching `added` rename) | traversal missed files — wrong mode or path | autocroissant-failure-archaeology |
| mass `path` changes in field counts | the removeprefix local-path bug fired — STOP, do not push | autocroissant-failure-archaeology (commit f7c915c) |
| mass `type` → unknown (check with `--verbose`) | classification broke | impossibility-cards-reference for what classify should do |
| huge growth in archived old_stats versions | duplicate-archiving regression (cca0aaf → eb9aa84 revert) | NOT visible in this script's output — measure with `inspect_pickle.py` summary (223 archived versions is the 2026-07-11 baseline) and `git diff --stat HEAD -- old_stats.pkl` |

Note on that last row: the docstring lists it among the red flags, but `diff_stats.py` reads
only stats.pkl-shaped files — old_stats growth is checked with the two commands above.

**Failure modes**: fewer than two args → prints usage, exit 2; unreadable/missing/incompatible
pickle → `ERROR loading pickles: ...` plus a reminder that unpickling CardInfo requires a
checkout defining it with `config.py` present, exit 2.

**Questions it answers**: Is this pickle safe to push? What exactly did the last update change?
Did a parser change touch cards it shouldn't have?

---

## Standard workflows

### Workflow 1 — pre-pickle-push gate (run EVERY time before pushing pickles, from the repo root)

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl && \
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
echo $?
```

1. exit 0 → no data change; nothing pickle-related to push.
2. exit 1 → review each bucket:
   - `added`: only cards you know are new in TTSCardMaker.
   - `removed`: pair each with a possible `added` rename; a true unexpected removal = traversal
     problem, do not push.
   - `modified`: read `field change counts` first. Mass `path` or mass `type` = red flag, stop.
     Legit updates look like a handful of `ability`/`timestamp`/`types` changes on cards you
     know changed. Use `--verbose` on anything surprising.
3. Check old_stats separately: `git diff --stat HEAD -- old_stats.pkl` plus the archived-versions
   count in `inspect_pickle.py` summary (baseline 223 as of 2026-07-11).
4. exit 2 → the gate did not run; fix the load error before doing anything else.
5. The commit/push itself (PICKLE convention, who may push) is owned by
   autocroissant-change-control — this gate is a precondition, not a substitute.

Extra rule as of 2026-07-11: any `/update_stats` run in LOCAL mode (`use_local_repo:True`) makes
this gate MANDATORY before any pickle push, because of the live removeprefix bug (see
autocroissant-failure-archaeology; operating rule in autocroissant-run-and-operate).

### Workflow 2 — single-card investigation pipeline ("why did THIS card parse wrong?")

Run in this order; each step tells you whether to continue:

```bash
# 0. What is stored now, and its history? (optional but cheap)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "The Lich King"

# 1. What is physically in the file? (layer names, visibility, bbox vs midline, raw text)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" --text

# 2. How did injection decide? (gaps vs icons vs prune vs counts)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"

# 3. Full parse + validator verdict (add --with-db to compare against stored fields)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" --with-db
```

Decision points: if step 1 shows the layer is missing/hidden/misnamed/on the wrong side of the
midline, the card FILE is the problem (fix in TTSCardMaker, not the bot). If step 1 looks right
but step 2 shows a gap/type mismatch or a bad prune, the heuristics are the problem — take it to
autocroissant-psd-extraction-campaign. If steps 1-2 look right but step 3 disagrees with the
stored entry, the stored entry is stale or was written by a buggy traversal — check history in
step 0 and the update mode used.

### Workflow 3 — DB health check cadence

Run `inspect_pickle.py` (summary) then `inspect_pickle.py --problems`:
- after every `/update_stats`,
- after `/pull` on a machine handoff (the pickles are the only synced state),
- before nominating pickles for a push (alongside Workflow 1),
- whenever a query result looks wrong.

Compare against the 2026-07-11 baseline: 813 cards, 218/223 old-stats, 1 problem card
(20 Creature Types), 0 orphans, 0 suspicious paths, newest = Anubisath Guardian @ 2026-06-18.
Card counts legitimately grow as cards are added; problems/orphans/suspicious-paths should not.

---

## Adding a new diagnostic script

Conventions (match the existing five — read one before writing):

1. `#!/usr/bin/env python3` and a module docstring that states usage, what it prints, and the
   read-only guarantee. Scripts print `__doc__` on missing args.
2. If it imports bot code or the pickles (the pattern parse_one/gap_trace/inspect_pickle follow):
   ```python
   REPO_ROOT = Path(__file__).resolve().parents[4]   # scripts/ is 4 levels below repo root
   sys.path.insert(0, str(REPO_ROOT))
   os.chdir(REPO_ROOT)  # pickle paths in global_config are repo-relative
   ```
   and expect the `Git token found...` noise lines.
3. READ-ONLY, always: never call `save()`, `update_stats()`, anything network-mutating, and never
   write a pickle. If a script needs scratch output, write to `/tmp` or a scratchpad, never the repo.
4. Prefer re-driving REAL parser methods over copying their logic; where orchestration must be
   mirrored, add a MAINTENANCE NOTE naming the mirrored function (see gap_trace.py's docstring).
5. `def main() -> int` + `sys.exit(main())`; document exit codes if callers branch on them.
6. THEN update this SKILL.md: add a routing-table row, a per-tool section (purpose, invocation,
   annotated output, failure modes, questions answered), and a provenance re-run line. A script
   without a section here does not exist as far as the skill library is concerned.

## When NOT to use this skill

- **Interpreting card-domain semantics** (what a folder maps to, why `creatures` vs `creature`,
  what stars/subtypes mean, the injection algorithm's constants) → **impossibility-cards-reference**.
- **Deciding whether measured results are acceptable** (golden-card inventory, acceptance
  procedure for parser changes, when a problem-count increase blocks a change) →
  **autocroissant-validation-and-qa**.
- **Fixing what you found**: parser/heuristic changes → **autocroissant-psd-extraction-campaign**;
  committing, pushing, or reverting anything (including pickles that passed the gate) →
  **autocroissant-change-control**.
- **Live bot symptom triage** (bot won't start, commands missing, messages not sending) →
  **autocroissant-debugging-playbook**.
- **The stories behind the red flags** (removeprefix bug, whitespace saga, old_stats ballooning)
  → **autocroissant-failure-archaeology**.
- **First-principles analysis beyond these five tools** (API budgets, queue timing, designing a
  new experiment) → **autocroissant-analysis-toolkit**.

## Provenance and maintenance

All facts and golden outputs verified 2026-07-11 by running the scripts on this Mac (Python
3.10.20, clone at `~/Desktop/TTSCardMaker`). Counts (813/218/223/1/0), the newest-card line, and
the two-snapshot diff figures drift with every PICKLE commit — re-run before quoting.

Re-verification one-liners (from repo root):

```bash
S=.claude/skills/autocroissant-diagnostics-and-tooling/scripts
ls .claude/skills/autocroissant-diagnostics-and-tooling/scripts/
# expect: diff_stats.py dump_psd_layers.py gap_trace.py inspect_pickle.py parse_one.py

# golden: DB summary (813 cards, 223 archived, 1 problem, 0 orphans, 0 suspicious)
python3 $S/inspect_pickle.py | grep -v "Git token"

# golden: Mini Doomer (minion, stars 2, ['wind','defensive_minion','light'], 5/2/5/5, problems NONE)
python3 $S/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd | grep -v "Git token"

# golden: The Freezer ability text matches stats.pkl character-for-character
python3 $S/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd" | grep -v "Git token"

# golden: Lich King trace (<GAP:7>, undead @ x=734,y=2239, prune max(746,1200)=1200, match, "Minions inherit [undead].")
python3 $S/gap_trace.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" | grep -v "Git token"

# golden: clean-tree gate is 0/0/0 with exit 0
git show HEAD:stats.pkl > /tmp/stats_head.pkl && python3 $S/diff_stats.py /tmp/stats_head.pkl stats.pkl; echo $?

# exit codes: no-args is 2; two differing snapshots is 1
python3 $S/diff_stats.py >/dev/null 2>&1; echo $?

# dump header (1200 x 2400, mid-y 1200)
python3 $S/dump_psd_layers.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" | head -2
```

Cited code locations (line numbers drift; re-find with these):

```bash
grep -n "TYPE_REGION_RATIO = " commands/psd_analyzer.py            # 0.5 (line 30 as of 2026-07-11)
grep -n "_extract_from_layers" commands/psd_analyzer.py            # gap_trace's mirrored method (def at 398)
grep -n "Preserve author and series" commands/psd_analyzer.py      # parse_one --with-db semantics
grep -n "removeprefix" commands/psd_analyzer.py                    # the live local-path bug (line 1002)
grep -n "LOCAL_DIR_LOC" global_config.py                           # "~/Desktop/TTSCardMaker" (line 4)
grep -rn "def to_dict" commands/psd_analyzer.py                    # field-omission behavior in output
```

If `_extract_from_layers` changes shape, update gap_trace.py (its docstring says so). If
TTSCardMaker grows a new top-level folder, update inspect_pickle.py's known-top-folders tuple.
If a script is added, this document must gain a routing row, a section, and a provenance line.
