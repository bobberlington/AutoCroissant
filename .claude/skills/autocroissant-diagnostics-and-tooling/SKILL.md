---
name: autocroissant-diagnostics-and-tooling
description: "Load this skill whenever you need to MEASURE something in AutoCroissant instead of eyeballing it — trigger phrases include \"diagnose\", \"inspect pickle\", \"dump layers\", \"trace injection\", \"diff stats\", \"is the database healthy\", \"measure why ONE specific card parsed the way it did\", \"what changed between two runs\", \"run the data commit diff (diff_stats)\", \"measure\", \"verify the parse\", or any symptom like wrong [type] placement in ability text, missing stats, a card with the wrong type, or a suspicious stats.pkl before a commit. It documents the five read-only diagnostic scripts in .claude/skills/autocroissant-diagnostics-and-tooling/scripts/ (dump_psd_layers.py, parse_one.py, gap_trace.py, inspect_pickle.py, diff_stats.py; the last two read db/ JSON too): exact invocations, annotated golden outputs, failure modes, the data gate (diff of Card sync data commits; until 2026-10-25 the pre-pickle-push gate) and the shadow's daily check, the single-card investigation pipeline, the DB health-check cadence, and the conventions for adding a new diagnostic script."
---

# AutoCroissant diagnostics and tooling

Measure, don't eyeball. This repo has no CI and only an offline pytest suite (`tests/`); the card data is the Card sync
Action's `db/` JSON on TTSCardMaker's `data` branch (the bot's copy in `card_data/`; this repo's pickles until
2026-10-25, in git history), and the costliest historical failures were corrupted pickle commits and silent parse
regressions. The five scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/` turn "it looks fine"
into numbers you can compare, diff, and re-run. All five are read-only: they never write card data, never touch the
network, never run the bot.

All commands below assume cwd = repo root `/Users/michaelsrouji/Desktop/AutoCroissant`.
Two of the five (`parse_one.py`, `gap_trace.py`) also work from anywhere —
they derive the repo root from their own location and `chdir` there at import. `inspect_pickle.py`
reads the repo root's `card_data/` from anywhere without a chdir (2026-10-25). `diff_stats.py`
does NOT chdir: relative arguments resolve against your SHELL's cwd (pass data commits with `--git`, or absolute
paths). `dump_psd_layers.py` has no repo-root logic
at all and runs from anywhere — it needs only psd-tools and an absolute/`~` PSD path.

**Which Python (2026-10-23).** `diff_stats.py` and `inspect_pickle.py` run on the Mac's plain `python3` (Homebrew
3.14.8). `parse_one.py`, `gap_trace.py` and `dump_psd_layers.py` need psd-tools and pandas, which that `python3` lacks
("No module named pandas" / "psd_tools"): use the production-equivalent venv `~/.venvs/ac-py314-fork` (the psd-tools fork,
tag `cards-20261023` = ecc4b773, with `range_reader`, as on the jetson), or `~/.venvs/ac-py314-psd125` (stock psd-tools
1.25.0). Below, `$P` means `~/.venvs/ac-py314-fork/bin/python` for those three.

**The card data in any form (2026-10-23).** `diff_stats.py` and `inspect_pickle.py` read it through
`commands.card_db.read_cards`: a `db/` folder (or a folder holding `db/`: `card_data/`, an Actions artifact), a
`stats.json` (the old_stats file beside it is the history), a data commit of `--git`'s repo (`origin/data` in a
TTSCardMaker clone), or a historical `stats.pkl` / folder of pickles (`git show <PICKLE commit>:stats.pkl`; the pickles
left this repo on 2026-10-25). Neither needs `config.py`: pickles load through card_db's allow-list unpickler, which
knows only the card classes.

**Noise lines**: since 2026-10-20 the `Git token found, API limited to 5000 requests/hour.` line is
printed only by main.py (`print_token_status`), so scripts print none; the `| grep -v "Git token"`
filters below are harmless leftovers. `dump_psd_layers.py` imports no
bot code and prints no noise. WARNING: do not pipe `diff_stats.py` through grep when you need
its exit code — the pipeline exit is grep's, not the script's (see the push gate below).

Golden outputs quoted below are dated snapshots (2026-07-11, some re-measured 2026-10-23) — the canonical current expected values
live in **autocroissant-validation-and-qa §3**; if a golden output stops matching, check there
before assuming a regression.

## Which tool answers which question

| Question | Tool | Invocation shape |
|---|---|---|
| Is the DB healthy right now? | `inspect_pickle.py` summary (the bot's `card_data/`) | no args |
| Which cards have recorded parse problems? | `inspect_pickle.py --problems` | one flag |
| What is stored for card X, and its history? | `inspect_pickle.py "Card Name"` | name (case-insensitive) |
| Why did THIS card parse wrong? | `dump_psd_layers.py` → `gap_trace.py` → `parse_one.py`, in that order | see the pipeline workflow |
| What does the parser physically see in this PSD? | `dump_psd_layers.py` (`--text` for raw text) | psd path |
| Why is a `[type]` injected in the wrong place / missing? | `gap_trace.py` | psd path |
| What would the card data build extract for one card? | `parse_one.py` (add `--with-db` to match a real run) | psd path |
| What did a Card sync run change? (the data gate) | `diff_stats.py origin/data~1 origin/data --git ~/Desktop/TTSCardMaker` | Workflow 1 |
| What changed between two runs / two snapshots? | `diff_stats.py` old vs new | two db/ folders, data commits or historical pickles |
| What does the data build hold for card X? | `inspect_pickle.py --db <db/ or data commit> [--git DIR] "Card Name"` | as above |
| Did old_stats balloon? | `diff_stats.py --history` between two data commits (`versions added`) | — |

What these tools do NOT answer: whether a measured value is *acceptable*
(autocroissant-validation-and-qa), what a card-domain value *means*
(impossibility-cards-reference), or how to *fix* what you found
(autocroissant-psd-extraction-campaign via autocroissant-change-control).

## Shared mechanics (read once)

- Scripts live in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`. Repo-root
  handling differs per script (re-verified 2026-10-23): `parse_one.py` and `gap_trace.py` compute
  `REPO_ROOT = Path(__file__).resolve().parents[4]`, insert it on `sys.path`, and `os.chdir(REPO_ROOT)`
  at import (`card_data/` is repo-relative). `inspect_pickle.py` and `diff_stats.py`
  only put REPO_ROOT on `sys.path` and import `commands.card_db` (and card_models / card_history); neither chdirs.
  inspect_pickle reads the repo root's `card_data/` by default; diff_stats's relative arguments resolve against the
  shell's cwd.
  `dump_psd_layers.py` imports only `sys` + psd-tools: no repo-root logic, no `config.py` needed.
- `parse_one.py` and `gap_trace.py` need `config.py` present in the repo root (secret;
  the bot module import chain reads it). Without it the import chain fails. `diff_stats.py` and
  `inspect_pickle.py` need none since 2026-10-23 (they import only `commands.card_db` and `card_models`), so
  the Mac's plain `python3` runs them.
- `parse_one.py` and `gap_trace.py` additionally need the TTSCardMaker clone at
  `~/Desktop/TTSCardMaker` (`LOCAL_DIR_LOC`, global_config.py:5) — both for the PSD itself and
  to populate the known-types list from the clone's `Types/` folder.
- Because `parse_one.py` and `gap_trace.py` `chdir` to the repo root at import, a **relative**
  PSD argument resolves against the repo root, not your shell's cwd and not the clone. Always
  pass absolute or `~/` PSD paths — to every script. Verified 2026-07-11:
  `parse_one.py Auxiliary/Minions/Mini_Doomer.psd` →
  `ERROR: file not found: /Users/michaelsrouji/Desktop/AutoCroissant/Auxiliary/...`.
- Stale-`.so` trap: `parse_one.py` and `gap_trace.py` import `commands/psd_analyzer.py`. If a compiled
  Cython `.so` exists it shadows the `.py`, so your parser edits silently won't be what these tools exercise
  (`inspect_pickle.py` and `diff_stats.py` import only uncompiled card_db/card_history and card_models). None
  exist on the Mac as of 2026-10-23; if in doubt run `python3 setup.py clean` (see autocroissant-build-and-env).

---

## 1. dump_psd_layers.py — what is physically in the PSD

**Purpose**: print the layer tree exactly as the parser will see it — name, kind, visibility,
pixel presence, bbox — plus (with `--text`) the raw engine text of every text layer. This is
step 1 of any "card parsed wrong" investigation: before blaming the parser, confirm what the
file contains. Needs only `psd-tools`; no `config.py`, no bot imports, no noise lines, works on
any PSD anywhere.

**Invocation** (from repo root):

```bash
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py \
  "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" --text
```

**Annotated output** (The Lich King, re-measured 2026-10-23 at TTSCardMaker a7d1fe56; excerpted):

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
  Ability                                 type        True True (23, 1767, 1176, 2293)
     TEXT: "'(OFS) Equip Frostmourne.\\r...These minions die at the end of their turn.\\rMinions inherit [undead].\\r'"
...
Undead                                    smartobject False True (734, 2239, 808, 2299) <- hidden since the
                                                                     2026-10-22 batch: the font token prints it
```

**Interpretation guide**:

| Column / line | Meaning |
|---|---|
| `size` + `mid-y` | The midline that splits type icons into creature-types (above) vs inline candidates (below). NOTE: this script hardcodes ratio 0.5 in the printout; `gap_trace.py` imports the real `TYPE_REGION_RATIO`. If the constant ever changes, trust gap_trace's value. |
| `kind` | psd-tools layer kind. `type` means a TEXT layer (yes, confusingly). `group`, `pixel`, `smartobject`, `exposure`, `shape` are the others you'll see. |
| `vis` | `is_visible()`. Visibility drives stat digits (only VISIBLE digit layers count, per bar row: see impossibility-cards-reference §3.4) and star counting. A stat "missing" is often a digit layer someone toggled off. |
| `pix` | `has_pixels()`. Star counting and type icons need pixel presence. |
| `bbox` | `(x1, y1, x2, y2)`. Compare y1 against mid-y to predict which side of the midline the parser puts an icon on. |
| `TEXT:` (with `--text`) | Raw engine text before cleanup. Line breaks are `\r` (the parser converts to `\n`). Runs of 3+ literal spaces are the injection gaps — count them here to predict gap_trace's `<GAP:n>` markers. |

Note the two "Undead" layers above: the pixel layer inside `Types` at y=21 (above midline →
goes to the card's `types` list) and the smartobject at y=2239 (below midline). Until the standardization batch
(TTSCardMaker 1c24dd48, 2026-10-22) that smartobject was visible and filled the 7-space gap of
"Minions inherit       ." as an inline `[undead]`; since then the text holds the stoogedings token `[undead]` and
the icon layer is hidden, so the card no longer exercises gap injection. Same name, two different roles,
distinguished purely by position: the heart of most mis-parse investigations on cards that still use icon layers.

**Failure modes**: file not found → prints usage/traceback; PSD outside the clone is fine
(this tool is location-agnostic). No exit-code contract beyond 0/1-on-missing-args.

**Questions it answers**: Is the layer named what the parser expects? Is it visible? Which side
of the midline is it on? What does the raw text (and its gap spacing) actually contain?

---

## 2. parse_one.py — what the card data build would extract for one card

**Purpose**: run ONE PSD through the real bot parser (`PSDParser.parse` + `CardValidator.validate`)
without Discord, without the network, and without writing anything. Fastest way to see the
full extracted `CardInfo` and validator problems for a single card.

**Invocation** (from repo root; absolute or `~/` path — relative paths resolve against the repo
root and will fail, see Shared mechanics):

```bash
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py \
  ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd
```

**Annotated output** (Mini Doomer, fresh mode, verified 2026-07-11; the types line changed 2026-10-19):

```
known types (35): ['active', 'attack', ... 'water', 'wind']
                       ^- 35 since 2026-10-19. Before that, a macOS .DS_Store leaked in and
                          the count was 36 (fixed: _populate_types_from_local skips dotfiles).
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
- `relative path` — computed with `relpath` against the clone root, the same way the local traversal
  did it from 2026-10-19 until local mode was archived on 2026-10-20 (now on branch `archive/local-update-stats`).
  Before that fix local mode computed it wrong (the removeprefix bug, autocroissant-failure-archaeology
  Entry 1). It matches the repo-relative paths `/update_stats` stores from the GitHub tree.
- **Icon font (2026-10-20).** parse_one.py and gap_trace.py pass
  `traverser._icons_from_local(clone_root)` to `PSDParser`, so the stoogedings vocabulary (Types/ and
  Types/Stars/ names, manifest aliases and codepoints) matches what `/update_stats` builds from the
  GitHub tree (`_icons_from_response`), and cards typed in the font parse as in production. These two
  `*_from_local` helpers stayed when local mode was archived, for these scripts; they live on `card_data.CardBuild` (the traverser's body since 2026-10-22; repo_traverser.py was deleted 2026-10-25). gap_trace.py calls `_process_layer`, whose signature did
  not change with the icon-font work.
- Field lines — `CardInfo.to_dict()` output. Empty fields are OMITTED (to_dict drops
  ability/stars/subtype/series/types/author/problems when empty and stats when invalid), so a
  fresh parse showing no `author:`/`series:` line is normal, not a bug.
- `ability` block between `===` fences — byte-for-byte what would be stored. Trailing spaces and
  blank runs are LOAD-BEARING (they are the injection signal; see the whitespace saga in
  autocroissant-failure-archaeology). Golden check: The Freezer's ability from
  `parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"` matches its stats.pkl
  entry character-for-character (verified 2026-07-11).
- `validator problems` — exactly what `/update_stats` would record in `card.problems`.

**`--with-db` vs fresh** (verified 2026-07-11): `PSDParser.parse` preserves `author`,
`series` and (since 2026-10-20) `subseries` from an existing DB entry when they are non-empty (psd_parser.py, the
"Preserve author and series" block in `parse()` — grep below). Default (fresh) parses against an
empty DB: no preservation, so Mini Doomer fresh shows neither field, while
`--with-db` adds `series: Kirby` and `author: Chestnut` and prints
`(loaded card_data/: 819 cards -- author/series preservation active)` (it read stats.pkl until 2026-10-25; 813 on
2026-07-11).
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

**Invocation** (a card that still injects from an icon layer; The Lich King uses a font token since 2026-10-22):

```bash
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py \
  "$HOME/Desktop/TTSCardMaker/Field/3 Stars/Nebulous_Precipice.psd"
```

**Annotated output** (Nebulous Precipice, measured 2026-10-23 at TTSCardMaker a7d1fe56 with `~/.venvs/ac-py314-fork`):

```
PSD 1200x1800; TYPE_REGION_RATIO=0.5 -> card_mid_y=900   <- the real imported constant
relative path: Field/3 Stars/Nebulous_Precipice.psd

[1] creature-types collected ABOVE midline: ['field']
        ^- these go to card.types; they are NOT injection candidates

[2] ability text layers found: 1
    at (x=26, y=1229): "'Friendly     gain +0.5 DEF and +1 SPD.\n(1GCD) Put a darkness counter ...'"
        ^- raw text + bbox of each text layer, pre-cleanup

[3] joined ability text with gaps marked (3+ spaces = injection slots):
Friendly<GAP:5>gain +0.5 DEF and +1 SPD.
...
        ^- every run of 3+ spaces becomes <GAP:n>. n=5 here: five literal spaces.
           No <GAP> where you expect one => the text layer lacks 3+ spaces there.

[4] type icons BELOW midline (candidate inline types): 1
    darkmatter      at (x=269, y=1207)
    prune threshold = max(last_y//3=402, card_mid_y=900)
    kept after prune: ['darkmatter']   dropped: []
        ^- icons with y above the threshold are dropped as "too high to be inline".
           threshold = max(last-icon-y // 3, card_mid_y); here max(402,900)=900.

[5] gap count = 1 vs kept types = 1   (match)
        ^- MISMATCH here is the classic wrong-injection signature: more types than
           gaps => leftovers get APPENDED to the last line; more gaps than types
           => some gaps stay as raw runs of spaces.

[6] FINAL ability text after injection + punctuation cleanup:
========================================
Friendly [darkmatter] gain +0.5 DEF and +1 SPD.
...
========================================
```

Nebulous Precipice is due to become a `[darkmatter:glow]` token once the glow font lands (owner, 2026-10-23); then
pick another icon-layer card (`psd_details.json`'s `icons` field `images` or `mixed`). The Lich King today is the
token case: `[2]` one layer at y=1767 reading "Minions inherit [undead].", `[3]` no GAP, `[4]` 0 icons, kept []
dropped [], `[5]` 0 vs 0 (match), `[6]` the same final text as before the batch.

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
`PSDParser._extract_from_layers` (commands/psd_parser.py:404 as of 2026-10-23): effect layers
(`EFFECT_LAYER_TAG`, " FX: ") set aside, hidden rule-page text skipped, hidden Ability layers read last, with
`RULEBOOK_DIR` and `stars_from_psd()` (the mirror gained the effect-layer skip and those two on 2026-10-23; before,
it fed "Ability FX: ..." pixel layers to `_process_layer` and hardcoded the folders). If that method changes shape,
update gap_trace to match; the one-line check is to compare against that method's body.

**Failure modes**: needs `config.py` and the clone at `~/Desktop/TTSCardMaker`; same
relative-path caveat as parse_one (no explicit warning branch here — a PSD outside the clone
just classifies from a `../...`-style path, i.e. garbage; keep inputs inside the clone).

**Questions it answers**: Which gap got which type and why? Why was an icon pruned? Why did a
type append to the last line? Is the failure in the text (gaps), the icons (position), or the
counts (mismatch)?

---

## 4. inspect_pickle.py — DB health and single-card lookup

**Purpose**: read-only view of the card data — the bot's `card_data/` by default, or with `--db` any `db/` folder, data
commit or historical pickles: a health summary, the list of cards with recorded problems, or one card in full detail
with its archived history.

**Invocations**:

```bash
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py              # summary
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --problems   # problem cards
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "the lich king"  # one card
# the data build: a db/ folder, or a data commit of a TTSCardMaker clone (fetch it first, as in Workflow 4)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --db ~/Documents/Tools/wt/scratch/carddata/db
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --db origin/data --git ~/Desktop/TTSCardMaker "Metal General"
```

The summary's first two lines read `stats     : N cards` and `old_stats : N cards with history, N archived
versions total` since 2026-10-23 (the file names left the labels). A card lookup also prints its `version` record
(start, cause, legacy) when it has one. Measured 2026-10-23 on a Mac rebuild from seed 077337a9 at TTSCardMaker
7b48db05: 818 cards, 355 names with 453 versions, 2 archived (1 backfilled).

**The last baseline on the pickles** (PICKLE a7c310e, the 2026-10-23 05:00 run; `python3` from the repo root; the data
branch's counts are in Workflow 4):

```
stats     : 818 cards
old_stats : 354 cards with history, 446 archived versions total
archived cards (PSD deleted): 2 (1 backfilled)
card_type distribution: creature 297, item 188, minion 91, debuff 52, field 49, aux item 48, rulebook 48, nme 26, buff 12, creatures 7
cards with recorded parse problems: 0
cards without author (orphans)    : 0
rulebook entries                  : 48
newest timestamp: Clot White @ 2026-10-22 20:56:26
entries with suspicious (non repo-relative) paths: 0
```

818 = 812 PSDs plus 6 alt-art duplicate rows (A Snack2/3, All Purpose Medical Kit2, Regeneration Potion2,
Watermelon2, NME Stock2). Item 188 and field 49 since 2026-07-11: new items, and White Palace moved to MDW/The
Morgue. 0 problem cards: 20_Creature_Types has a `wind` layer since TTSCardMaker 100511d3. **Expected next:** the
first /update_stats after the 2026-10-23 10:36 deploy recovers history from the bot's PICKLE commits (old_stats
446 versions / 354 names -> about 1,155 / 670; autocroissant-run-and-operate §7).

**Annotated summary output** (the 2026-07-11 snapshot, kept for its annotations; the labels then read `stats.pkl` /
`old_stats.pkl` and there was no archived-cards line):

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

`--problems` prints nothing on 2026-10-23 (0 stored problem cards): anything it lists is new. On 2026-07-11 it
listed exactly one card, `20 Creature Types [Auxiliary/Rulebook/20_Creature_Types.psd]: ['MISSPELT TYPE: tornado']`.

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
It exists to catch exactly the removeprefix corruption signature (paths starting `Users/...`; that
bug was fixed 2026-10-19, but keep the check: it is cheap and catches any future path regression).
Two caveats: (a) a legitimately NEW top-level folder in TTSCardMaker would false-positive —
update the tuple in the script if the card repo grows one; (b) 0 here does not prove every path
is *correct*, only that top folders look sane.

**Failure modes**: loads the repo root's `card_data/` (present once the bot synced; else pass `--db`) unless `--db`
names other data; data it can't read (no such folder, file or commit) → `ERROR loading the card data: ...`, exit 2.
Missing card name → `not found` + suggestions, exit 1.

**Questions it answers**: Is the DB healthy? What is stored for card X and when did it last
change? Which cards are flagged? Did old_stats balloon?

---

## 5. diff_stats.py — the numbers between two snapshots (pickles or the data build's JSON)

**Purpose**: diff two stats.pkl-format snapshots into added/removed/modified with per-field
counts. The measure of a data commit (the data gate, autocroissant-change-control) — it turns "the data looks fine"
into numbers.
Handles both modern CardInfo pickles and legacy plain-dict pickles (pre-366c8d9), so it can diff
across the Oct-2025 schema change. Since 2026-10-23 it also diffs history: `--history <old
old_stats.pkl> <new old_stats.pkl>` (or an `old_stats.pkl` beside each stats.pkl) adds a `-- history --`
section: names and versions on each side, versions added and dropped per name, starts moved (a
`version` record's `from`, on stored versions and current cards), archive tags changed, kept versions
whose data changed, and cards with another blob or parser version. Without history it prints a note.
Measured 2026-10-23 on the post-deploy rehearsal (main's cleanup vs work/history-fixes): `0 modified`,
`versions added: 1 ... dropped: 25`, `starts moved: 18`, `archive tags changed: 2`.

Both sides take any form of the card data (2026-10-23, `card_db.read_cards`): pickles, a `stats.json`, a `db/`
folder, or a data commit of `--git DIR`'s repo. The history matches one name's versions up before counting: the same
timestamp and blob; else the same blob (re-dated from git); else, where one side has no blob (the seed pass gives a
legacy version its commit's blob, and may re-date it to that blob's intro), the same timestamp, or the same data in
today's forms (`card_history.data_of`). So the data build's bookkeeping shows as `starts moved` and `kept versions
with other data`, not as versions added and dropped; a kept version's data is compared without its timestamp.

**Exit codes** (all verified 2026-07-11): `0` = no changes; `1` = changes found; `2` = usage or
load error. Do not pipe through grep/head when you consume the exit code.

**Relative paths resolve against your shell's cwd** (diff_stats.py does not chdir); a missing file or unknown commit
exits 2 with `ERROR loading the card data: <arg>: no such file or folder, nor a commit of <git dir> with db/stats.json`.

**Invocation A — the data gate** (a Card sync data commit vs the one before):

```bash
git -C ~/Desktop/TTSCardMaker fetch -q --depth 2 origin +data:refs/remotes/origin/data
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py origin/data~1 origin/data \
    --git ~/Desktop/TTSCardMaker
echo $?
```

exit 0 — no data change between the two commits (a run that only moved the posted cursor).
(Until 2026-10-25 Invocation A was the pre-push gate, HEAD's pickles vs the working copy.)

**Invocation B — what changed between two runs/snapshots** (any two pickles; the 2026-07-11 example below is
the two PICKLE commits newest then, 8753489 -> 284d13c; the newest on 2026-10-23 are a7c310e, 077337a and 3888392):

```bash
git show 8753489:stats.pkl > /tmp/stats_old.pkl && git show 284d13c:stats.pkl > /tmp/stats_new.pkl && \
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
| mass `path` changes in field counts | a path regression (the removeprefix local-path bug's signature, fixed 2026-10-19) — STOP, do not push | autocroissant-failure-archaeology Entry 1 |
| hundreds of modified cards whose only changed field is `timestamp` | pickles from a local-timestamp sweep (docs/BUGS_AND_IMPROVEMENTS.md B6; since 2026-10-20 only the archived traversal on branch `archive/local-update-stats` can produce them); old_stats.pkl will have grown by one archived copy per card — STOP, do not push | autocroissant-run-and-operate §7 |
| mass `type` → unknown (check with `--verbose`) | classification broke | impossibility-cards-reference for what classify should do |
| huge growth in archived old_stats versions | duplicate-archiving regression (cca0aaf → eb9aa84 revert; the 2026-10-22 15:41 run: +280) — UNLESS the run printed `History: N versions recovered from the bot's PICKLE commits ...`: the one-time recovery (2026-10-23) is expected. Compare with the rehearsal instead of stopping: "698 versions recovered (553 cards), 171 starts moved earlier; 64 cards gone before archiving archived; 42 type misreads merged; 12 impossible values set to unknown; 32 brought to today's series ...; 299 carried starts given their commit", old_stats 446 versions / 354 names -> about 1,155 / 670, plus about 219-236 re-parsed B2 PSDs; a second History line "103 carried starts given their commit" is known noise | the `-- history --` section's `versions added` (with `--history`, 2026-10-23); SESSION3_NOTES 07:5x and 10:3x |

**Failure modes**: fewer than two args → prints usage, exit 2; unreadable/missing/incompatible
data (a pickle holding anything but the card classes, a folder without card data, an unknown commit) →
`ERROR loading the card data: ...`, exit 2.

**Questions it answers**: What exactly did the last Card sync run change? How did two snapshots differ?
Did a parser change touch cards it shouldn't have?

---

## Standard workflows

### Workflow 1 — reviewing a Card sync data commit (the data gate)

```bash
git -C ~/Desktop/TTSCardMaker fetch -q --depth 2 origin +data:refs/remotes/origin/data
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py origin/data~1 origin/data \
    --git ~/Desktop/TTSCardMaker
echo $?
```

1. exit 0 → no card data change in that commit.
2. exit 1 → review each bucket against the data commit's per-card lines and the run's job summary:
   - `added`: only cards you know are new in TTSCardMaker.
   - `removed`: pair each with a possible `added` rename (an archive tag with `renamed_to`); a true unexpected removal =
     a build problem: pin the builder ref, dispatch `restore=<good data commit>` (DATA_BUILD_ACTION.md §13).
   - `modified`: read `field change counts` first. Mass `path` or mass `type` = red flag. Legit updates look like a
     handful of `ability`/`timestamp`/`types` changes on cards you know changed. Use `--verbose` on anything surprising.
   - `-- history --`: versions added should match the run's changed/edited events; mass `versions added` = a
     duplicate-archiving regression.
3. exit 2 → the gate did not run; fix the load error (an unfetched commit) first.
4. Gate refusals (G1-G6) never reach the branch: the run is red and its artifact holds the would-be db/ (diff it with
   `diff_stats.py origin/data <artifact>/db`).

Until 2026-10-25 this workflow was the pre-pickle-push gate (HEAD's pickles vs the working copy, before every /push).

### Workflow 2 — single-card investigation pipeline ("why did THIS card parse wrong?")

Run in this order; each step tells you whether to continue:

```bash
P=~/.venvs/ac-py314-fork/bin/python; C="$HOME/Desktop/TTSCardMaker/Field/3 Stars/Nebulous_Precipice.psd"
# 0. What is stored now, and its history? (optional but cheap)
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "Nebulous Precipice"

# 1. What is physically in the file? (layer names, visibility, bbox vs midline, raw text)
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py "$C" --text

# 2. How did injection decide? (gaps vs icons vs prune vs counts)
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py "$C"

# 3. Full parse + validator verdict (add --with-db to compare against stored fields)
$P .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py "$C" --with-db
```

Decision points: if step 1 shows the layer is missing/hidden/misnamed/on the wrong side of the
midline, the card FILE is the problem (fix in TTSCardMaker, not the bot). If step 1 looks right
but step 2 shows a gap/type mismatch or a bad prune, the heuristics are the problem — take it to
autocroissant-psd-extraction-campaign. If steps 1-2 look right but step 3 disagrees with the
stored entry, the stored entry is stale or was written by a buggy traversal — check history in
step 0 and the update mode used.

### Workflow 3 — DB health check cadence

Run `inspect_pickle.py` (summary) then `inspect_pickle.py --problems` (the bot's `card_data/`, or `--db origin/data
--git ~/Desktop/TTSCardMaker`):
- after a Card sync run that changed many cards (a full re-parse),
- alongside Workflow 1 when a data commit looks odd,
- whenever a query result looks wrong.

Compare against the 2026-10-23 baseline (a7c310e): 818 cards, 354 names / 446 versions in old_stats, 2 archived
cards, 0 problem cards, 0 orphans, 0 suspicious paths, newest = Clot White @ 2026-10-22 20:56:26 (2026-07-11: 813
cards, 218/223, 1 problem card, newest Anubisath Guardian @ 2026-06-18). Card counts legitimately grow as cards are
added, and old_stats grows once with the history recovery (above); problems/orphans/suspicious-paths should not.

### Workflow 4 — (history) the shadow's daily check (data branch vs the jetson's pickles, 2026-10-23)

Done: the shadow ended at the cutover (2026-10-24) and the pickles left main at step 10 (2026-10-25); to repeat it, take
`stats.pkl` from the last PICKLE commit (diff_stats.py's docstring shows how). Kept for its expected-numbers reading.
During the shadow (DATA_BUILD_ACTION.md §14 step 7), once a day after the jetson's /push:

```bash
# from this repo's root, at AutoCroissant main with the jetson's last PICKLE
git -C ~/Desktop/TTSCardMaker fetch -q --depth 1 origin +data:refs/remotes/origin/data   # a single-branch clone: name the ref
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py stats.pkl origin/data \
    --git ~/Desktop/TTSCardMaker
echo $?
```

Expect exit 1 with only the predicted refinements. Measured 2026-10-23 with AutoCroissant's PICKLE 077337a9 against
a Mac rebuild seeded from it at TTSCardMaker 7b48db05 (work/shadow-prep with work/history-fixes 93e3458; the shadow's
first data commit should look the same):

```
old: 818 cards   new: 818 cards
added: 0   removed: 0   modified: 796
  field change counts: {'created': 770, 'timestamp': 305, 'ability': 8, 'display_name': 6, 'problem': 3, 'stars': 2, 'types': 1}
-- history --
old: 364 names 475 versions   new: 355 names 453 versions
versions added: 14 (14 names)   dropped: 36 (33 names)
starts moved: 71   archive tags changed: 2   kept versions with other data: 52   cards with another blob or parser: 281
```

- `created` 770: the data build dates every card from `site/Card Creation Dates.json`; the jetson does too once it
  runs the created-dates code, so this count falls to the cards it hasn't dated.
- `timestamp` 305: 281 PSDs the seed's PICKLE had not read yet plus 24 cards re-dated to their blob's intro (28 re-dated
  by the seed pass, some among the 281). Against a current jetson only the re-dated ones stay (about 30).
- `ability`, `display_name`, `problem`, `stars`, `types`: PSD changes the jetson had not read when it pickled (plus
  Shadow Duelist and 00 Table of Contents until both writers re-read them with the current parser).
- History: dropped 36 = the seed pass's duplicate versions (the build log's `Duplicate versions dropped: 36`);
  added 14 = this build's new versions (the summary's 9 changed and 5 edited); starts moved = re-dated versions
  (5 of them legacy versions the seed pass also gave a blob, such as Shell Shield 1739181026 -> 1699228331) and
  carried starts; other data = versions normalized (series filled or moved under a parent, retired type names);
  archive tags = ZOEY!!!!! and White Palace dated from git.
- Not shown, and also expected: `version` records (the data build gives every card one; `starts moved` shows
  where it moved a start), `parsed_values` on about 600 cards (the series carried into Card Edits.json), the
  duplicates' copies of their original's `version`.

Anything else (a removed card, a mass `path` or `type` change, a field the jetson and the build read differently)
stops the shadow until it is explained. `inspect_pickle.py --db origin/data --git ~/Desktop/TTSCardMaker "Name"` shows
one card's record and versions on the data side.

---

## Adding a new diagnostic script

Conventions (match the existing five — read one before writing):

1. `#!/usr/bin/env python3` and a module docstring that states usage, what it prints, and the
   read-only guarantee. Scripts print `__doc__` on missing args.
2. If it imports bot code that reads repo-relative paths (the pattern parse_one/gap_trace follow; a script that only
   reads the card data should import `commands.card_db.read_cards` with no chdir, like inspect_pickle and diff_stats):
   ```python
   REPO_ROOT = Path(__file__).resolve().parents[4]   # scripts/ is 4 levels below repo root
   sys.path.insert(0, str(REPO_ROOT))
   os.chdir(REPO_ROOT)  # card_data/ and the clone paths in global_config are repo-relative
   ```
   (no token line: `print_token_status` runs only in main.py since 2026-10-20).
3. READ-ONLY, always: never call `update_stats()`, anything network-mutating, and never
   write card data. If a script needs scratch output, write to `~/Documents/Tools/wt/scratch/` (reboots wipe /tmp) or a
   scratchpad, never the repo.
4. Prefer re-driving REAL parser methods over copying their logic; where orchestration must be
   mirrored, add a MAINTENANCE NOTE naming the mirrored function (see gap_trace.py's docstring).
5. `def main() -> int` + `sys.exit(main())`; document exit codes if callers branch on them.
6. THEN update this SKILL.md: add a routing-table row, a per-tool section (purpose, invocation,
   annotated output, failure modes, questions answered), and a provenance re-run line. A script
   without a section here does not exist as far as the skill library is concerned.

## Measurement tools beyond the five (2026-10-20)

Not part of the five, and not read-only in the same sense (most need a sandbox copy), but kept:

| Tool | Where | What it measures |
|---|---|---|
| Offline test suite | `tests/` (`~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests`) | Behaviour contracts, parser goldens; owner: autocroissant-validation-and-qa row 1c |
| Whole-corpus parser gate | `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/parse_all_ns.py <out.json> <sandbox root>` + `compare.py a.json b.json` (table below; venv interpreter) | Every field of all 813 card PSDs, old vs new code (validation-and-qa §4.1 method A) |
| Profiling and benches | `docs/Finished/optimization-pass-2026-10-20/bench/` (expects the `/tmp/acprof/sb` sandbox: `git archive HEAD` plus copies of config.py and, then, the pickles) and `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/*_ns.py` (sandbox root as first argument) | Queue hops, search timings, import time and RSS; results in `docs/Finished/optimization-pass-2026-10-20/profile.md` and `network-speed-results.md` on that branch |
| Skill line-ref refresh | `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/refs6.py` + `apply6.py`; `grepcheck.py` runs every quoted grep | Stale `file:line` references after code moves (follows code across files) |

### Optimization-pass scripts

`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/` (moved byte-identical from `~/Documents/ac-optimization/`
on 2026-10-20); the invocations below are relative to that folder. ROOT is a sandbox copy of the repo (`git archive HEAD` plus
config.py and a `card_data/` copy, under /tmp); "live" means read-only GitHub GETs with the real token. parse_all_ns.py, the `_ns`
scripts and search_ab.py import the bot (psd_tools, discord): run them with `~/.venvs/ac-py314-psd125/bin/python`, not the
system `python3`; compare.py and the line-ref tools are stdlib.

| Script | Invocation | What it does |
|---|---|---|
| parse_all_ns.py | `parse_all_ns.py <out.json> [ROOT]` (ROOT defaults to `/tmp/psdreg/ac`) | Parses every card PSD of the `~/Desktop/TTSCardMaker` clone with ROOT's parser (16 processes); dumps every field plus the validator's list, `timestamp`/`author` dropped |
| compare.py | `compare.py a.json b.json` | Field-by-field differences between two parse_all_ns dumps; expect `813 cards, 0 differences` |
| import_ns.py | `import_ns.py ROOT [N]` | `import main` wall time, RSS and heavy modules loaded, best of N fresh processes (default 5) |
| startup_ns.py | `startup_ns.py ROOT` | Live: tree requests at startup (init_query and the type-emoji listing in parallel), then one /sync_emojis listing |
| search_ns.py | `search_ns.py ROOT OUT.json` | Search timings and every result on ROOT (its `card_data/` and the `~/Desktop/TTSCardMaker` clone); diff two OUTs for identity |
| search_ab.py | `search_ab.py ROOT` | Interleaved A/B in one process: old fuzzy step / name lookup vs the new ones, same queries, results asserted equal; `exec`s search_ns.py beside it |
| refs5.py, final.py, table.py, to_md.py / apply.py | no arguments, in that order (each reads the previous JSON) | Skill `file:line` audit against HEAD: anchor each ref to the commit where its symbol sat on that line (refs5.json), decide with overrides.json (final.json), list the edits (table.json), render them as markdown or apply them to the skills |
| refs6.py, apply6.py `[--dry]` | no arguments | Refresh against the WORKING TREE, following code moved across files (the psd_analyzer split): final.json → refs6.json, applied with overrides6.json; refs5_lib.py holds the shared regexes. Only the hand-written overrides*.json are kept; refs5/final/table/refs6.json are regenerated by the run |
| grepcheck.py | no arguments | Runs every read-only grep quoted in the skills; lists the ones that now print nothing |
| provcheck.py | no arguments | For `grep -n ...` commands followed by expected line numbers, flags numbers the grep does not hit |

The line-ref tools `chdir` to `/Users/michaelsrouji/Desktop/AutoCroissant` and keep their JSON under the old
`~/Documents/ac-optimization/docs_refaudit`: a `D` / `OUT` constant at the top of apply.py, apply6.py, to_md.py and refs6.py, but
inline in each `open()` of refs5.py (1 line), final.py (3) and table.py (2); point all of them at this folder before re-running.
The JSON files here are their last outputs; overrides.json and overrides6.json are hand-written inputs.

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
- **Whole-corpus parser regression checks** (old parser vs new parser on every PSD; sandboxed local
  sweep) → **autocroissant-validation-and-qa** §4.1 (method A is scripted: `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/parse_all_ns.py` + `compare.py`, table above).
- **First-principles analysis beyond these five tools** (API budgets, queue timing, designing a
  new experiment) → **autocroissant-analysis-toolkit**.

## Provenance and maintenance

All facts and golden outputs verified 2026-07-11 by running the scripts on this Mac (Python
3.10.20, clone at `~/Desktop/TTSCardMaker`). Counts (813/218/223/1/0), the newest-card line, and
the two-snapshot diff figures drift with every PICKLE commit — re-run before quoting.
Updated 2026-10-19 by code reading only (the scripts were not re-run): the types count, the
local-path fix and the new red-flag row; 2026-10-20: the scripts' icon-vocabulary change. On 2026-10-19 the Mac's
`python3` (Homebrew 3.14.8) has no psd-tools, so the scripts need a venv first
(autocroissant-build-and-env §1). On 2026-10-20 parse_one.py and gap_trace.py gained the icon
vocabulary and parse_one.py's note on the removeprefix bug was rewritten; diff_stats.py's docstring
(line 21) still names "the removeprefix local-path bug" as the meaning of mass path changes, which
reads as history. After that day's psd_analyzer split, parse_one.py, gap_trace.py and inspect_pickle.py were
re-run on Mini Doomer and the live pickles (imports through the `commands.psd_analyzer` facade): same output.

Re-verification one-liners (from repo root):

```bash
S=.claude/skills/autocroissant-diagnostics-and-tooling/scripts
ls .claude/skills/autocroissant-diagnostics-and-tooling/scripts/
# expect: diff_stats.py dump_psd_layers.py gap_trace.py inspect_pickle.py parse_one.py (plus optimization/, 2026-10-20)

P=~/.venvs/ac-py314-fork/bin/python                 # psd-tools + pandas; plain python3 runs inspect_pickle and diff_stats
# golden: DB summary (2026-10-23 at a7c310e: 818 cards, 354/446 old_stats, 2 archived, 0 problems, 0 orphans, 0 suspicious)
python3 $S/inspect_pickle.py

# golden: Mini Doomer (minion, stars 2, ['wind','defensive_minion','light'], 5/2/5/5, problems NONE)
$P $S/parse_one.py ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd

# golden: The Freezer ability text matches stats.pkl character-for-character
$P $S/parse_one.py ~/Desktop/TTSCardMaker/"Field/1 Stars/The_Freezer.psd"

# golden: Nebulous Precipice trace (<GAP:5>, darkmatter @ x=269,y=1207, prune max(402,900)=900, 1 vs 1 match,
#   "Friendly [darkmatter] gain +0.5 DEF and +1 SPD.")
$P $S/gap_trace.py "$HOME/Desktop/TTSCardMaker/Field/3 Stars/Nebulous_Precipice.psd"
# token case: The Lich King (no GAP, 0 icons, 0 vs 0 match, "Minions inherit [undead].")
$P $S/gap_trace.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"

# golden: clean-tree gate is 0/0/0 with exit 0
H=~/Documents/Tools/wt/scratch/head; mkdir -p $H; git show HEAD:stats.pkl > $H/stats.pkl && python3 $S/diff_stats.py $H/stats.pkl stats.pkl; echo $?

# exit codes: no-args is 2; two differing snapshots is 1
python3 $S/diff_stats.py >/dev/null 2>&1; echo $?

# dump header (1200 x 2400, mid-y 1200)
$P $S/dump_psd_layers.py "$HOME/Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd" | head -2
```

Cited code locations (line numbers drift; re-find with these):

```bash
grep -n "TYPE_REGION_RATIO = " commands/psd_parser.py               # 0.5 (psd_parser.py since the 2026-10-20 split)
grep -n "_extract_from_layers" commands/psd_parser.py               # gap_trace's mirrored method (404)
grep -n "EFFECT_LAYER_TAG\|stars_from_psd\|RULEBOOK_DIR" .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py   # the mirror's skip and folders
grep -n "def _process_layer" commands/psd_parser.py                 # gap_trace calls it; signature unchanged by the 2026-10-19 icon work
grep -n "Preserve author and series" commands/psd_parser.py         # parse_one --with-db semantics
git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py   # the 2026-10-19 local-path fix; local mode archived 2026-10-20
grep -n "_icons_from_local" .claude/skills/autocroissant-diagnostics-and-tooling/scripts/*.py   # parse_one.py and gap_trace.py load the icon vocabulary (2026-10-20)
grep -n "LOCAL_DIR_LOC" global_config.py                           # "~/Desktop/TTSCardMaker" (line 5)
grep -n "def to_dict" commands/card_models.py                      # field-omission behavior in output
```

If `_extract_from_layers` changes shape, update gap_trace.py (its docstring says so). If
TTSCardMaker grows a new top-level folder, update inspect_pickle.py's known-top-folders tuple.
If a script is added, this document must gain a routing row, a section, and a provenance line.

2026-10-20 (later): the optimization-pass scripts and their JSON moved from `~/Documents/ac-optimization/` into
`scripts/optimization/` (byte-identical, described above); the reports and `bench/` went to `docs/Finished/optimization-pass-2026-10-20/`.

2026-10-22 (data build): RepositoryTraverser's body moved to commands/card_data.py (`CardBuild`); the `*_from_local` helpers the scripts call are there, unchanged.

2026-10-23 (work/shadow-prep): `diff_stats.py` and `inspect_pickle.py` read the data build's `db/` JSON and data
commits (`card_db.read_cards`, `--git`), need no `config.py`, and diff_stats matches versions the seed pass dated
(also a legacy version given a blob and re-dated, by its data);
Workflow 4 (the shadow's daily check) and its numbers were measured on a Mac rebuild from seed 077337a9;
`tests/test_diagnostics.py` runs both scripts on pickles, a db/ folder, a stats.json and a data commit.

2026-10-23 (docs drift): which interpreter each script needs (`$P` = the fork venv for the three PSD readers); only
parse_one and gap_trace chdir (inspect_pickle and diff_stats import card_db, no chdir, no psd_analyzer); diff_stats's
load error quoted as it reads now; the Lich King examples re-measured (token `[undead]`, the Undead icon hidden since
TTSCardMaker 1c24dd48) and the gap walk-through moved to Nebulous Precipice; gap_trace.py's mirror gained the
effect-layer skip, `RULEBOOK_DIR` and `stars_from_psd()` (outputs unchanged on Nebulous Precipice, The Lich King,
Gasoline, Miracle Matter, Dark Daroach and PoD02Black Bloon), its docstring names `_process_abilities`; a dated
2026-10-23 baseline (818 / 354 / 446 / 2 archived / 0 problems / Clot White); the one-time recovery growth in the red-flag
row and Workflow 1; the gate snapshots go to `~/Documents/Tools/wt/scratch/head/` like the script's docstring.

2026-10-25 (step 10, work/pickles): the pickles left the repo: inspect_pickle.py reads `card_data/` by default, parse_one.py's `--with-db` loads it, diff_stats.py's examples are data commits (the last PICKLE by git show); Workflow 1 is the data commit's review, Workflow 4 is history; commits_ns, e2e_update_ns, etag_handoff_ns and race_ns deleted with the legacy /update_stats.
