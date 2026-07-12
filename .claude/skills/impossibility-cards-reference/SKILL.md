---
name: impossibility-cards-reference
description: Domain reference for the Impossibility Simulator card game AS ENCODED in AutoCroissant and the TTSCardMaker repo. Load whenever a task mentions cards, PSDs, PSD layers, ability text, type injection or [type] tokens, stats extraction (hp/def/atk/spd), stars/Warpstars, folder structure or card classification, CardInfo/CardStats, the stats.pkl / old_stats.pkl / aliases.pkl schemas, query syntax or the query DSL (/query, /query_ability, /query_rulebook; the | & ! "exact" hp>5 type==creature type~x operators), rulebook pages, aliases or ambiguous card names, validation problems (UNKNOWN TYPE, ABILITY TEXT NOT FOUND, STATS TOO HIGH, MISSPELT TYPE), the exclusion lists, or the creatures-vs-creature anomaly. Contains the folder-to-classification table, PSD layer semantics with an annotated real layer dump, the precise type-injection algorithm with constants and a worked trace, validation rules with both exclusion lists spelled out, pickle schemas with measured counts, and the full query DSL with gotchas. This is the "what the data means" pack — not an operating runbook (run-and-operate), not a bug-fixing campaign (psd-extraction-campaign), not script usage docs (diagnostics-and-tooling).
---

# Impossibility Simulator: the card data model, as encoded here

All counts and line numbers verified 2026-07-11. Line numbers drift; every volatile fact has a
re-verification one-liner in "Provenance and maintenance" at the bottom. Commands assume
cwd = `/Users/michaelsrouji/Desktop/AutoCroissant` (this repo's root).

This skill explains what a mid-level newcomer cannot infer from the code alone: what the folders
mean, what the PSD layers mean, how ability text and stats are actually extracted, what the pickles
contain, and how the query language behaves. It documents the system as it IS, including its
anomalies — do not "fix" anything you learn about here without reading
`autocroissant-change-control` first.

## 1. What the game and the two repos are

"Impossibility Simulator" is a homemade card game. Its data lives in a separate GitHub repo,
`MichaelJSr/TTSCardMaker`, whose README opens: "A Tabletop-Simulator port of a card game created by
https://github.com/JasonBechdolt and maintained by friends since middle school." Each card is a
Photoshop file (`.psd`, the editable source of truth) with a rendered `.png` sibling (the image the
bot posts in Discord, served from `raw.githubusercontent.com`). TTSCardMaker is cloned at
`~/Desktop/TTSCardMaker` — that path is the documented convention, set as `LOCAL_DIR_LOC` in
`global_config.py:4`. This bot (AutoCroissant) READS TTSCardMaker — over the GitHub API or from the
local clone — parses the PSDs into a stats database (`stats.pkl`), and answers queries against it.
The bot never writes to TTSCardMaker.

Scale, measured 2026-07-11: 904 PSDs and 1003 PNGs in the clone; 813 entries in `stats.pkl`.
The books balance exactly: 904 − 90 (`MDW/`, excluded) − 1 (`Auxiliary/Markers/`, excluded) = 813
non-excluded PSDs, all with unique names.

**Card identity**: a card's name is its PSD filename with underscores turned into spaces
(`The_Freezer.psd` → "The Freezer"; `commands/psd_analyzer.py:361`). Names are the keys of
`stats.pkl` and the index of the query dataframes. Two PSDs with the same basename would silently
share one entry — today there are none outside the excluded folders.

## 2. Folder taxonomy → card classification

`CardClassifier.classify` (`commands/psd_analyzer.py:256-340`) derives a card's type and several
fields purely from its repo-relative path. Every row below was verified against the code and the
live folder tree.

| Path pattern | `type` | Extra fields | Stars come from |
|---|---|---|---|
| `MDW/...` | `MDW`, ability forced `None` | — | — (excluded from traversal anyway, see below) |
| `Field/<N Stars>/x.psd` | `field` | — | leaf folder name |
| `Items/<Subtype>/<N Stars>/x.psd` | `item` | `subtype` = `<Subtype>`.lower() | leaf folder name |
| `Creatures/<Series>/<N Stars>/x.psd` | `creature` | `series` = `<Series>`.lower(); hp/def/atk/spd seeded −1 | leaf folder name |
| `N.M.E/x.psd` | `nme` | — | PSD star layers |
| `Auxiliary/Minions/x.psd` | `minion` | hp/def/atk/spd seeded −1 | PSD star layers |
| `Auxiliary/Items/x.psd` | `aux item` | — | PSD star layers |
| `Auxiliary/Status Effects/Debuffs/<N Stars>/x.psd` | `debuff` | — | leaf folder name |
| `Auxiliary/Status Effects/Buffs/x.psd` | `buff` | — | none |
| `Auxiliary/<AnythingElse>/x.psd` (fall-through) | literal leaf folder, lowercased | — | none |
| anything else | `unknown` | — | none |

Observed live subtrees (2026-07-11): `Items/` subtypes = Active, Attack, Counter, Deploy, Equip,
Heal, Infliction, Negation, Support; `Field/` and `Items/*/` have `1 Stars`..`4 Stars`; Creatures
series (24 of them: BTD, Kirby, Souls, World Of Warcraft, ...) have star folders in roughly the
`2 Stars`..`5 Stars` range; `Auxiliary/` contains Creatures, Items, Markers, Minions, Rulebook,
Status Effects.

**The stars parse** (psd_analyzer.py:285, 290, 309, 335) is:

```python
"stars": int(folders[len(folders) - 1].split()[0])
```

i.e. the integer prefix of the LEAF folder name ("3 Stars" → 3). Consequence: for Field, Items,
Creatures, and Debuffs, the leaf folder MUST start with an integer or `classify` raises
`ValueError`. (The `x[len(x)-1]` instead of `x[-1]` is the repo's Cython negative-indexing ban —
see `autocroissant-build-and-env`; do not "clean it up".)

**Stars from the PSD instead**: for `Auxiliary/Items`, `Auxiliary/Minions`, and `N.M.E` (exactly
these three, matched as substrings of the relative path — psd_analyzer.py:408-411), stars are
COUNTED from the PSD: each visible pixel-bearing layer whose ancestry (up to 3 levels) contains
"stars" adds 1 (`_is_star_layer`, psd_analyzer.py:517-527; assignment at 425-426). In practice
these are "Warpstar" smart-object layers under a "Stars" group — Mini Doomer has two, hence
stars=2.

### The creatures-vs-creature anomaly (do not fix silently)

`Auxiliary/Creatures/` has no dedicated branch in `_classify_auxiliary`, so its 7 cards get the
fall-through type — the literal leaf folder name, lowercased: **"creatures" (plural)**. The 297
cards under top-level `Creatures/` get **"creature" (singular)**. A `type==creature` query MISSES
the 7. They are, as of 2026-07-11: Azamoth the Possessed Vessel, Igor Roballtowski, Mettaton EX,
Mettaton NEO, Militron (No Armor), Morpho Knight, Shadow Queen. Workaround: `type~creature`
(contains) matches both. The same fall-through gives `Auxiliary/Rulebook/` pages type
**"rulebook"** (48 entries). Also note the 7 "creatures" cards skip the creature-specific handling:
no series, no −1 stat seeding, so the stat validators (section 5) never fire for them.

### Excluded folders

`EXCLUDE_FOLDERS = ["Markers", "MDW"]` (psd_analyzer.py:31) is checked against every path segment
during BOTH remote and local stats traversal (psd_analyzer.py:922, 994). `MDW/` (90 PSDs) is
Photoshop templates — card backs, `.aco`/`.atn` assets; `Auxiliary/Markers/` (1 PSD) is a board
marker. Neither reaches `stats.pkl` (verified: 0 MDW-typed entries). The `MDW` classifier branch is
therefore dead in normal traversal; it only matters if `classify` is called on such a path directly
(e.g. by `parse_one.py`).

Distribution in `stats.pkl` (2026-07-11, from `inspect_pickle.py`): creature 297, item 182,
minion 91, debuff 52, field 50, aux item 48, rulebook 48, nme 26, buff 12, creatures 7 — total 813.

## 3. PSD anatomy — what the parser actually reads

This is the part newcomers lack. A card PSD is a layer TREE; the parser walks
`psd.descendants()` and dispatches on layer kind, name, visibility, and position
(`PSDParser._process_layer`, psd_analyzer.py:428-456). Annotated excerpt of a real dump
(`dump_psd_layers.py`, reproduced 2026-07-11):

```
PSD: ~/Desktop/TTSCardMaker/Auxiliary/Minions/Mini_Doomer.psd
size: 1200 x 2400  (mid-y at ratio 0.5 = 1200)
layer (indent = depth)              kind         vis   pix   bbox
Stars                               group        True  False (936, 12, 1173, 126)
  Warpstar                          smartobject  True  True  (936, 14, 1060, 126)   <- +1 star
  Warpstar                          smartobject  True  True  (1049, 12, 1173, 124)  <- +1 star => stars=2
Creature Types                      group        True  False (0, 2, 356, 356)
  wind                              smartobject  True  True  (214, 208, 356, 356)   <- y=208 < 1200 => card.types
  Defensive_Minion                  smartobject  True  True  (0, 223, 170, 341)     <- ditto
  Light                             smartobject  True  True  (51, 2, 269, 219)      <- ditto => types=['wind','defensive_minion','light']
Stats Bars / Darks / Hp Dark        group
  0..9                              exposure     ...   False  <- digit-named layers; ONLY "5" is visible => hp=5
Text                                group
  Ability                           type         True  True  (40, 1815, 1144, 2245) <- kind "type" = TEXT layer
    TEXT: "'Has stats equal to the base stats -2 of the creature that
            summoned it. \x03(BOT) Die.\r'"
  Name                              type         True  True   <- text layer NOT named "ability": ignored
```

`parse_one.py` on this file (golden output): type=minion, stars=2,
types=['wind','defensive_minion','light'], hp=5 def=2 atk=5 spd=5, ability ending "(BOT) Die.",
problems NONE.

### 3.1 The naming trap: `layer.kind == "type"` means TEXT

In psd-tools, "type" is the layer KIND for text layers (typography), and has nothing to do with
card types. `psd_analyzer.py:434` dispatches text layers on exactly this. Card-type ICONS are
pixel/smart-object layers matched by NAME (3.3). Misread this and nothing else in the parser makes
sense.

### 3.2 Ability text

A text layer contributes ability text iff its name (lowercased) is `"ability"` — OR the card is a
rulebook page (`"Rulebook" in relative_path`), in which case ALL text layers contribute
(psd_analyzer.py:469, 407). The raw string is `str(layer.engine_dict["Editor"]["Text"])` — a
repr-like string wrapped in quotes, with literal escape sequences. Cleanup (psd_analyzer.py:460-467):

| Raw sequence | Becomes | Why |
|---|---|---|
| `\r` | `\n` | Photoshop line breaks are CR |
| `\n` | `\n` | normalize |
| `\t` | one space | tabs |
| `\x03` | `\n` | ETX control char Photoshop emits for some breaks |
| `\ufeff` (BOM char) | removed | byte-order mark (BOM) |
| trailing whitespace | `.rstrip()` | |

Multiple ability text layers (common on rulebook pages) are sorted by position (3.5's row sort) and
joined with `\n`; the wrapping `'`/`"` quotes are stripped during the join and at the end
(psd_analyzer.py:529-539). If NO ability text layer exists at all, the parse-time problem
`NO ABILITY LAYER` is recorded (psd_analyzer.py:422).

### 3.3 Card-type icons and the midline

`all_types` is the vocabulary of icon names, populated from TTSCardMaker's `Types/` folder
(excluding `Types/Stars/`), lowercased file stems: 'fire', 'ice', 'undead', 'defensive_minion', ...
Remote traversal yields 35 (psd_analyzer.py:875-881); LOCAL traversal (psd_analyzer.py:883-890)
walks the directory and leaks macOS artifacts — verified: 36 entries including `'.ds_store'`.
Harmless today (no layer is named .DS_Store) but visible in `parse_one.py` output; ignore it, and
ignore the "Git token found..." lines the import chain prints.

A visible, pixel-bearing layer whose lowercased name is in `all_types` is a type icon
(psd_analyzer.py:442-447). Position decides its meaning, using
`card_mid_y = int(psd.height * TYPE_REGION_RATIO)` with `TYPE_REGION_RATIO = 0.5`
(psd_analyzer.py:30, 405):

- bbox top `y < card_mid_y` (upper half, the corner badges) → appended to `card.types`
  (in raw traversal order, NOT sorted);
- `y >= card_mid_y` (lower half, sitting inside the ability paragraph) → candidate for INLINE
  INJECTION into the ability text (section 4).

### 3.4 Stats (hp/def/atk/spd)

Two-stage name matching (psd_analyzer.py:473-515):

1. A layer is a stat layer if any ancestor (up to 3 levels) has "dark" or "bars" in its name
   (`Stats Bars > Darks > Hp Dark` qualifies).
2. It contributes iff its name `isdigit()`; the nearest 2 ancestor names are matched for the
   substrings `hp` / `def` / `atk` / `spd` (first hit wins).

Semantics: a digit layer FOUND under e.g. "Hp Dark" marks hp as present, whether visible or not.
All VISIBLE digit layers for a stat are SUMMED (`tracker.value += int(layer.name)`,
psd_analyzer.py:511-514). If a stat was found but every digit layer is hidden, the value defaults
to **10** (psd_analyzer.py:553) — that is how maxed stats are encoded in the template. Never found
→ stays −1 (only seeded for creatures/minions), which the validator reports. In Mini Doomer:
"5" visible under Hp Dark → hp=5; visibility alone matters (the digit layers are `exposure` layers
with no pixels).

### 3.5 Position sorting (shared by text layers and icons)

`_sort_by_position` (psd_analyzer.py:556-583): sort by bbox top `y`; group into visual rows where
`|y − first_in_row.y| <= 40` px (`row_threshold=40`); sort each row left-to-right by `x`; flatten.
This is reading order — top-to-bottom, then left-to-right — and it determines both multi-text-layer
join order and type-icon injection order.

## 4. The type-injection algorithm (the hardest live problem)

Card authors drop type ICONS inside the ability paragraph in Photoshop; the text layer just has a
run of spaces where each icon sits. Extraction must splice `[typename]` tokens back into those
gaps. This is the owner-stated hardest live problem; the spec below is exact
(`_inject_type_names`, psd_analyzer.py:593-637, plus callers at 529-539).

Constants:

| Constant | Value | Where |
|---|---|---|
| `TYPE_REGION_RATIO` | 0.5 (midline) | psd_analyzer.py:30 |
| gap pattern | `\s{3,}` (3+ whitespace chars) | psd_analyzer.py:348 |
| row grouping threshold | 40 px | psd_analyzer.py:557 |
| prune rule | keep icon iff `y >= max(last_sorted_icon_y // 3, card_mid_y)` | psd_analyzer.py:590 |
| punctuation cleanup | `sub(r'\s+([:;,\.\?!])', r'\1')` after injection | psd_analyzer.py:538 |

Algorithm:

1. Build the joined, cleaned ability text (3.2).
2. Take the below-midline icons, sort by position (3.5), prune with the prune rule above.
   Arithmetic note, verified: with ratio 0.5, every candidate already has `y >= card_mid_y` and
   `last_y // 3 < card_mid_y` for any on-canvas layer, so the prune is currently a NO-OP; it bit
   when the region rule was "top 400px" (changed 2026-01-30 — history in
   autocroissant-failure-archaeology). Keep it in mind if `TYPE_REGION_RATIO` ever changes.
3. Split the ability into lines (each rstripped of `' \n` then `\n` re-appended). For each line,
   in order, while unconsumed types remain: find all `\s{3,}` gaps; replace gap i with
   ` [next_type] ` (text before the gap rstripped, after lstripped, line lstripped). Lines without
   gaps pass through unchanged. Types are consumed strictly in sorted (reading) order.
4. LEFTOVER types (more icons than gaps) are appended to the LAST line as ` [t1] [t2] ...`
   (psd_analyzer.py:632-635). More gaps than icons: extra gaps stay as-is as literal spaces.
5. Punctuation cleanup: whitespace before `: ; , . ? !` is deleted. This glues `[ice] :` into
   `[ice]:` — and, derived from the code, it also silently swallows an UNFILLED gap that sits
   directly before punctuation, which can mask a missed icon.

**Worked example — The Lich King** (reproduced 2026-07-11 via
`python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py ~/"Desktop/TTSCardMaker/Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"`):
PSD 1200x2400, card_mid_y=1200. Above midline: icons
`ice`, `undead` → `card.types`. One ability text layer at (x=23, y=1773) whose last line is
`"Minions inherit<GAP:7>."` (a 7-space gap). One below-midline icon: `undead` at (x=734, y=2239).
Prune threshold = max(2239//3=746, 1200) = 1200 → kept. 1 gap vs 1 type → match. Final line:
`"Minions inherit [undead]."` The Freezer (golden) shows the multi-gap case — three icons injected
across two lines, producing exactly:
`"For friendly creatures:\n[water] Turns into [ice]. [ice]: +1 DEF\n(1PGT) Choose a friendly creature to turn into a [ice] and lower its speed by 1."`

(The golden values quoted in sections 3-4 — Mini Doomer, The Freezer, The Lich King — are
snapshots of 2026-07-11; canonical current expected values live in autocroissant-validation-and-qa
§3. If one stops matching, check there before assuming a regression.)

**Load-bearing invariant: gap whitespace IS the signal.** Never collapse runs of spaces in ability
text, in parser code or in stored data — a 2026-02-08 eight-commit saga (7 parser commits + a
mid-saga PICKLE) added `\s{2,}` collapsing and had to remove it entirely (3bbaa2b "Fix the issue
of collapsing spaces" + 4bcee6b). Full story: autocroissant-failure-archaeology Entry 2.

## 5. Validation semantics (`CardValidator`, psd_analyzer.py:640-733)

Problems come in two kinds, and only one kind is persisted — a common newcomer misread:

- **Parse-time problems** are appended to `card.problems` during parsing and stored in the pickle:
  `NO ABILITY LAYER` (psd_analyzer.py:422) and `MISSPELT TYPE: <name>` (psd_analyzer.py:455-456,
  fired by a visible pixel layer named in `MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']`,
  line 42).
- **Validate-time problems** are computed by `CardValidator.validate` during `/update_stats`,
  REPORTED to Discord, and NOT written back to the card.

So the `problem` field in stats.pkl only ever holds parse-time strings. As of 2026-07-11 exactly
one card has one: rulebook page "20 Creature Types" with `MISSPELT TYPE: tornado`.

| Problem string | Fires when | Source |
|---|---|---|
| `UNKNOWN TYPE` | `card_type == "unknown"` (path matched no classifier row) | :683-685 |
| `ABILITY TEXT NOT FOUND` | type is not unknown/MDW, ability empty, name not in ABILITY_EXCLUSIONS | :687-692 |
| `HP NOT FOUND` / `DEF NOT FOUND` | creature or minion with that stat still −1 | :713-717 |
| `ATK NOT FOUND` / `SPD NOT FOUND` | same, UNLESS `"active"` is in `card.types` (active creatures have no atk/spd) | :719-726 |
| `STATS TOO HIGH` | any found stat > 10 and name not in EXCESSIVE_STAT_EXCLUSIONS | :728-731 |
| `NO ABILITY LAYER` | no ability text layer found at parse time; validate() DELETES it again for names in ABILITY_EXCLUSIONS | :422, :700-702 |
| `MISSPELT TYPE: <t>` | visible layer named undread/tornado/error | :455-456 |

**The exclusion lists are the "perfect extraction" debt register** — per-card allowances the owner
wants to eliminate (see autocroissant-research-frontier; the campaign to shrink them is
autocroissant-psd-extraction-campaign). Verbatim, psd_analyzer.py:642-668:

- `EXCESSIVE_STAT_EXCLUSIONS` (5, stats legitimately above 10): Royal Eradicator Main Cannon, Pix,
  Tainted Lazarus, Sonic, Twin Emperors.
- `ABILITY_EXCLUSIONS` (18, allowed to have no ability text): Crystal Bits, Bugzzy, Electro Probe,
  Galacta Warrior, God Tamer, Mini Bee, Paint Warrior, PoD01Red Bloon, Sabre, Shadow Duelist,
  Sword Knight, Tentacle, Whelp, Warrior Dee, The Master, Panda, Chomp, Abyss Watcher.

## 6. Data schemas: CardInfo and the pickles

`CardInfo` and `CardStats` are `@dataclass`es in `commands/psd_analyzer.py` (:122-193, :93-119).
Consequence: **unpickling stats.pkl/old_stats.pkl requires this repo on sys.path with a config.py
present** (module import chain; it also prints token-presence noise). Pickles from before the
2025-10-20 "Big massive refactor" (366c8d9) were plain dicts. Loading pattern the diagnostics
scripts use: run from the repo root with `sys.path.insert(0, '.')`.

CardInfo fields → pickle/dataframe keys (`to_dict` :138-168, `from_dict` :170-193):

| Field | Dict key | Notes |
|---|---|---|
| `name` | (dict key of stats.pkl, not stored inside) | |
| `card_type` | `type` | always present |
| `path` | `path` | repo-relative, always present |
| `timestamp` | `timestamp` | float epoch, always present |
| `ability` | `ability` | omitted when None (MDW, ability-less cards) |
| `stars` | `stars` | omitted when None |
| `subtype` / `series` | same | omitted when falsy |
| `types` | `types` | list[str] of upper-half icons; omitted when empty |
| `stats.hp/defense/attack/speed` | `hp` / `def` / `atk` / `spd` | SHORT keys; each omitted unless >= 0 |
| `problems` | `problem` | list[str], singular key, omitted when empty |
| `author` | `author` | omitted when falsy |

Omitted keys become NaN in the query dataframes — which is why numeric queries naturally skip
cards lacking that stat. `COLUMN_ORDER` (psd_analyzer.py:36-40) fixes export/display order:
aliases, type, ability, hp, def, atk, spd, types, path, timestamp, author, stars, problem, series,
subtype.

**Timestamp semantics**: local traversal = file mtime (psd_analyzer.py:1009); remote traversal =
last-commit committer date (psd_analyzer.py:1072-1074). A card re-parses when its timestamp is
newer than stored OR its path changed (`_should_update_card` :775-802); `force_update` overrides.
The Python default `update_stats(force_update=True)` (:1177) is NOT what Discord uses — the slash
command defaults it to False (main.py:363). **Author** = committer name of the FIRST commit
touching the file (`commits[totalCount - 1]`, psd_analyzer.py:1072), fetched only in
remote-timestamp mode and only when the card has no author yet; author and series survive reparses
(`parse` copies them from the existing entry, psd_analyzer.py:379-382). `/update_metadata` can set
any of this manually; `/list_orphans` lists author-less cards (0 today).

Pickles (all in the repo root, committed to git except reminder.pkl — commit discipline lives in
autocroissant-change-control):

| File | Shape | Measured 2026-07-11 |
|---|---|---|
| `stats.pkl` | `dict[str name -> CardInfo]` | 813 entries |
| `old_stats.pkl` | `defaultdict[str name -> list[CardInfo]]` | 218 names, 223 archived versions |
| `aliases.pkl` | `dict[str alias -> str target png filename]` | 60 entries, e.g. `'red_bloon' -> 'pod01red_bloon.png'` |
| `reminder.pkl` | reminders, gitignored — not this skill's domain | |

`old_stats.pkl` receives a snapshot whenever a card is about to be re-parsed, when its path
changed, and when it disappears from the repo (`prune_clean_cards`, psd_analyzer.py:244-253 —
deletion means "moved to history", never hard-deleted). Quirk: after each traversal,
`_update_old_stats_paths` (psd_analyzer.py:763-773) rewrites ALL archived entries' `path` to the
card's CURRENT path, so history entries do not preserve historical locations.

Exports: `/export_cards` writes `stats.csv`/`stats.txt`, `/export_rulebook` writes `rules.txt`,
both to the repo root (gitignored). Rulebook entries are excluded from card exports.

## 7. Querying: name lookup, aliases, and the ability DSL

### 7.1 Name lookup and ambiguous names (`/query`)

`CardRepository` (`commands/query_card.py:27`, singleton `card_repo` at :626) maintains
`git_files`: lowercase png filename → repo path, built from one GitHub trees API call
(`populate_files` :55-79). It indexes EVERY `.png` in TTSCardMaker — including `Types/` icons,
`Auxiliary/Markers`, and MDW card backs; `EXCLUDE_FOLDERS` applies only to stats traversal, not
here. So `/query panda` can legitimately return a type icon.

Lookup normalizes to `name.replace(' ', '_').lower() + '.png'` and fuzzy-matches with difflib
`get_close_matches`, cutoff `match_ratio` (default 0.6, `DEFAULT_MATCH_RATIO` query_card.py:23;
runtime-settable via `/set_ratio`, resets on restart — see autocroissant-config-and-flags).

**Duplicate filenames**: when two repo paths share a basename, both also get prefixed keys
`<top_folder>/<name>.png` (lowercased), and `ambiguous_names` records the options
(:81-102). The bare name STILL resolves — to the first-seen path — and `/query` appends a "try
these instead" list when the hit is ambiguous. Duplicated basenames in the repo today: active.png,
cardback.png, equip.png, field.png, fragile equip.png, nme.png, panda.png (mostly card-vs-Types
icon collisions).

**Aliases** (`aliases.pkl`, 60 today): `/alias` with key+value creates (target must resolve to an
existing png; suffix match allowed), key alone deletes, no args lists; saves the pickle immediately
(query_card.py:526-579). At `populate_files` time each alias key is inserted into `git_files` as
`<alias>.png` pointing at the target's path (:110-124), so aliases work everywhere names do.
Everything is lowercased on load. Examples: `abyss_watchers -> the_abyss_watchers.png`,
`red_bloon -> pod01red_bloon.png`.

### 7.2 The ability-search DSL (`/query_ability` → `ability_search_engine`, query_card.py:194-264)

Search runs over `cards_dff`, a pandas DataFrame built from stats.pkl (`prep_dataframes`
:315-334): index = card names, columns = the dict keys of section 6, **ability column lowercased**,
rulebook pages split into a separate `rulebook_dff`, and rows with no ability text (MDW-style)
excluded entirely. Refreshed at startup and queued after every `/update_stats`.

| Syntax | Meaning | Example |
|---|---|---|
| bare text | substring match on ability + fuzzy fallback | `frostmourne` |
| `"quoted text"` | word-boundary-ish regex match on ability | `"draw 2"` |
| `!expr` | NOT — negated ability-text match only | `!die` |
| `a & b` | AND (index intersection) | `type==creature & hp>7` |
| `a \| b` | OR (index union, deduplicated) | `hp>9 \| spd>9` |
| `col<n` `<=` `==` `>=` `>` | numeric column compare (int only) | `stars<=3`, `def==10` |
| `col==value` | string equality, case-insensitive | `series==kirby` |
| `col~value` | string contains, case-insensitive | `type~creature` |

Numeric-queryable columns: hp, def, atk, spd, stars. String-queryable: type, subtype, series,
author, path, ability, types (the list is stringified, so `types~undead` works).

**Evaluation order and gotchas** (all verified in code, 2026-07-11):

1. Precedence: `|` splits BEFORE `&`, so `a & b | c` = `(a AND b) OR c`. No parentheses exist.
2. Stat/column queries are recognized AFTER splitting but BEFORE `!` — so `!type==creature` is NOT
   a negated filter; it becomes a negated TEXT search for the literal string "type==creature"
   (matches nearly everything). `!` only composes with text terms.
3. `/query_ability` lowercases the whole query, and abilities are pre-lowercased in the dataframe:
   text search is case-insensitive; so are `==`/`~` values.
4. A column query on a nonexistent column (`hp2>3`, typo'd `serie==kirby`) silently falls through
   to TEXT search of that literal string — expect 0 or nonsense hits, not an error.
5. `type==creature` misses the 7 plural-"creatures" cards (section 2); use `type~creature`.
6. Numeric compares drop NaN rows, i.e. only cards that HAVE the stat participate.
7. Bare-text search also runs a fuzzy pass: `SequenceMatcher(None, query, whole_ability).ratio() >
   match_ratio` (query_card.py:257-260). It compares against the ENTIRE ability, so it only adds
   near-whole-text matches; short queries effectively rely on the substring pass.
8. `"quoted"` builds `(?:^|\s|$|\b)<escaped>(?:^|\s|$|\b)` (:244) — approximately word-bounded,
   punctuation-tolerant.
9. Rulebook pages are NEVER returned by `/query_ability`; use `/query_rulebook` (plain
   case-insensitive substring over `rulebook_dff`, no DSL — query_card.py:266-290). 48 pages today.
10. Results return card IMAGES (urls via `get_card_url`); `howmany:True` returns just the count;
    `limit` caps output (main.py:268-281).

## 8. When NOT to use this skill

- Running the bot, running `/update_stats`, pushing/pulling pickles, machine handoff →
  **autocroissant-run-and-operate** (it owns the operating rule: as of 2026-07-11 always run
  `/update_stats` with `use_local_repo:False` — a live local-path bug, story in
  **autocroissant-failure-archaeology**).
- Diagnosing or FIXING a mis-parsed card, shrinking the exclusion lists, changing injection
  heuristics → **autocroissant-psd-extraction-campaign** (executable plan) with
  **autocroissant-validation-and-qa** as the acceptance gate.
- How to invoke `parse_one.py` / `gap_trace.py` / `dump_psd_layers.py` / `inspect_pickle.py` /
  `diff_stats.py` and read their output → **autocroissant-diagnostics-and-tooling**.
- Constants as CONFIG (defaults, what is runtime-settable) → **autocroissant-config-and-flags**.
- Committing pickles or any schema/code change discipline → **autocroissant-change-control**.
- Queue/threading mechanics behind these commands → **autocroissant-architecture-contract**.

## 9. Provenance and maintenance

Every load-bearing claim above is from `commands/psd_analyzer.py`, `commands/query_card.py`,
`global_config.py`, `main.py`, the TTSCardMaker clone, and live script runs on 2026-07-11.
Line numbers drift — re-find facts with these before relying on cited lines:

| Fact | Re-verify with |
|---|---|
| Constants (ratio, gaps, folders) | `grep -n "TYPE_REGION_RATIO\|EXCLUDE_FOLDERS\|UPDATE_RATE\|MISSPELT_CARD_TYPES\|re_compile(r'\\\\s{3,}')" commands/psd_analyzer.py` |
| Classification table | `sed -n '/class CardClassifier/,/class PSDParser/p' commands/psd_analyzer.py` |
| Exclusion lists (5 + 18 names) | `sed -n '/class CardValidator/,/def validate/p' commands/psd_analyzer.py` |
| Injection / prune / sort internals | `grep -n "_inject_type_names\|_prune_type_bboxes\|_sort_by_position\|row_threshold" commands/psd_analyzer.py` |
| Stat summing + hidden→10 default | `grep -n "tracker.value\|else 10" commands/psd_analyzer.py` |
| Stars-from-PSD folder list | `grep -n "Auxiliary/Items" commands/psd_analyzer.py` |
| CardInfo/CardStats mapping | `sed -n '/^class CardStats/,/^class StatsDatabase/p' commands/psd_analyzer.py` (approx; or grep `def to_dict`) |
| DSL internals | `grep -n '_stat_pattern\|ability_search_engine\|SequenceMatcher\|duplicated' commands/query_card.py` |
| Defaults (repo, match ratio) | `grep -n "DEFAULT_REPOSITORY\|DEFAULT_MATCH_RATIO" commands/query_card.py` |
| Clone location | `grep -n LOCAL_DIR_LOC global_config.py` |
| Slash surface + update_stats defaults | `grep -n 'tree.command(name="query\|name="alias\|name="update_stats' main.py` and `sed -n '359,365p' main.py` |
| Folder tree | `ls ~/Desktop/TTSCardMaker ~/Desktop/TTSCardMaker/Auxiliary ~/Desktop/TTSCardMaker/Items` |
| PSD/PNG counts + 813 reconciliation | `find ~/Desktop/TTSCardMaker -name "*.psd" \| wc -l` and same with `! -path "*/MDW/*" ! -path "*/Markers/*"` |
| Pickle counts, distribution, problem cards, newest card | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |
| Alias count | `python3 -c "import pickle;print(len(pickle.load(open('aliases.pkl','rb'))))"` |
| The 7 "creatures" cards | `ls ~/Desktop/TTSCardMaker/Auxiliary/Creatures/*.psd` |
| Golden parses (Mini Doomer / Freezer / Lich King) | `parse_one.py` / `gap_trace.py` with the paths used in sections 3-4 (absolute paths into the clone; outputs include ignorable "Git token found" lines) |
| Types vocabulary (35 remote vs 36 local w/ .ds_store) | `git -C ~/Desktop/TTSCardMaker ls-files Types/ \| grep -vc Stars` vs the `known types` line of any `parse_one.py` run |
| README citation | `head -3 ~/Desktop/TTSCardMaker/README.md` |

Maintenance: after any parser or repo change, re-run `inspect_pickle.py` and the three golden
parses and update the numbers here (counts, distribution, exclusion-list sizes, duplicated
basenames) plus the date stamps. If `TYPE_REGION_RATIO`, the gap regex, or the exclusion lists
change, sections 4-5 are stale until rewritten.
