---
name: impossibility-cards-reference
description: Domain reference for the Impossibility Simulator card game AS ENCODED in AutoCroissant and the TTSCardMaker repo. Load whenever a task mentions cards, PSDs, PSD layers, ability text, type injection or [type] tokens, stats extraction (hp/def/atk/spd), stars/Warpstars, folder structure or card classification, CardInfo/CardStats, the stats.pkl / old_stats.pkl / aliases.pkl schemas, query syntax or the query DSL (/query, /query_ability, /query_rulebook; the | & ! "exact" hp>5 type==creature type~x operators), rulebook pages, aliases or ambiguous card names, validation problems (UNKNOWN TYPE, ABILITY TEXT NOT FOUND, STATS TOO HIGH, MISSPELT TYPE), the exclusion lists, or the creatures-vs-creature anomaly. Contains the folder-to-classification table, PSD layer semantics with an annotated real layer dump, the precise type-injection algorithm with constants and a worked trace, validation rules with the ability exclusion list spelled out (the stat list is gone since 2026-10-21; `allow_high_stats` in Card Edits.json replaces it), pickle schemas with measured counts, and the full query DSL with gotchas. This is the "what the data means" pack — not an operating runbook (run-and-operate), not a bug-fixing campaign (psd-extraction-campaign), not script usage docs (diagnostics-and-tooling).
---

# Impossibility Simulator: the card data model, as encoded here

All counts and line numbers verified 2026-07-11; §3.3, §4.1, §6 timestamp semantics and §8-9
updated 2026-10-19 (icon font, `.DS_Store` fix, `force_update` default) and 2026-10-20 (exact token
spellings, remote-mode default, blob-sha re-parsing, new CardInfo fields incl. the database-only `subseries`, §6 line refs). Line numbers drift; every volatile fact has a
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
`global_config.py:5`. This bot (AutoCroissant) READS TTSCardMaker over the GitHub API (reading the
local clone, "local mode", was archived on 2026-10-20; the clone still serves the offline diagnostics
scripts), parses the PSDs into a stats database (`stats.pkl`), and answers queries against it.
The bot never writes to TTSCardMaker.

Scale, measured 2026-07-11: 904 PSDs and 1003 PNGs in the clone; 813 entries in `stats.pkl`.
The books balance exactly: 904 − 90 (`MDW/`, excluded) − 1 (`Auxiliary/Markers/`, excluded) = 813
non-excluded PSDs, all with unique names.

**Card identity**: a card's name is its PSD filename with underscores turned into spaces
(`The_Freezer.psd` → "The Freezer"; `commands/psd_parser.py:337`). Names are the keys of
`stats.pkl` and the index of the query dataframes. Two PSDs with the same basename would silently
share one entry — today there are none outside the excluded folders.

## 2. Folder taxonomy → card classification

`CardClassifier.classify` (`commands/psd_parser.py:230-315`) derives a card's type and several
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

`subseries` (2026-10-20) never comes from the folder or the PSD: only Card Edits.json or `/update_metadata` set it, and a
re-parse keeps it like `author` and `series` (§6).

Observed live subtrees (2026-07-11): `Items/` subtypes = Active, Attack, Counter, Deploy, Equip,
Heal, Infliction, Negation, Support; `Field/` and `Items/*/` have `1 Stars`..`4 Stars`; Creatures
series (24 of them: BTD, Kirby, Souls, World Of Warcraft, ...) have star folders in roughly the
`2 Stars`..`5 Stars` range; `Auxiliary/` contains Creatures, Items, Markers, Minions, Rulebook,
Status Effects.

**The stars parse** (psd_parser.py:260, 259, 278, 304) is:

```python
"stars": int(folders[len(folders) - 1].split()[0])
```

i.e. the integer prefix of the LEAF folder name ("3 Stars" → 3). Consequence: for Field, Items,
Creatures, and Debuffs, the leaf folder MUST start with an integer or `classify` raises
`ValueError`. (The `x[len(x)-1]` instead of `x[-1]` is the repo's Cython negative-indexing ban —
see `autocroissant-build-and-env`; do not "clean it up".)

**Stars from the PSD instead**: for `Auxiliary/Items`, `Auxiliary/Minions`, and `N.M.E` (exactly
these three, matched as substrings of the relative path), stars are COUNTED from the PSD: `PSDParser._count_stars`,
since 2026-10-20.
- **Search area:** every outermost visible group with "stars" in its name (`_star_groups`).
- **Counting** (`_stars_in`): child groups count recursively. Visible pixel layers that overlap count as one star
  (`_count_clusters`). Overlap means one layer's *core* center lies inside the other's core, where the core is the
  box of pixels at least 50% opaque (`STAR_CORE_ALPHA`, `_core_box`). Glows, frames and the pieces of one star
  merge; separate stars don't, even when, like Termination Shock's lens flares, their full boxes overlap.
- **Fallback:** with no such group, visible layers whose name has the word "star" ("Patrick Star", "Emerald_Star",
  not "starfish") are counted the same way.
- **Zero stars** gives the persisted problem `NO STARS FOUND (put the star layers in a group named "Stars")`. On
  2026-10-20 that was God Tamer (five "Delicate Flower" layers) and Bubble (a starfish).
- **Header text:** icon-font stars in it are added on top (§4.1).

**Why it changed:** the old rule added 1 per visible pixel layer with "stars" in its 3 nearest parent groups.
A star made of several layers was counted several times: Indoctrination 20 (4 stars × 5 layers), Michael's
Blessing 11 (5), Spawn of the Forbidden One 10 (4). Cards whose stars aren't under a "Stars" group came out as 0
(Nasty Patty 3, Shell Shield 2). Checked over all 165 such cards: only those 5 changed, and every other count
stayed the same. The fix reaches stats.pkl and the website at the next `/update_stats` + `/push`. Mini Doomer still
has two "Warpstar" smart objects under "Stars", so stars=2.

### The creatures-vs-creature anomaly (do not fix silently)

`Auxiliary/Creatures/` has no dedicated branch in `_classify_auxiliary`, so its 7 cards get the
fall-through type — the literal leaf folder name, lowercased: **"creatures" (plural)**. The 297
cards under top-level `Creatures/` get **"creature" (singular)**. A `type==creature` query MISSES
the 7. They are, as of 2026-07-11: Azamoth the Possessed Vessel, Igor Roballtowski, Mettaton EX,
Mettaton NEO, Militron (No Armor), Morpho Knight, Shadow Queen. Workaround: `type~creature`
(contains) matches both. The same fall-through gives `Auxiliary/Rulebook/` pages type
**"rulebook"** (48 entries). Also note the 7 "creatures" cards skip the creature-specific handling:
no series, no −1 stat seeding, so the stat validators (section 5) never fire for them. Their printed stars (2-5) are
counted from the PSD since 2026-10-23 (`psd_parser.STARS_FROM_PSD`); the stored cards get them at a `cards:` re-parse.

### Excluded folders

`EXCLUDE_FOLDERS = ["Markers", "MDW"]` (card_data.py:42 since 2026-10-22) is checked against every path segment
during stats traversal and duplicate-PNG detection (card_data.py:438, 377). `MDW/` (90 PSDs) is
Photoshop templates — card backs, `.aco`/`.atn` assets; `Auxiliary/Markers/` (1 PSD) is a board
marker. Neither reaches `stats.pkl` (verified: 0 MDW-typed entries). The `MDW` classifier branch is
therefore dead in normal traversal; it only matters if `classify` is called on such a path directly
(e.g. by `parse_one.py`).

Distribution in `stats.pkl` (2026-07-11, from `inspect_pickle.py`): creature 297, item 182,
minion 91, debuff 52, field 50, aux item 48, rulebook 48, nme 26, buff 12, creatures 7 — total 813.

## 3. PSD anatomy — what the parser actually reads

This is the part newcomers lack. A card PSD is a layer TREE; the parser walks
`psd.descendants()` and dispatches on layer kind, name, visibility, and position
(`PSDParser._process_layer`, psd_parser.py:522-549). Annotated excerpt of a real dump
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
card types. `psd_parser.py:528` dispatches text layers on exactly this. Card-type ICONS are
pixel/smart-object layers matched by NAME (3.3). Misread this and nothing else in the parser makes
sense.

### 3.2 Ability text

A text layer contributes ability text iff its name (lowercased) is `"ability"` — OR the card is a
rulebook page (`"Rulebook" in relative_path`), in which case ALL text layers contribute
(psd_parser.py:562, 377). The raw string is `str(layer.engine_dict["Editor"]["Text"])` — a
repr-like string wrapped in quotes, with literal escape sequences. Cleanup (psd_parser.py:561-582):

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
(psd_parser.py:832-842). If NO ability text layer exists at all, the parse-time problem
`NO ABILITY LAYER` is recorded (psd_parser.py:414).

### 3.3 Card-type icons and the midline

`all_types` is the vocabulary of icon names, populated from TTSCardMaker's `Types/` folder
(excluding `Types/Stars/`), lowercased file stems: 'fire', 'ice', 'undead', 'defensive_minion', ...
Remote traversal yields 35 (`_populate_types_from_response`); the clone-based helper used by the
diagnostics scripts (`_populate_types_from_local`, card_data.py:294) also yields 35 since 2026-10-19, when it started
skipping dotfiles — before that it leaked `'.ds_store'` as a 36th entry (archaeology Entry 10.3).
Since 2026-10-19 a second vocabulary
sits beside `all_types`: `IconVocabulary`, which adds `Types/Stars/` names and the font manifest's
aliases for icon-font tokens (§4.1).

A visible, pixel-bearing layer whose lowercased name is in `all_types` is a type icon
(psd_parser.py:536-541). Position decides its meaning, using
`card_mid_y = int(psd.height * TYPE_REGION_RATIO)` with `TYPE_REGION_RATIO = 0.5`
(psd_parser.py:23, 375):

- bbox top `y < card_mid_y` (upper half, the corner badges) → appended to `card.types`
  (in raw traversal order, NOT sorted);
- `y >= card_mid_y` (lower half, sitting inside the ability paragraph) → candidate for INLINE
  INJECTION into the ability text (section 4).

### 3.4 Stats (hp/def/atk/spd)

Two-stage name matching (psd_parser.py:690-740):

1. A layer is a stat layer if any ancestor (up to 3 levels) has "dark" or "bars" in its name
   (`Stats Bars > Darks > Hp Dark` qualifies).
2. It contributes iff its name `isdigit()`; the nearest 2 ancestor names are matched for the
   substrings `hp` / `def` / `atk` / `spd` (first hit wins).

Semantics: a digit layer FOUND under e.g. "Hp Dark" marks hp as present, whether visible or not.
The VISIBLE digit layers are grouped by bar row (their parent group) and each row gives a value
(`StatTracker._row_value` / `total`, psd_parser.py):

- **Darkening adjustment layers** (the template: `exposure` layers named 0-9, no pixels): the one
  visible digit; if several are visible, the lowest (it already darkens the rest of the bar; Sans had
  "1" and "9" visible: defense 1, not 10). Mini Doomer: "5" visible under Hp Dark → hp=5.
- **One drawn layer per cell, named 1-10** (heart bars: Blank Card, Dirty Bed, Tainted Lazarus):
  count the lit cells (`pixel` layers) visible, or 10 minus the dark cells (`smartobject`) when only
  those are visible. Adding up the names gave 55. A "Light" row and a "Dark" row in the same stat
  group draw the same bar and count once.
- **Anything else** (value layers or groups, plus "0" decorations: Clots, False Knight): the visible
  digits add up.

Rows add up, so stats can pass 10 (Royal Eradicator Main Cannon's attack is 10 + 2). Multi-creature
cards (rows named "1", "2", … or a repeated stat block: Pix, Twin Emperors, Tainted Lazarus) get
`card.creatures` (CardStats each; `stats` = creature 1) and the note "SEVERAL CREATURES" (a guess:
Twin Emperors' 2nd DEF read 4, the card shows 10). Fix them with the website's per-creature edit (all three done 2026-10-21 in Card Edits.json). A stat past 10 (two bar rows: the canonical encoding is a top row 10 plus a
bottom row N-10) is the problem "STATS TOO HIGH" unless the card's Card Edits.json entry sets `allow_high_stats` (Sonic spd 20,
Royal Eradicator Main Cannon atk 12; `EXCESSIVE_STAT_EXCLUSIONS` went on 2026-10-21). A digit-named group that holds digit layers is a row,
not a value (Sonic's rows "2" and "1" once added 3 speed), and an empty one is skipped (Twin
Emperors' empty "1" in "shared health"). If a stat was found but every digit layer is hidden, the
value defaults to **10**: that is how maxed stats are encoded in the template. Never found → stays
−1 (only seeded for creatures/minions), which the validator reports.

### 3.5 Position sorting (shared by text layers and icons)

`_sort_by_position` (psd_parser.py:864-891): sort by bbox top `y`; group into visual rows where
`|y − first_in_row.y| <= 40` px (`row_threshold=40`); sort each row left-to-right by `x`; flatten.
This is reading order — top-to-bottom, then left-to-right — and it determines both multi-text-layer
join order and type-icon injection order.

## 4. The type-injection algorithm (the hardest live problem)

Card authors drop type ICONS inside the ability paragraph in Photoshop; the text layer just has a
run of spaces where each icon sits. Extraction must splice `[typename]` tokens back into those
gaps. This is the owner-stated hardest live problem; the spec below is exact
(`_inject_type_names`, psd_parser.py:901-945, plus callers at 529-539).

Constants:

| Constant | Value | Where |
|---|---|---|
| `TYPE_REGION_RATIO` | 0.5 (midline) | psd_parser.py:23 |
| gap pattern | `\s{3,}` (3+ whitespace chars) | psd_parser.py:324 |
| row grouping threshold | 40 px | psd_parser.py:865 |
| prune rule | keep icon iff `y >= max(last_sorted_icon_y // 3, card_mid_y)` | psd_parser.py:898 |
| punctuation cleanup | `sub(r'\s+([:;,\.\?!])', r'\1')` after injection | psd_parser.py:841 |

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
   (psd_parser.py:940-943). More gaps than icons: extra gaps stay as-is as literal spaces.
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
(Since 2026-10-21 the PSD reads `(1GCD)` there: the team's cooldown update, impossibility-simulator-rules §3.)

(The golden values quoted in sections 3-4 — Mini Doomer, The Freezer, The Lich King — are
snapshots of 2026-07-11; canonical current expected values live in autocroissant-validation-and-qa
§3. If one stops matching, check there before assuming a regression.)

**Load-bearing invariant: gap whitespace IS the signal.** Never collapse runs of spaces in ability
text, in parser code or in stored data — a 2026-02-08 eight-commit saga (7 parser commits + a
mid-saga PICKLE) added `\s{2,}` collapsing and had to remove it entirely (3bbaa2b "Fix the issue
of collapsing spaces" + 4bcee6b). Full story: autocroissant-failure-archaeology Entry 2.

### 4.1 Cards that use the stoogedings icon font (added 2026-10-19)

From 2026-10-19, card authors can type `[fire]` in the **stoogedings** icon font instead of placing an icon
layer over a gap. The font lives at TTSCardMaker `MDW/Fonts/stoogedings.ttf`. Its build source and manifest are
`MDW/Fonts/stoogedings-src/stoogedings.json`, which holds names, aliases and codepoints. The PSD text then
literally contains the token, so no gap guessing is needed.

`_process_text_layer` finds the character ranges set in that font by matching PostScript names that start with
`stoogedings`. It does this with `_icon_font_spans`, using the style runs and the FontSet, with run lengths
counted in UTF-16 units. If a layer has no such range, it takes the **old path byte-for-byte**. This was verified
on all 813 PSDs on 2026-10-19 by comparing old and new parser output: 0 differences.

For layers that do use the font, `_read_icon_tokens` rewrites what's in it:
- Tokens are matched in any case, but otherwise only in the spellings the font itself draws (exact since
  2026-10-20, `IconVocabulary.spellings`): the name's words joined by one space, `_`, `-` or nothing, plus the
  same forms of each manifest alias. The result is canonical: `[Fragile_Equip]` becomes `[fragile equip]`,
  `[dark matter]` (an alias) becomes `[darkmatter]`, and `[N.M.E]` works through its alias `n.m.e`.
  Spellings the font would draw as letters — `[fragile.equip]`, `[fragile  equip]` (two spaces), `[ fire]`,
  `[fire: no]` — are `UNKNOWN ICON TOKEN` problems. Until 2026-10-20 the parser also ignored dots, apostrophes
  and extra spaces and so accepted those silently (archaeology Entry 10.11).
- `:tint` and `:sticker` are dropped. `[x:no]` is kept as `[x:no]`, because it means "non-x".
- Private-use characters inserted from the Glyphs panel become `[name]`, and `★` becomes `[star]`.

The result is passed through `repr()`, since `str()` of a psd-tools engine string *is* its repr, and then the
usual cleanup runs.

Visible non-ability text in the font that sits above the midline is treated as a header:
- Type tokens are appended to `card.types`.
- Star tokens add to the star count, for the folders whose stars come from the PSD.

Gap injection (section 4) still runs afterwards for image-layer icons, so mixed cards work.

The vocabulary is `IconVocabulary`. It is built from `Types/` (the same names as `all_types`), `Types/Stars/`
(lowercased, spaces as `_`) and the manifest. If the manifest is missing, it falls back to plain names.

There are four new parse-time problems, all persisted:

| Problem | Meaning |
|---|---|
| `UNKNOWN ICON TOKEN: [x]` | Typo or unknown variant. The card shows letters. |
| `UNKNOWN ICON CHARACTER: U+XXXX` | A private-use character that isn't in the manifest. |
| `ICON TOKEN NOT IN STOOGEDINGS FONT: [x]` | A known icon name typed in the body font. |
| `ICON TOKEN PARTLY IN STOOGEDINGS FONT: [x]` | The font was only applied to part of the token. |

**Badges.** The font also turns timing tags typed in it, such as `(1GCD)`, `(1/2CD)`, `(2U)` or `(OYT)`, into
colored pill badges (exact uppercase only), adopted as part of the font on 2026-10-19; the cards' cooldown and uses
tags of 2026-10-21 join the rulebook's old tags, which stay (tag list, count, look and sizes: ttscardmaker-companion
§1.2; meanings: impossibility-simulator-rules §3). That is purely a rendering ligature: the text is still `(1GCD)`,
so the parser stores it unchanged, and nothing in AutoCroissant parses badges. Matching rule for the
parser: a tag in parentheses is never a token, because tokens need square brackets.

**Vocabulary at a glance (2026-10-19).** 91 icons in the manifest (35 `type` from `Types/`, 56 `star` from
`Types/Stars/`), fixed codepoints U+E000–U+E05A, 15 icons with aliases; variants `:tint`, `:sticker`, `:no`.
The font, its badge list, codepoint allocation and rebuild procedure are owned by
**ttscardmaker-companion** (§1). Remaining known mismatch: the manifest keeps entries for icons whose PNG
was renamed or deleted, so their old tokens still parse while the rebuilt font draws letters
(docs/BUGS_AND_IMPROVEMENTS.md C13).

**Adoption (2026-10-19).** No card PSD uses the font yet: 0 of 813 have a text layer in it, so every stored
token today still comes from gap injection (§4). About 45 PSDs (81 smart-object icon layers) are the
migration backlog; the conversion prototype is described in ttscardmaker-companion §5.

**Discord rendering** lives in `commands/type_emojis.py`. `/sync_emojis` and the startup sync upload
`Types/` and `Types/Stars/` as application emojis for whichever bot identity is running, plus a
crossed-out `<name>_no` copy of each `Types/` icon (35 of them; stars get none), 126 in all.
`render_type_tokens` turns `[fire]` into the fire emoji and `[fire:no]` into the `fire_no` emoji; only
if that emoji is missing does it fall back to 🚫 followed by the fire emoji. Brackets that aren't a
known emoji (`[X]`, `[Status]`) stay as text. Emoji names are derived from the canonical token
(lowercase, other characters → `_`, max 32 characters), so `[fragile equip]` maps to `fragile_equip`.
`/query_rulebook` now sends escaped plain text through it instead of a code block, because code
blocks don't render emojis. Operations: autocroissant-run-and-operate §11.

### 4.2 Word effects: overlay text layers and effect layers (added 2026-10-20)

Some cards give a word or phrase in the ability text a glow, stroke or gradient, such as Gasoline's glowing red
"Blazing". Since only layers named `ability` are read (§3.2), none of the decoration below changes the parsed text.

- **Old way: overlay text layers.** A second text layer holds a copy of the word, hand-placed over the word in the
  `Ability` text, with a layer style. Survey 2026-10-20: 53 overlays on 40 cards, each in the same font and size as
  the word underneath, and the word always also in the `Ability` text. 27 of them use a different color from the
  word underneath. Overlays sit outside the `Text` group 50 times out of 63 (13 inside), because that group's own
  Stroke and Drop Shadow (on 652 of 765 cards) would otherwise outline the glow. Four more cards (Bulborb Larva,
  Butterfly, Empress Bulblax, Apex Predator) hide `Ability` and lay the text out again in separately styled
  layers. The hidden `Ability` is what gets parsed only while those visible layers over it print exactly its words
  (`PSDParser._relaid`, 2026-10-22: only printed text counts); otherwise the card has no ability. Overlays go stale
  silently when the text is edited.
- **New way: effect layers.** TTSCardMaker `MDW/Scripts/Effect Layers.jsx` makes a pixel layer named
  `<text layer> FX: <words>`, e.g. `Ability FX: Blazing` or `Ability FX: Overkill DMG carries over | Frozen`
  (`|` separates phrases, matched in any case and across line breaks). It holds exactly those words' pixels, cut
  from the text layer, with a layer style. The words keep the card's font, size and color, the script places the
  layer above the text's group, and its "Update all" button rebuilds every effect layer after the text changes.
  How it works and why fonts were rejected: ttscardmaker-companion §6.
- **Parser handling** (`EFFECT_LAYER_TAG = " FX: "`, `PSDParser._check_effect_layer`): a non-group layer whose
  name contains ` FX: ` is set aside, not parsed. After the walk, each **visible** effect layer is checked, giving
  three new parse-time problems (persisted like the icon problems):

| Problem | Meaning |
|---|---|
| `EFFECT LAYER WITHOUT TEXT LAYER: <name>` | No text layer has the name before ` FX: ` (renamed or deleted). |
| `EFFECT LAYER WORDS NOT IN TEXT: <name>` | An orphaned layer (idea 14): a phrase is no longer in that text layer, or `Frozen#2` names an occurrence past the last match (`PSDParser._occurrences` counts like Effect Layers.jsx: whole words when there are any, else every match). Checked first, so an orphan reads as this, not as OUT OF DATE (Mettaton with a Blazing layer but no Blazing in its text scored 0.498 too). |
| `EFFECT LAYER OUT OF DATE: <name> (run Effect Layers > Update all)` | Under 95% (`EFFECT_COVERAGE_MIN`) of the effect layer's opaque pixels sit on the text layer's own stored pixels: the text was edited or moved after the layer was made. A fresh layer scores 1.0. Gasoline with words inserted before "Blazing" scored 0.268, even though the word was still in the text. |

Measured 2026-10-21: no card PSD in TTSCardMaker has an effect layer yet (0 of 813, so all three fire 0 times);
on 72 Photoshop-made trial copies outside the corpus (/tmp, the JSX trials and migration rehearsals) they flagged 3, all
true: 2 OUT OF DATE (the deliberately stale Gasoline copies) and 1 WORDS NOT IN TEXT (Mettaton_newlayer).

The pixel check uses psd-tools `layer.numpy("shape")` on both layers (alpha only, so recoloring is fine). Text
layers keep their last-rendered pixels in the PSD even when hidden. If either layer has no pixel data, the check
is skipped.

## 5. Validation semantics (`CardValidator`, psd_parser.py:966-1056)

Problems come in two kinds, and only one kind is persisted — a common newcomer misread:

- **Parse-time problems** are appended to `card.problems` during parsing and stored in the pickle (the website's Parser notes):
  `NO ABILITY LAYER` (psd_parser.py:414), `NO STARS FOUND` (:420), `MISSPELT TYPE: <name>` (:566-567,
  fired by a visible pixel layer named in `MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']`,
  line 24), the effect-layer and icon-token ones through `_add_problem` (:641; callers :489-504, :674-706) and `SEVERAL CREATURES` (:880).
- **Validate-time problems** are computed by `CardValidator.validate` during `/update_stats`,
  REPORTED to Discord, and NOT written back to the card. Since 2026-10-21 the same `validate()` (after the card's
  `site/Card Edits.json` entry) also runs per pushed PSD in `commands/card_check.py`, the push guard's card check
  (TTSCardMaker card-check.yml; autocroissant-run-and-operate §7); of its output only the parse-time problems above reach the website.

So the `problem` field in stats.pkl only ever holds parse-time strings. As of 2026-07-11 exactly
one card has one: rulebook page "20 Creature Types" with `MISSPELT TYPE: tornado`.

| Problem string | Fires when | Source |
|---|---|---|
| `UNKNOWN TYPE` | `card_type == "unknown"` (path matched no classifier row) | :1008-1009 |
| `ABILITY TEXT NOT FOUND` | type is not unknown/MDW, ability empty, name not in ABILITY_EXCLUSIONS | :1012-1016 |
| `HP NOT FOUND` / `DEF NOT FOUND` | creature or minion with that stat still −1 | :1038-1041 |
| `ATK NOT FOUND` / `SPD NOT FOUND` | same, UNLESS `"active"` is in `card.types` (active creatures have no atk/spd) | :1044-1050 |
| `STATS TOO HIGH` | any found stat > 10 and not `allow_high_stats` (Card Edits.json; the stat exclusion list went 2026-10-21) | psd_parser.py:1053-1054 |
| `NO ABILITY LAYER` | no ability text layer found at parse time; validate() DELETES it again for names in ABILITY_EXCLUSIONS | :414, :1024-1026 |
| `MISSPELT TYPE: <t>` | visible layer named undread/tornado/error | :566-567 |

**The exclusion list is the "perfect extraction" debt register** — per-card allowances the owner
wants to eliminate (see autocroissant-research-frontier; the campaign to shrink them is
autocroissant-psd-extraction-campaign). `EXCESSIVE_STAT_EXCLUSIONS` (5 names) was deleted on 2026-10-21: stats above 10
are allowed per card by `allow_high_stats` in Card Edits.json. Verbatim, psd_parser.py:968-990:

- `ABILITY_EXCLUSIONS` (21, allowed to have no ability text): Crystal Bits, Bugzzy, Electro Probe,
  Galacta Warrior, God Tamer, Mini Bee, Paint Warrior, PoD01Red Bloon, Sabre, Shadow Duelist,
  Sword Knight, Tentacle, Whelp, Warrior Dee, The Master, Panda, Chomp, Abyss Watcher, and since
  2026-10-21 (user) the three minions with an empty Ability layer: Anubisath Guardian, Qiraji Soldier, Silithid.
- **Shadow Duelist is not stale** (BUGS D3): its PNG prints no ability, and its hidden `Ability` layer ("Heals the
  creature this is summoned on by 2 HP at the end of its turn.") is laid out nowhere, so since 2026-10-22 (owner: hidden
  Ability layers don't count, only what is printed) it parses to an empty ability and the entry keeps the validator
  quiet. A hidden `Ability` layer counts only when visible text layers over it print exactly its words (the four
  relaid-out cards of §4.2); the 7 cards with only a hidden one on 2026-10-22: those four keep their text, Shadow
  Duelist, Abyss Watcher and Warrior Dee (all excluded) parse to "". `tests/test_parser_goldens.py::test_shadow_duelist_exclusion`
  pins the PSD (skips when it changes upstream); `tests/test_rulebook_hidden.py` the rule.

## 6. Data schemas: CardInfo and the pickles

`CardInfo` and `CardStats` are `@dataclass`es in `commands/card_models.py` (:81-151, :52-79), pickled as
`commands.psd_analyzer.*` (`__module__` pinned since the 2026-10-20 split).
Consequence: **unpickling stats.pkl/old_stats.pkl requires this repo on sys.path with a config.py
present** (module import chain). Pickles from before the
2025-10-20 "Big massive refactor" (366c8d9) were plain dicts. Loading pattern the diagnostics
scripts use: run from the repo root with `sys.path.insert(0, '.')`.

CardInfo fields → pickle/dataframe keys (`to_dict`, commands/card_models.py; the lossy `from_dict` was deleted 2026-10-20):

| Field | Dict key | Notes |
|---|---|---|
| `name` | (dict key of stats.pkl, not stored inside) | |
| `card_type` | `type` | always present |
| `path` | `path` | repo-relative, always present |
| `timestamp` | `timestamp` | float epoch, always present |
| `ability` | `ability` | omitted when None (MDW, ability-less cards) |
| `stars` | `stars` | omitted when None |
| `subtype` / `series` | same | omitted when falsy |
| `subseries` | `subseries` | a series within the series (Goombella: series Mario, subseries Paper Mario: The Thousand-Year Door); database-only — from Card Edits.json or `/update_metadata`, never the PSD or folder, kept on re-parse like `author`; omitted when falsy (2026-10-20) |
| `types` | `types` | list[str] of upper-half icons; omitted when empty |
| `stats.hp/defense/attack/speed` | `hp` / `def` / `atk` / `spd` | SHORT keys; each omitted unless >= 0 |
| `problems` | `problem` | list[str], singular key, omitted when empty |
| `author` | `author` | omitted when falsy |
| `display_name` | same | the printed title when it differs from the file name (since 2026-10-21, PARSER_VERSION 2: the visible `Text/Name` text layer, several in the title region joined in reading order; else the visible `Name` text layers in the top `TITLE_REGION_RATIO` of the card; else a visible layer under `Text` named after the card; byte-order mark, control characters and whitespace runs collapsed, `psd_parser.title_text`; a `Name` layer holding part of the file name beside layers holding the rest, Royal Eradicator Main Cannon's "Main Cannon", or WX-78's "wx-" beside an image layer named "78", reads as the file name; hidden or rasterized titles read as nothing; 157 of 813 cards, 94 of them differing only in letter case or spacing, shown as printed (the user's call, 2026-10-21); `/query` finds a card by its printed title). A Card Edits.json `name` overrides it (the title is kept in `parsed_values`); duplicates get theirs from `_duplicate_name`; omitted when None |
| `note` / `hidden` | same | from Card Edits.json; omitted when falsy |
| `duplicate_of` | same | the original card, for another PNG of the same PSD; omitted when falsy |
| `creatures` | `creatures` | list of short-key stat dicts, one per creature (multi-creature cards); omitted when falsy |

`problems_ok`, `allow_high_stats`, `parsed_values`, `alt_names`, `blob_sha`, `parser_version` (the `PARSER_VERSION`
that parsed the card, 2026-10-21; None in older pickles, which makes `/update_stats` re-parse them once) and `archived`
(2026-10-22, set only on an archived card's record in old_stats.pkl, below) live only in the pickle (stats.pkl stores
CardInfo objects); they are not `to_dict` keys.

**Two kinds of Card Edits.json edits, PSD-backed or not** (`CardEdits.KIND`, card_edits.py, 2026-10-20, splits the second into folder and database; the same table is `PSD_FIELDS`
in build_site.py and `FROM` in edit.js). "psd": `name`, `stars`, `types`, `hp`/`def`/`atk`/`spd`, `ability` are printed
on the card image, so the edit pipeline (docs/EDIT_PIPELINE_DESIGN.md §3) bakes them into the PSD and PNG and the
website shows "PSD pending" until then (`name` only when it differs from the printed title the parser reads into
`display_name`; since 2026-10-21 a baked or already-printed name is dropped from the file like any other field). "folder": `type`, `subtype`, `series` come from the card's path. "database": `subseries`, `note`, `aliases`,
`dup_of`, `hidden`, `problems_ok`, `allow_high_stats`, `creatures` live in the file alone (`creatures` overrides the
parser's per-creature rows but is baked nowhere). Whatever the kind, `apply()` saves the pre-edit value in
`parsed_values` so dropping the edit restores it — the PSD's value, the folder's, or `False`/`None` for the
database-only ones (unless `/update_metadata` had set them). The pipeline's verdict rides beside `_by`/`_at` as `_psd`
(`{"state": "baked" | "refused", "for": {field: value}, "psd": "<blob8>", "reason": ...}`; "pending" is never written, the
website computes it from the values; `CardEdits.meta`, `psd_state()`/`set_psd_state()`); `parse()`
keeps every `_`-key out of `cards`. `aliases` is read by `CardEdits.names()` for name lookups, never by `apply()`;
`dup_of` makes a card PNG with no PSD (Stubborn_Queen.png, Gust_of_Trees.png) a duplicate of the named card in the bot
too (`card_data.duplicate_pngs`, 2026-10-23, build_site's rule); on a card with its own PSD the bot ignores it. `/update_metadata subseries:` sets the same field by hand;
a PSD re-parse keeps it with author and series (psd_parser.py:355-361) and a Card Edits.json value wins again at the
next `/update_stats`.

Omitted keys become NaN in the query dataframes — which is why numeric queries naturally skip
cards lacking that stat. `COLUMN_ORDER` (psd_analyzer.py:60-64) fixes export/display order:
aliases, type, ability, hp, def, atk, spd, types, path, timestamp, author, stars, problem, series,
subseries, subtype.

**Timestamp semantics**: the last-commit committer date of the PSD (`_get_remote_timestamp`,
card_data.py:169), looked up once per re-parsed card (the card data build dates a card by its blob's first commit instead, from git). The timestamp does not decide re-parsing; the
blob sha does (`_psd_changed` :1379-1391, called at :1652). A card re-parses if it is new, its path
changed, or its stored `blob_sha` differs from the git tree's sha. Legacy cards with no sha re-parse only
if committed since the newest stored timestamp; otherwise they just get their sha recorded.
`force_update` or naming the card in `cards:` skips the check. Since 2026-10-19 the Python default
is `update_stats(force_update=False)` (repo_traverser.py:59), the same as the slash command (main.py:383);
before that the Python default was True. Every stored timestamp is a commit date. Local mode, which used
file mtimes instead (`use_local_timestamp`) and so re-parsed and archived every card, was archived out of
the bot on 2026-10-20 (kept on branch `archive/local-update-stats`; autocroissant-run-and-operate §7). **Author** comes from TTSCardMaker's `site/Card Authors.txt` (`AUTHORS_PATHS`, `CardAuthors`, since 2026-10-20; in `MDW/` before), the
team's record of who made each card.
- **Format:** two sections of `left = right` lines.
  - `[people]`: git/GitHub name = display name.
  - `[cards]`: Card Name = person. Matched in any case, with `_` and spaces alike.
- **When:** every remote `/update_stats` applies it to every card, not just re-parsed ones (`_apply_authors`).
- **Rule:** a listed card gets its listed person; any other card keeps its author. Either way the name goes through `[people]`.
- **Seed:** the file was seeded from stats.pkl on 2026-10-20 (765 cards + 48 rulebook pages, 8 people, all identity
  mappings), so applying it changed nothing.
- **The website** applies the same rules at build time.

Without a file entry, a new card's author is the *author* (not the committer, changed 2026-10-20) of the first
commit touching its path (the last entry of the per_page=100 commit page, or `.reversed[0]` for a file with 100+ commits). It is fetched only when the card has no author yet. Many
first commits are MichaelJSr's bulk "init" or folder-rename commits, which is why the file exists. Author and series
survive reparses (`parse` copies them from the existing entry). `/update_metadata author:` writes the card's `[cards]`
line in that file (commands/discord_edits.py, branch work/discord-writers), and `/mass_replace author` rewrites its
right-hand sides where they stand (the group renamed, rulebook pages left in place) plus a `[people] old = new` line. `/list_orphans` lists author-less cards.

Pickles (all in the repo root, committed to git except reminder.pkl — commit discipline lives in
autocroissant-change-control):

| File | Shape | Measured 2026-07-11 |
|---|---|---|
| `stats.pkl` | `dict[str name -> CardInfo]` | 813 entries |
| `old_stats.pkl` | `defaultdict[str name -> list[CardInfo]]` | 218 names, 223 archived versions |
| `aliases.pkl` | `dict[str alias -> str target png filename]` | 60 entries, e.g. `'red_bloon' -> 'pod01red_bloon.png'` |
| `reminder.pkl` | reminders, gitignored — not this skill's domain | |

`old_stats.pkl` receives a snapshot when a re-parsed card's PSD blob changed or the card moved and its data changed
(`_history` and `_record_history`, card_data.py; since 2026-10-20 a `cards:`/`force_update` re-parse of an unchanged PSD
archives nothing, since 2026-10-22 neither does a new blob with the same data: `card_history.same_data` compares the
data fields, not the blob, path, dates or parse bookkeeping; the card keeps its start in a `version` record that
/query as_of reads), and when its PSD leaves the repo. The next /update_stats after 2026-10-22 removes the duplicates
stored before once (`card_history.dedupe`: the first of a run stays, a run equal to the card goes into it). Since
2026-10-23 (`card_data.clean_history`) "the same data" is one form per field (a title equal to the name is none, an empty
ability none, the retired type `offense` is `negation`), a field a reading could not know is no difference (a title
before parser 2; stars, types or ability its parse reported a problem with) nor is the series split's move under a parent;
stored versions' series are filled from the next version (items) or moved under that parent, and an earlier reading of
the same commit (one start, no other blob) goes first, never the card before an edit's version (cause `edit` or
`authors`, which starts no earlier than the version it ends). The rule runs one way (`card_history.knows`): the reading
that knew more absorbs the one that knew less, never the reverse, so a commit with a broken ability, star or type layer
keeps the card before as a version. An applied database edit that changes a card's data (Card Edits.json at
/update_stats, a Discord edit's in-memory refresh) keeps the card before as a version too (`record_edits`, cause `edit`,
since 2026-10-23). The same cleanup (2026-10-23, `card_history.recover`, with the history clone and the bot's own full
clone `BOT_CLONE`) puts back what the bot's PICKLE commits held and old_stats lost: a reading whose printed ability words
(any case, icon tokens and punctuation aside), types, stars, stats or card type no neighbouring version prints becomes a
legacy version (`version.restored_from` its PICKLE), one that prints the same only moves the next version's start
earlier (the version knowing all it knew); readings are keyed by name, path and blob (the same bytes in another folder
print another card type: Air Totem a minion, then an item), dated where the PICKLE read them (never a time the file sat
at another path), each restored version's blob that of its start's commit; a name the 2025 PICKLEs held at two paths
keeps its own path's (Auto Croissant, not Mini Auto Croissant's minion); the cards gone before archiving are archived
from their last reading (`backfill`, dated from git, `renamed_to` along exact rename chains, `moved_to` into MDW), and a
renamed card's history starts at the rename (`GitIndex.intro_of`: the old name's archive holds the time before).
Consecutive versions whose PSD changed, PNG did not (from the older reading to the newer one's start) and types alone
differ are one misread (`misreads`), and impossible history values become unknown with an `OUT OF RANGE: ...` problem
keeping the old value (`unknown_values`): stars outside 0..7, a stat above 10 without an allowance (`allows_high`: the
card's allow_high_stats, the retired EXCESSIVE_STAT_EXCLUSIONS names, or no current card), any stat of three digits;
the kept ones above 10 are listed for the owner (`high_values`). **Archived cards** (2026-10-22): `prune_clean_cards(at)`
(card_models.py) moves a card whose PSD left the tree from stats.pkl to the end of its old_stats list, tagged
`archived = {"at": when, "path": last PSD path, "blob": last blob sha or None, "commit": the deleting commit or None}`
(`at`: the run's epoch, replaced by the deleting commit's committer date when `_archive_sources` finds it, and by the
history clone's `clean_history`, 2026-10-23)
plus `"renamed_to"` when the PSD was renamed (same blob, or the commit's rename), `"moved_to"` when it moved into MDW
(retired; set from the history clone, 2026-10-23), and `"backfill": True` on names that left
stats.pkl before archiving existed (tagged by the first full run; one on 2026-10-22, ZOEY!!!!!). It runs only behind
`_can_prune` (a full tree, not truncated, no failed new-name download, some cards). An archived card is a name not in
stats.pkl whose last old_stats version is tagged; read them only through `StatsDatabase.archived()`. stats.pkl still
means current cards, so no query, export, alias or website reader sees them. A PSD back under the name at the tag's
path or with its blob is restored (`restore()` copies the record into stats, tag cleared, no download when the blob is
the same); another PSD under the name is a new card (its own author), the record staying as history. A listed card's
PSD under a new name (its blob; the data build also its PNG's exact rename) is parsed with the old card
(`renamed_from`), so it keeps that card's author, series and subseries, not the renamer's credit or the folder's series;
the data build credits a new card's first author at the oldest path of its exact renames (`GitIndex.origin`). Side-file
lines (Card Authors.txt, Card Edits.json, Card Creation Dates.json) under no listed name are printed each run, and an
archived card's are named in the archive report (move them to the new name or drop them). An alt-art duplicate whose
PNG or original left is archived too (2026-10-23; its tag at its own PNG, `duplicate_of` kept, dated like its original
when the PNG stayed; listed apart, never counted by G2) and back when it is found again. `save()` writes old_stats.pkl, then stats.pkl, then handles each listed name whose last old_stats version is
still tagged and writes old_stats.pkl again: a copy of the listed card (same path and blob; no blob: equal but the tag)
is dropped (a same-blob restore, a kill's leftover), anything else is untagged and stays as history (a restore that was
re-parsed, a /pull mix). So a kill at any point leaves the card listed, archived or both, never unlisted and untagged;
the next run cleans up. `add_history` keeps a tagged copy as is (the save untags it) and replaces a last version equal
but for the path (a kill after an edit or a same-blob move) instead of appending. `/update_metadata` never makes a stats
entry for an archived name (it shows the archived record and refuses edits), and `/pull` puts back into old_stats.pkl a
card its file-by-file merge dropped from both pickles (`rescue_dropped`, committed as PICKLE; a rescued listed card
comes back tagged, so the next run restores it).
Each history version keeps the path it was parsed at (`_update_old_stats_paths`, which rewrote them all to the card's
current path after each /update_stats, was removed 2026-10-23), so git can date it there.

Exports: `/export_cards` writes `stats.csv`/`stats.txt`, `/export_rulebook` writes `rules.txt`,
both to the repo root (gitignored). Rulebook entries are excluded from card exports.

## 7. Querying: name lookup, aliases, and the ability DSL

### 7.1 Name lookup and ambiguous names (`/query`)

`CardRepository` (`commands/query_card.py:35`, singleton `card_repo` at :796) maintains
`git_files`: lowercase png filename → repo path, built from one GitHub trees API call
(`populate_files` :62-103). It indexes EVERY `.png` in TTSCardMaker — including `Types/` icons,
`Auxiliary/Markers`, and MDW card backs; `EXCLUDE_FOLDERS` applies only to stats traversal, not
here. So `/query panda` can legitimately return a type icon.

Lookup normalizes to `name.replace(' ', '_').lower() + '.png'` and fuzzy-matches with difflib
`get_close_matches`, cutoff `match_ratio` (default 0.6, `DEFAULT_MATCH_RATIO` query_card.py:29;
runtime-settable via `/set_ratio`, resets on restart — see autocroissant-config-and-flags).

**Duplicate filenames**: when two repo paths share a basename, both also get prefixed keys
`<top_folder>/<name>.png` (lowercased), and `ambiguous_names` records the options
(:81-102). The bare name STILL resolves — to the first-seen path — and `/query` appends a "try
these instead" list when the hit is ambiguous. Duplicated basenames in the repo today: active.png,
cardback.png, equip.png, field.png, fragile equip.png, nme.png, panda.png (mostly card-vs-Types
icon collisions).

**Aliases** (`aliases.pkl`, 60 today): At `populate_files` time each alias key is inserted into `git_files` as
`<alias>.png` pointing at the target's path, so aliases work everywhere names do. Everything is lowercased on load.
Examples: `abyss_watchers -> the_abyss_watchers.png`, `red_bloon -> pod01red_bloon.png`. Since branch
work/discord-writers (2026-10-22) `/alias` (commands/discord_edits.py) writes `aliases` on the Card Edits.json entry of
the card the PNG belongs to (one commit; target: an existing png, suffix match allowed), key alone deletes, no args
lists. `python -m commands.discord_edits migrate` writes the 60 into Card Edits.json, with the Discord-only series values the
parity re-parse found (596) and the top-level `aliases_moved` key, for one commit; from then on aliases.pkl counts nowhere (query_card `_populate_aliases`, card_data's duplicate
names), and before that `/alias` refuses to change an aliases.pkl alias.

### 7.2 The ability-search DSL (`/query_ability` → `ability_search_engine`, query_card.py:218-304)

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
| `col~value` | string contains, case-insensitive | `type~creature`, `subseries~door` |
| `created<op>date` | creation date at the precision given (YYYY, YYYY-MM or YYYY-MM-DD); undated cards never match | `created>=2025`, `created==2023-10` |

Spaces around an operator are fine (`hp > 5`, `type == creature`, `created >= 2025`; 2026-10-23, as the website's
search). `created` is the card's `site/Card Creation Dates.json` date, or its first upload when the file misses it or
is later (`CardDates.created`, the website's rule); it is in `/export_cards` and `/update_metadata`'s view too.

Numeric-queryable columns: hp, def, atk, spd, stars. String-queryable: type, subtype, series,
subseries, author, path, ability, types (the list is stringified, so `types~undead` works). `prep_dataframes`
adds the `subseries` column even when no card has one, so `subseries==x` is a real (empty) filter, not the
gotcha-4 text fall-through; `series==mario & subseries~door` is the website's own example.

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
   match_ratio` (`_fuzzy_labels`, query_card.py:341-353). It compares against the ENTIRE ability, so it only adds
   near-whole-text matches; short queries effectively rely on the substring pass.
8. `"quoted"` builds `(?:^|\s|$|\b)<escaped>(?:^|\s|$|\b)` (:244) — approximately word-bounded,
   punctuation-tolerant.
9. Rulebook pages are NEVER returned by `/query_ability`; use `/query_rulebook` (plain
   case-insensitive substring over `rulebook_dff`, no DSL — query_card.py:355-379). 48 pages today.
10. One result posts its image URL (a preview, via `get_card_url`); several are listed as `<name> — <url>` lines,
    10 per message (`RESULTS_PER_MESSAGE`, query_card.py:30; since 2026-10-21, one message per result before);
    `howmany:True` returns just the count; `limit` caps output (main.py:291-304).

## 8. When NOT to use this skill

- Running the bot, running `/update_stats`, pushing/pulling pickles, machine handoff →
  **autocroissant-run-and-operate** (it owns the operating rule: since 2026-10-20 `/update_stats` reads
  GitHub only; local mode was archived because local timestamps re-archive every card; history in
  **autocroissant-failure-archaeology** Entries 1, 10.10 and 10.12).
- The icon font itself (token grammar, badges, codepoints, rebuilding it), Types/ icon naming across
  repos, the card website → **ttscardmaker-companion**.
- Diagnosing or FIXING a mis-parsed card, shrinking the exclusion lists, changing injection
  heuristics → **autocroissant-psd-extraction-campaign** (executable plan) with
  **autocroissant-validation-and-qa** as the acceptance gate.
- How to invoke `parse_one.py` / `gap_trace.py` / `dump_psd_layers.py` / `inspect_pickle.py` /
  `diff_stats.py` and read their output → **autocroissant-diagnostics-and-tooling**.
- Constants as CONFIG (defaults, what is runtime-settable) → **autocroissant-config-and-flags**.
- Committing pickles or any schema/code change discipline → **autocroissant-change-control**.
- Queue/threading mechanics behind these commands → **autocroissant-architecture-contract**.

## 9. Provenance and maintenance

Every load-bearing claim above is from `commands/psd_analyzer.py` (since 2026-10-20 also `psd_parser.py`,
`card_models.py`, `card_edits.py`, `repo_traverser.py`, `card_data.py`), `commands/query_card.py`,
`global_config.py`, `main.py`, the TTSCardMaker clone, and live script runs on 2026-07-11.
Line numbers drift — re-find facts with these before relying on cited lines:

| Fact | Re-verify with |
|---|---|
| Constants (ratio, gaps, folders) | `grep -n "TYPE_REGION_RATIO\|MISSPELT_CARD_TYPES\|re_compile(r'\\\\s{3,}')" commands/psd_parser.py; grep -n "EXCLUDE_FOLDERS\|UPDATE_RATE" commands/card_data.py` |
| Classification table | `sed -n '/class CardClassifier/,/class PSDParser/p' commands/psd_parser.py` |
| Exclusion list (21 names) | `sed -n '/class CardValidator/,/def validate/p' commands/psd_parser.py` |
| Injection / prune / sort internals | `grep -n "_inject_type_names\|_prune_type_bboxes\|_sort_by_position\|row_threshold" commands/psd_parser.py` |
| Stat summing + hidden→10 default | `grep -n "tracker.value\|else 10" commands/psd_parser.py` |
| Stars-from-PSD folder list | `grep -n "Auxiliary/Items" commands/psd_parser.py` |
| CardInfo/CardStats mapping | `sed -n '/^class CardStats/,/^class StatsDatabase/p' commands/card_models.py` (approx; or grep `def to_dict`) |
| DSL internals | `grep -n '_stat_pattern\|ability_search_engine\|SequenceMatcher\|duplicated' commands/query_card.py` |
| subseries is database-only | `grep -n subseries commands/card_models.py commands/card_edits.py commands/query_card.py commands/psd_analyzer.py commands/psd_parser.py` (psd_parser: only the preserve block in `parse()`, nothing reads a layer) |
| Edit kinds psd/folder/database and the `_psd` meta (2026-10-20, S1 of docs/EDIT_PIPELINE_DESIGN.md; states baked/refused only, review fix the same day) | `grep -n "KIND = \|PSD_STATES\|def psd_state\|def set_psd_state" commands/card_edits.py` (19 fields: 8 psd, 3 folder, 8 database; `tests/test_card_edits.py::test_kind_table`) |
| Defaults (repo, match ratio) | `grep -n "DEFAULT_REPOSITORY\|DEFAULT_MATCH_RATIO" commands/query_card.py` |
| Clone location | `grep -n LOCAL_DIR_LOC global_config.py` |
| Slash surface + update_stats defaults | `grep -n 'tree.command(name="query\|name="alias\|name="update_stats' main.py` and `sed -n '359,365p' main.py` |
| Folder tree | `ls ~/Desktop/TTSCardMaker ~/Desktop/TTSCardMaker/Auxiliary ~/Desktop/TTSCardMaker/Items` |
| PSD/PNG counts + 813 reconciliation | `find ~/Desktop/TTSCardMaker -name "*.psd" \| wc -l` and same with `! -path "*/MDW/*" ! -path "*/Markers/*"` |
| Pickle counts, distribution, problem cards, newest card | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |
| Alias count | `python3 -c "import pickle;print(len(pickle.load(open('aliases.pkl','rb'))))"` |
| The 7 "creatures" cards | `ls ~/Desktop/TTSCardMaker/Auxiliary/Creatures/*.psd` |
| Golden parses (Mini Doomer / Freezer / Lich King) | `parse_one.py` / `gap_trace.py` with the paths used in sections 3-4 (absolute paths into the clone; outputs include ignorable "Git token found" lines) |
| Types vocabulary (35 remote and, since the 2026-10-19 dotfile fix, 35 local) | `git -C ~/Desktop/TTSCardMaker ls-files Types/ \| grep -vc Stars` vs the `known types` line of any `parse_one.py` run |
| Icon-font parsing hooks (2026-10-19) | `grep -n "^ICON_\|class IconVocabulary\|def _icon_font_spans\|def _read_icon_tokens\|def _flag_misplaced_tokens" commands/psd_parser.py` |
| Exact token spellings (2026-10-20) | `grep -n -A8 "def spellings" commands/psd_parser.py` (words joined by " ", "_", "-" or ""; `lookup` = exact lowercase match) |
| Manifest size / codepoint range | `python3 -c "import json;m=json.load(open('$HOME/Desktop/TTSCardMaker/MDW/Fonts/stoogedings-src/stoogedings.json'))['icons'];p=sorted(int(v['pua'],16) for v in m.values());print(len(m),hex(p[0]),hex(p[-1]))"` (91 0xe000 0xe05a) |
| Emoji count (126) and `_no` rule | `grep -n "NO_SUFFIX\|ICON_FOLDERS" commands/type_emojis.py`; count = Types PNGs × 2 + Stars PNGs |
| README citation | `head -3 ~/Desktop/TTSCardMaker/README.md` |

Maintenance: after any parser or repo change, re-run `inspect_pickle.py` and the three golden
parses and update the numbers here (counts, distribution, exclusion-list sizes, duplicated
basenames) plus the date stamps. If `TYPE_REGION_RATIO`, the gap regex, or the exclusion lists
change, sections 4-5 are stale until rewritten.

2026-10-21: §7.2 gotcha 10 describes the grouped /query_ability output; query_card.py, card_models.py and main.py line numbers re-pointed (review: CardRepository, card_repo, populate_files, manage_alias).

2026-10-21 (later): the parser reads the printed title into `display_name` (PARSER_VERSION 2; §6 table and the pickle-only list, `parser_version`); `EXCESSIVE_STAT_EXCLUSIONS` deleted (§3.4, §5); psd_parser.py line numbers re-pointed.

2026-10-21 (review): frontmatter and the `display_name` row (157 cards; WX-78's "wx-" + image layer "78" reads as the file name; case/spacing-only titles shown as printed, the user's call); `commands/psd_parser.py` citations with the folder prefix re-pointed (`card_name` :337, `CardClassifier` :230-315) and the rest after the WX-78 title fix; repo_traverser.py `update_stats` :472, history append :371-372.

2026-10-21 (S6/S7 gate): main.py, repo_traverser.py and help.py line numbers re-pointed after the /bake_cards block, the bake_lock in update_stats and the /bake_cards help entry; nothing else in this skill changed.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-21 (card check): §5 names `commands/card_check.py` as the second place the validator runs; the `CardValidator` range is :948-1038. Same day: the parse-time list gained `NO STARS FOUND` (:420), the effect layer's (:626) and `SEVERAL CREATURES` (:862), `MISSPELT_CARD_TYPES` re-pointed to line 24, and the card-check line says only those reach the website. Gate: the :626 entry names `_add_problem` and its effect-layer and icon-token callers.

2026-10-21 (D3, idea 14): §4.2 WORDS NOT IN TEXT also covers a `#n` past the last match (orphaned effect layers) and records the corpus and trial-copy counts; §5 says why Shadow Duelist stays excluded; psd_parser.py citations re-pointed (+1 to +3 inside the effect check, +18 after it); the §5 table rows, stale before, re-pointed to the validator (:1008-1050) and MISSPELT TYPE to :566-567.

2026-10-22 (data build): the traversal's citations (EXCLUDE_FOLDERS, the type list, the commit lookup, the history rule, `_update_old_stats_paths`) re-pointed to the uncompiled commands/card_data.py; `_update_old_stats_paths` noted as legacy-mode only. The db/ JSON schemas (`version`, the manifest, the changelog events) are in docs/Campaigns/DATA_BUILD_ACTION.md section 8 until the cutover.
