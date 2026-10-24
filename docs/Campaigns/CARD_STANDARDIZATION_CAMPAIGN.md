# Card PSD edits and standardization campaign (2026-10-20)

**Status:** on hold until the user lifts it (2026-10-20): other developers are editing the PSDs, so nothing writes a PSD or PNG in
TTSCardMaker until the user says so. Plan item 21 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-23"). This file is the single ledger of every PSD
change we intend (section 3); `docs/EDIT_PIPELINE_DESIGN.md`, the rulebook campaign and `docs/NEXT_SESSION_PLAN.md` (item 21) point here;
so does the companion skill's push-guard paragraph (rules 1 and 2, 2026-10-21). Numbers measured 2026-10-20 over the 813 card PSDs of `~/Desktop/TTSCardMaker` (every PSD outside `MDW/`
and `Auxiliary/Markers/`); 651 are creatures, minions, items, auxiliary items and N.M.E. items, the cards a template with stat bars or a
`Name` layer applies to; the other 162 are 48 rulebook pages, 64 status effects, 50 field cards. Evidence in section 8 (`/tmp` scans vanish
on reboot: copy `/tmp/psdmap` and `/tmp/stdcamp` into `~/Documents/Tools/psd-research/` first).

## 1. The hold

- **Waits for the user's go:** every write to a PSD or PNG in TTSCardMaker: the pipeline's live runs (design S5, S8), the six title
  reprints (L8), Sonic's conversion (L3), star folder moves (L14), the `Effect Layers - All Cards.jsx` batch (L7), the token migration
  (L1), rulebook page fixes (L11), any layer rename or effect-block copy however small: 0 changed pixels is still a PSD write.
- **Runs meanwhile:** code (`commands/psd_edit.py`, `commands/psd_regions.py`, `commands/card_bake.py` (new), `tests/`, the fork branches
  and tag), fonts (`MDW/Fonts` additions, L13), JSON (`site/Card Edits.json` entries such as Thunderfury's, L8;
  `MDW/Fonts/stoogedings-src/stoogedings.json` aliases), the guard script (section 5, report-only), and dry runs on PSDs copied to `/tmp`
  (section 4). `~/Desktop/TTSCardMaker` is the live clone.

## 2. Goal and non-goals

**Goal.** Every card PSD matches its template's layer tree and names (`MDW/Templates/`: `Creature_Template.psd`, `2_`/`3_Creature_Template.psd`,
`Item_Template.psd`, `ActiveItem_Template.psd`, `Raid_Boss_Template.psd`, `Status_Effect_Template.psd`; the folder's eighth PSD, `Tile Bars.psd`, is
bars only and sets no card layout; read with the fork 2026-10-20:
`Text[DropShadow,Stroke]` > `Name`, `Ability`; `Stats Bars[Stroke]` > `Darks` > `<Stat> Dark` > exposure `0`..`9`; `Creature Types` /
`Item Types` / `Types`; `Stars`), so four consumers can rely on it: the parser (`commands/psd_parser.py`), the fork renderer and harness
(`~/Documents/Tools/psd-research/score/`), the edit pipeline (`docs/EDIT_PIPELINE_DESIGN.md`; it refuses `unsupported-card`, `stat>10`,
`duplicate`, `styled-word` today), the website and the TTS save (both read the PNG, so a pixel change re-exports it). **Non-goals.** No change
to what a card says or shows except where an entry says so (L7, L8, L9: the owner's call per card); no file renames (L12); no rulings.

## 3. The ledger (one entry per intended change; all on hold)

Checks: **parser gate** = offline parse of all 813 cards equal to the pre-batch parse except the intended fields; **harness** = per-region
score vs the official PNG (`score/score_cards.py`, |diff| > 24); **goldens** = `tests/test_parser_goldens.py` (38); **Photoshop** = the user's Mac.

### L1. Type image layers -> stoogedings tokens (the user's example)
- **What.** Inline icons (visible pixel or smart-object layers named like a `Types/` icon, top at or below the midline; Gasoline
  `Types/Types/text/fire` x3 + `machine` x2) become `[fire]` runs in the stoogedings font inside `Text/Ability`, timing tags become badges,
  the icon layers are hidden. Scope 2 (question 1): the header badges `Creature Types/<type>`, `Item Types/<type>`, `Types/<type>` become
  a header text layer in the font (cards-reference §4.1). **Why.** Parser: literal tokens, no gap guessing; pipeline: an ability edit on
  an icon card is refused (`unsupported-card`); website and Discord already render tokens.
- **How many.** 0 of 813 PSDs use the font (`/tmp/psdmap/corpus.jsonl` fonts). Inline: 42 cards / 71 layers of any kind, 39 cards /
  66 smart objects (`/tmp/stdcamp/scan2.jsonl`: visible layers with pixels below the midline; `agg2.py` prints 54 / 115, of which 44 pixel
  layers on 12 cards are the `Dark`/`Light`/`Machine` stat bars L2 names, not icons; the 2026-10-19 count was 45 / 81 and 61 / 148,
  companion §5). Within the 651: 12 cards / 23 layers; the rest sit on status effects and field cards.
  Header: 688 cards / 875 layers (815 smart objects, 60 pixel). 71 cards have no icon layer at all.
- **Tool.** Photoshop JSX `/private/tmp/fontexplore/std/std.jsx` (scratch; copy it into `MDW/Scripts/` before a reboot) or the fork:
  `set_text` + `set_span_style(font=stoogedings-Regular)`, which needs the face in the layer's `FontSet` first
  (`/tmp/wt-text-engine/src/psd_tools/api/text_edit.py:109-150`, branch feat/text-engine; not in `/tmp/wt-cards` until the S0 merge).
- **Check.** Parser gate (only token spellings differ); 0 `UNKNOWN ICON TOKEN` / `ICON TOKEN NOT IN STOOGEDINGS FONT`; harness ability
  region; Photoshop with ligatures on. **Risk.** Font icons are narrower than the gaps: lines re-break and strand overlay words (Gasoline's
  "Blazing"), so L7 runs first; `[fire]/ [machine]` spacing and token order vs gap order were open (answered 2026-10-21 below); bar layers named
  `Dark`/`Light`/`Machine` are not icons (L2). **Depends.** L7; question 1; the fork's font-run API unless Photoshop.
- **2026-10-21 (question 1 answered; std.jsx).** Inline icons only: the header badges stay images, permanently, so Scope 2 and its 688
  cards leave the batch. `std.jsx` still lives only in `/private/tmp/fontexplore/std/` (gone on a reboot): copy it before one, into
  `MDW/Scripts/` only if the user agrees (a script file, under the hold only with their yes), otherwise under `~/Documents/Tools/psd-research/`.
  The fork's text engine is in cards since a6363a04 (tag cards-20261020), so the `/tmp/wt-cards` caveat above is past.
- **2026-10-21 (spacing, token order).** Spacing = gap injection's (`[fire] / [machine]`); token order = icon order on all 42, tokens safe; 15 cards with no parser gap get their tokens appended, as the parser does; Tainted Lazarus out (header icon); fixed walk `~/Documents/Tools/psd-research/std/std_l1.jsx`, `l1_check.py` 41/41 (companion §5).
- **2026-10-22 (`(Action)` badge, TTSCardMaker f6ce500).** The walk's `TAG` (`std/std.jsx`, `std_l1.jsx`, `l1_check.py`) gained
  `Action`; pre-change copies in `std/pre-action-2026-10-22/`. The std batch `std-batch-2026-10-22` started before it and keeps its own
  regex (`scripts/std_card.jsx:12` `STD_TAG`, `tools/std_driver.py:29` `TAG`), so its cards leave `(Action)` in the body font: re-run
  the tag step on them, or add `|Action` there, before the next batch.
  Its manifest tools also still assume 340: `tools/build_manifest.py:123` says so, and `tools/scan.py` imports `l1_check`
  from `std/`, which now expects `(Action)` in stoogedings. To re-run that batch's scan, point `STD` at `std/pre-action-2026-10-22/`.
  Run the updated walk only after ~/Library/Fonts/stoogedings.ttf has the `(Action)` badge (TTSCardMaker f6ce500); with the old
  font it sets `(Action)` in stoogedings and the text draws as plain letters.
- **2026-10-22 (any-case badges, TTSCardMaker work/badges-any-case).** The font draws a tag in any letter case as its badge
  (`(action)` is (Action)). The walk follows with `std/case_insensitive_tag.patch`, which is not applied yet (std-batch2 was
  running). Apply it with the new font installed, then run `std/case_badge_check.jsx`. If Photoshop ignores the rule, also
  apply `case_canonical_tag_fallback.patch`; it re-cases the text, a PSD + database edit (companion §1.2). The only
  other-case tag on the cards is A P E X P R E D A T O R's `(action)`, which is in neither batch.

### L2. Stat rows to the template layout (45 deviating cards)
- **What.** `Stats Bars/Darks/<Stat> Dark/0..9` exposure digits with masks, one row per stat; `Bars` leaves named `hp`/`def`/`atk`/`spd`.
  **Why.** The pipeline toggles digits only on template rows (`psd_edit.template_row`); the parser's drawn-layer rules in `StatTracker`
  exist only for these cards; `psd_regions.stat_rows` reads the mask boxes.
- **How many.** Of 651: 371 template, 235 no rows (items, N.M.E. items, and `Creatures/Kirby/3 Stars/Computer_Virus.psd`, a creature with
  none, BUGS D1), 45 deviate (`/tmp/psdmap/scripts/classify.py` over `corpus.jsonl`; names in `lists.py`; the design's 234 / 46 counted
  Computer_Virus among the deviants): 24 drawn or value-named layers (7 Isaac tainted cards: Eden, Lost, Isaac, Jacob, Magdalene, Bethany,
  Keeper; Dark_Esau, Skeleton_Mate, Grimora, Puzzled_Cat, Baba...); 7 Clot-style "0" decorations plus a value layer (the 5 Clots,
  False_Knight, Minecraft_Panda); 6 one drawn layer per cell 1-10 (Bulborb Larva, Tainted_Lazarus, Tainted_Eve, Empress Bulblax, Dirty_Bed,
  Blank_Card); 4 several exposure rows per stat (Royal Eradicator Main Cannon, Pix, Sonic, Twin Emperors); 4 exposure rows under odd paths
  (Uumuu, Whispy_Woods, Spinni, Little_Guy). Also 16 PSDs swap the `atk`/`def` bar names and 26 use
  `Healthbar`/`DefemseBar`/`Attackbar`/`Speedbar` (card-structure.md §3). Hazard: bar layers named `Dark`/`Light`/`Machine` (the Isaac
  cards, Sans, Minecraft_Panda, Alex_YIIK, Pandevourer) are `Types/` names; harmless since `_is_stat_layer` runs first
  (`psd_parser.py:526-537`; no stray token in `site/dist/data/cards.json`), but scanners must skip them and the rewrite renames them.
- **Tool.** A fork script (new) adding the template's `Darks` rows from `Creature_Template.psd` and setting the digit; bars painted into the
  art (Grimora, Skeleton_Mate) or drawn as pixel cells: Photoshop per card. **Check.** Parser gate (hp/def/atk/spd unchanged); harness stat
  rows; goldens; Photoshop on the pixel-bar cards. **Risk, depends.** A value silently changed by a rewrite; a look change where drawn bars
  differ (question 3); multi-creature cards (question 2); until then the pipeline refuses them (`unsupported-card`).
- **2026-10-21 (questions 2 and 3 answered).** The 37 drawn or pixel-bar cards are rebuilt as template digit rows (the look change is
  accepted). Multi-creature cards (Pix, Tainted Lazarus, Twin Emperors) and raid bosses get no templates of their own: they stay
  deviations the pipeline refuses (`unsupported-card`).
- **2026-10-21 (owner: drawn value differs from the layer name).** Tainted_Isaac `hp` is drawn as 6 hearts with the lit layer named `5`;
  Tainted_Eden `spd` as 4 with it named `7`. The parser reads the name, so the rebuild would print the name's value and the parse gate cannot
  see it; `l2_template_rows.py` counts the lit cells and leaves these two until the owner picks the stat. Alex_YIIK's `Dark` is an inline
  icon (L1), not a bar layer: nothing to rename.

### L3. Stats above 10 -> top row 10 + bottom row N-10 (design section 14, question 3: answered)
- **What.** Royal Eradicator's layout is canonical: `Stats Bars/Darks/atk dark/Top/0..9` all hidden (= 10), `Bottom/<N-10>` visible. Sonic
  (`Creatures/Other/4 Stars/Sonic.psd`, spd 20) is converted once: today `Stats Bars/spd` holds half-height bars `spd top` (y 1569-1620)
  and `spd bottom` (1620-1668) with `spd dark/1` and `/2` all hidden, plus the template `Spd Dark` whose `9` is renamed `20` (visible,
  0 pixels). Clean-ups: Royal Eradicator's `atk dark/Top/10` (visible at opacity 0; hidden digits already read 10) deleted; Sonic's `20`
  renamed `9` and hidden. **Why.** `EXCESSIVE_STAT_EXCLUSIONS` went on 2026-10-21 (Sonic and the Cannon carry `allow_high_stats` in Card Edits.json
  meanwhile); `stat>10` becomes a two-row toggle.
- **How many.** 2 cards (`/tmp/psdmap/card_Sonic.txt`, `card_Royal_Eradicator_Main_Cannon.txt`). **Tool.** Fork script or Photoshop,
  one-off; then drop the list and refresh `tests/make_goldens.py`. **Check.** `parse_one.py`: Sonic spd 20, Royal Eradicator atk 12, problems
  NONE before and after; clean-ups PNG byte-equal; Sonic within the stat-row bar (its pixel-art bars stay; only the `Darks` groups move).
- **2026-10-21 (pipeline, design §7 idea 13).** The Top `10` is no stray: the parser skips a row with no visible digit, so without it the
  Cannon reads 2 (measured); `psd_edit.template_rows` requires it in Top (visible, opacity 0) and takes one in Bottom for 20, so Sonic's
  conversion adds a marker to each row and the "deleted" clean-up above is withdrawn. The Cannon's ColorOverlay + BevelEmboss digits
  miss the render bar (5-7% bad): its toggles refuse `parity` until the fork paints bevels right.

### L4. Effect blocks on every digit of a styled row (plan item 12; 107 cards)
- **What.** In each `<Stat> Dark` row where a digit carries an enabled effect, the look must survive a toggle: either all ten digits get the
  same `lfx2` block, or the pipeline moves the block to the new digit and clears the old (design §7 idea 12); question 5 decides
  (Mettaton_EX `Atk Dark/6` is styled while `7` is visible: a stale toggle; the visible digit gets it). **Why.** The pipeline's `parity`
  refuses a toggle when the old or new digit carries an enabled effect (design step 4); renderer: class A.
- **How many.** 107 cards / 396 rows (`/tmp/psdmap/aggregate.txt` line 29); 452 row-effect pairs, a row with two kinds counted twice:
  GradientOverlay 379, InnerShadow 45, ColorOverlay 12, InnerGlow 8, BevelEmboss 8. The visible digit carries it on 87 cards / 326 layers
  (RENDER_STATUS.md class A; critic scan 101 / 350).
- **Tool.** Fork tagged-block copy for `lfx2` (design S0; effects are a read-only proxy, `api/effects.py:5`) or Photoshop `copyEffects` /
  `pasteEffects` per layer (companion §4). **Check.** Harness per stat cell on Mettaton_EX before and after (also settles whether the fork
  and Photoshop agree the effect's shape is the mask); parse and PNG unchanged; blind until the renderer paints effects on empty-bbox
  adjustment layers (RENDER_STATUS critic cause 1, 41% of all bad pixels). **Risk, depends.** Copy to all ten vs move on toggle: question 5.
- **2026-10-21 (pipeline, design §7 idea 12): move on toggle, in code.** `set_stat` moves the hidden digit's `lfx2`/`lrFX` onto the shown
  one (gate `bytes` checks the same bytes left and landed); Mettaton_EX hp 5 -> 8 passes every gate (parity mean 0.90, 0% bad), so no
  PSD needs all ten digits styled; stale hidden styled digits (Mettaton Atk 6) are left alone. Toggles whose region takes in lit cells
  refused at the baseline until 2026-10-22 (mean 1.03-1.04 vs the 1.0 bar: the pixel bars' GradientOverlay, not the digits);
  the bar is mean 2.0 since then (owner).

### L5. Ability layer naming, visibility and blank content
- **What.** Exactly one visible text layer named `Ability` directly under top-level `Text`, with text. **Why.** The parser reads every
  layer named `ability` (`psd_parser.py:559-584`); since 2026-10-23 the pipeline takes the one visible `Ability` wherever it is
  (`psd_edit.ability_layer`), so only the hidden, several and blank ones still need work.
- **How many.** Of 651: 605 template; 34 one visible `Ability` elsewhere (top-level `Ability` on 16, `Types/Text/Ability` on 8, nested
  groups; paths in `lists.py`); 7 hidden (Abyss_Watcher, Bulborb Larva, Shadow_Duelist, Warrior_Dee, Butterfly, Empress Bulblax,
  Apex_Predator; 4 re-laid out in separately styled layers); 5 several visible (Lit_Bob-omb 2, Tainted_Lazarus 2, Miracle_Matter 5,
  Twin_Emperors 5, Warcraftian_Druid 2); 20 blank: 17 visible with no text (`lists.py`), Galacta_Warrior's visible two spaces and 2 hidden
  (Abyss_Watcher, Warrior_Dee; `corpus.jsonl` `text_len`); all 20 are in `ABILITY_EXCLUSIONS` (21 names, `psd_parser.py:950-972`; the user
  added Anubisath_Guardian, Qiraji_Soldier and Silithid on 2026-10-21, D1).
- **Tool.** Fork script: rename and move the layer under `Text` (layer move API on the fork: to confirm), else Photoshop. Blank content:
  the owner fills it or certifies it (then an empty layer is the standard for ability-less minions and `ABILITY_EXCLUSIONS` shrinks to the
  certified list; Shadow Duelist is stale there already, D3).
- **Check.** Parser gate (ability text identical); 0 `NO ABILITY LAYER`; PNG byte-equal for a rename. **Risk.** Moving a layer under
  `Text` adds the group's Stroke and DropShadow: pixels change (that is why 50 of 63 overlays sit outside `Text`, cards-reference §4.2);
  per card, template look or rename only (question 4). **Depends.** Question 2 for the 5 several-Ability cards.
- **2026-10-21 (questions 2 and 4 answered).** The 34 out-of-`Text` Ability layers move under `Text` (where the group carries effects the
  ability takes them: pixels change, the PNG is re-exported). The 5 several-Ability cards stay deviations the pipeline refuses.

### L6. Name layer
- **What.** A visible `Text/Name` text layer holding the printed title; the parser reads it into `display_name` since 2026-10-21 (design question 1; `psd_parser._printed_title`).
  **Why.** 22 of the 29 live name edits become redundant. Since 2026-10-23 the pipeline bakes whichever layer the parser
  reads the title from (`psd_edit.name_layer`), so a title named after the card or a `Name` outside `Text` needs no rename;
  nor, since title-own-text (2026-10-23), the one title layer named after its own text (Papyrus, Sans).
- **How many.** Of 651: 552 visible `Name` (506 under `Text`, 46 elsewhere; paths in `lists.py`); 91 titled by a layer named after the
  card (153 over all 813, status effects included); 3 hidden (Sonic, whose title is the pixel layer `Layer 0 copy`; Baba; Man_Serpent_Shield;
  plus the field card E._Gadd's_Garage); 5 none or rasterized (ABBI, Tyler, Minecraft_Panda, Pandanoko, Holy_Card); 16 with two or more
  visible `Name` layers (17 over all 813 with the field card The_Gates_of_Ahn'Qiraj; `aggregate.txt`, `corpus.jsonl`).
- **Titles vs file names (2026-10-21 parse).** 157 of 813 differ: 94 by letter case or spacing only, 63 otherwise. All are shown
  as printed (the user, 2026-10-21). The odd ones (The Bagel "The Bagel Or... is it", Leeroy Jenkins "LEEEEEEEEEROY", Blockade "The Wall By
  Ech", Sir Grodus "Grodus", The Chicken "Chicken", Rasta Beast "Pasta Beast", Indoctrination "Indoctrinati0n", Kyle) are INTENTIONAL
  in-jokes: NOT standardization work, never retitle them. Six case-only titles look like typos (SPringtrap, Dark MAtter, MEtal General,
  MAsked Dedede, StaR DREAM, Zero OVerride): ask the user card by card before any reprint; a name edit reprints a title through the
  pipeline. Sans and Papyrus title their layers after the print, not `Name` (read and baked since 2026-10-23: the
  parser's last title rule).
- **Tool.** Fork: `layer.name` set, 0 pixels; rasterized titles: Photoshop retype (pixels change; question 6) or leave them to
  `display_name` from `Card Edits.json`. **Check.** Parser reads `Text/Name` == the printed title (after the question-1 code); PNG
  byte-equal. **Depends.** `Effect Layers.jsx` keys FX layers on the text layer's name (`<layer> FX:`): rename before L7 or rerun Update all.
- **2026-10-21 (question 6 answered).** The six rasterized or pixel titles (Sonic, ABBI, Tyler, Minecraft_Panda, Pandanoko, Holy_Card) are
  retyped as text in the batch (pixels change). The in-joke titles above stay as printed; the six case-typo titles wait for the user, card by card.

### L7. Status words: off-chart colours and the 52 overlay word layers -> chart-styled FX layers (plan item 14; question 6: answered)
- **What.** Each status word in `Text/Ability` takes the chart's fill (`MDW/Croissant's Special Folder/Satus Effect Colors.psd`: `Buffs` 12
  + `Debuffs` 54 text layers, 15 styled); styled words get a pixel layer `Ability FX: Word | Word` above `Text` (`Effect Layers.jsx`
  convention, `FX_TAG = " FX: "` line 18, style pasted from the chart layer); the hand-made overlay text layers are converted and hidden.
  **Why.** Fill-only recolouring of a styled word is not Photoshop parity (`styled-word`); overlays go stale on every ability edit and
  strand on reflow (L1); the parser already checks FX layers (`_check_effect_layer`, coverage >= 0.95).
- **How many.** 52 overlay layers (`aggregate.txt`; companion survey 53 on 40 cards, 27 off-colour); 0 ` FX: ` layers today; words off the
  swatch about 5% of the time (companion §6: Smoldering 20/23); `AUTO_STYLES` would add FX layers to 3 more cards (Hexed, Panseida, Credit Card).
- **Tool.** `MDW/Scripts/Effect Layers - All Cards.jsx` now (Photoshop; trial on copies of the 40 overlay cards: all 40 parse identically,
  at most 7,679 pixels change per card; the 13-card batch ran twice, the second run changed 0); the fork port later (Task 3c, design S6).
- **Check.** Parser gate; 0 `EFFECT LAYER *` problems; harness ability region; Photoshop: Single-line composer kept (L10). **Risk.** The JSX
  deletes converted overlays and the answer says hide: set that first; non-sRGB cards are converted to sRGB (0.9% of pixels differ by
  rounding); the 15 styles must be loaded from `MDW/Debuff And Attributes Layer Effects.asl`. **Depends.** Before L1; L6 renames first.
- **2026-10-21 (pipeline, design §7 idea 14): FX regeneration in code.** `psd_edit.set_text_layer` rebuilds the ability's `Ability FX: …`
  layers on every ability edit (one per styled chart word, the chart's `lfx2`/`lrFX` copied, directly above `Text`; against the JSX cuts
  IoU 0.966 Gasoline / 0.937 Cannon, validator coverage 1.0) and HIDES the overlay word layers it replaces. Census of the 53 overlay
  layers on 40 cards (`/tmp/i2_census.py`, fork venv): 37 name a styled chart status and go hidden with that card's first baked ability
  edit (30 cards wholly, Gasoline and the Cannon among them; Monkey_D_Goofy and Morpho_Knight also carry refusing ones); 16 layers on
  10 cards refuse `unsupported-card` (Smoldering x3, Burning, Chilled, Despair, +2 DEF, +2/4/6 ATK, Attacks inflict 2T, 2T, 1, 1st: Slime /
  2nd: Puppet / 3rd: Magician, and "1 Become Back Money" x2 whose stack count keeps them off the chart name): those stay Photoshop work
  (Convert overlays keeps the overlay's own style) or lose the overlay by hand. No PSD touched; the pipeline lands nothing under the hold.
- **Decision (the user, 2026-10-21):** the 16 overlay word layers on 10 cards without a chart style (Smoldering, Burning, Chilled, Despair,
  +2 DEF, +2/4/6 ATK, 1st: Slime, 1 Become Back Money) stay refused by the pipeline (`unsupported-card`); no chart additions; the
  status-word recolouring covers every chart status word in an edited ability, edited or not (design question 6 confirmed).

### L8. Six title reprints; Thunderfury's entry (design question 7: answered)
- **What.** `Text/Name` set to the website value: Clot_White "White CLot" -> "White Clot"; PoD08-PoD12 "M.O.A.B" / "D.D.T" / "B.F.B" /
  "Z.O.M.G" / "B.A.D" gain their trailing dot. Thunderfury keeps its brackets: its `Card Edits.json` entry is aligned to the print (JSON,
  may run now; redundant after question 1). **Why.** The pipeline's first push-triggered run would do exactly this. **How many.** 6 PSDs,
  1 JSON entry (design §2: fork read-back of the 7 `Text/Name` layers; live file 29 name edits).
- **Tool.** The edit pipeline (`set_text` + `rasterize` on `Text/Name`, design S3/S5), `--dry-run` on `/tmp` copies first. **Check.**
  Pipeline gates (`bytes`: one record + its channels; `region`: bbox union + the `Text` group's 7 px Stroke; `reparse`; `font` exact);
  Photoshop: no missing-font or "update text layers" prompt. **Depends.** S0-S5; `card-edits.yml`'s push trigger stays off until fixed.

### L9. Text group effects: DropShadow + Stroke on top-level `Text`
- **What.** `Text` carries `DropShadow` + `Stroke` as in 6 of the 7 templates (`Status_Effect_Template.psd` has neither). **Why.** `_fx_pad` and the
  name-edit region assume it; the look is the team's standard. **How many.** 615 of 813 have both (`aggregate.txt`); of the 651: 566 both,
  37 Stroke only (33 items: Heal 19, Counter 5, Equip 4, Support 2, Active, Attack, Infliction 1 each; 4 creatures: Pure_Vessel, Doc,
  The_Master, Mari), 12 DropShadow only, 25 none, 11 no top-level `Text` group (lists in section 8). **Tool.**
  Fork `lfx2` copy from the template (design S0) or Photoshop paste style. **Check.** Parse unchanged; the PNG changes by design (an
  outline): per-card owner review. **Risk.** A card whose look depends on no outline (Flame_Trap, the food items): question 4.
- **2026-10-21 (question 4 answered).** Not forced: the 85 holdouts (37 + 12 + 25 + 11) keep their look and L9 leaves the batch.

### L10. Composer: Single-line everywhere (an invariant, not a change)
- 769 of 769 visible `Ability` and 622 of 622 visible `Name` layers have `EveryLineComposer` false (`/tmp/stdcamp/scan2.jsonl`, all 813
  PSDs). No edit; the guard checks it and every text-writing tool keeps it (`Effect Layers.jsx` `layoutMode`: Photoshop resets it on `setd`,
  17 of 40 cards re-broke before; the fork's `set_text` keeps paragraph sheets).

### L11. Rulebook pages (48 PSDs; the decision log rules, this ledger only lists the PSD edits)
- `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` §3: H1 six hidden text layers on four pages (09_TakingYourTurn x2, 11_Abilities x2,
  12_Reactions2, 22_Negation) to delete (hidden text is parsed); H2 off-canvas and overflow boxes (11_Abilities6; 12, 14, 20); V1 the stray
  box on 16_The_Field (a decision); X2 icons after the page number on 6 pages; C1-C6 text once decided; BUGS D2: the layer named `tornado`
  on 20_Creature_Types is supposed to be `wind` (user, 2026-10-21): rename it. Tool: Photoshop with Consolas active (companion §3); later plan item 15. Check: that campaign's G3 (rules.json diff).
- 2026-10-21: the cards moved to the cooldown tags ((nCD), (a/bCD), (nGCD), (a/bGCD), (nU); TTSCardMaker df02b3b..7cb4c86); 02_Abbreviations, 11_Abilities2, 11_Abilities4 and 22_Negation still show the old tags: the team updates them (that campaign's U1).

### L12. Files beside the PSDs (no edit wanted; recorded so nobody "fixes" them)
- 813 PSDs, 821 PNGs (`git ls-files`). 5 case-only stem mismatches (Dark_matter.psd/Dark_Matter.png, grand_doomer/Grand_Doomer, hr-h/HR-H,
  ABBI/Abbi, omori/Omori): leave (lookups are case-insensitive; a rename breaks aliases, authors, edits keys, `old_stats`, the TTS save).
  6 numbered duplicate PNGs over 5 PSDs (A_Snack2/3, All_Purpose_Medical_Kit2, Regeneration_Potion2, Watermelon2, NME_Stock2): the standard
  is one PSD whose hidden title layers are toggled per export, so an edit re-exports every PNG. 2 PNGs with no PSD (Stubborn_Queen,
  Gust_of_Trees, commit 3a268ca 2026-10-19): ask their author for the PSDs.

### L13. Fonts
- 10 faces resolve only from the Mac's platform directories (112 layers: Century, ComicSansMS, TimesNewRomanPSMT, Impact, ArialMT,
  CenturyGothic-Bold, MicrosoftSansSerif, ArialNarrow-Bold, BookAntiqua, ComicSansMS-Bold): **done**, committed to `MDW/Fonts` with
  Papyrus.ttc (TTSCardMaker fb37995 + ec3a6f1, 12 files, 5.4 MB; the engine resolves 136 of 144 names from the repo alone). 6 faces missing everywhere (30 layers: JavaneseText, MyriadArabic-Regular, BabaUI, OCRAExtended, Pristina-Regular,
  Elephant-Regular) and 2 near-misses (Papyrus-Regular, Kung-Fu-Panda): find the files (an Office PC) or re-set those runs to an available
  face, a PSD edit (question 7); until then text edits on those layers refuse `font` (TRIAGE.md, section 8).
- **2026-10-21 (question 7 answered).** The missing fonts are the user's. They added Elephant, OCR A Extended, Pristina, Javanese Text and
  the Windows Papyrus (TTSCardMaker a88b232); Papyrus-Regular and Kung-Fu-Panda resolve through `FONT_ALIASES`; MyriadArabic-Regular sits
  only on a leading byte-order-mark run (ignored). Only BabaUI (`Baba.psd`) was still missing, and the user deferred it.
- **2026-10-23: done.** The owner added BabaUI (`MDW/Fonts/Baba UI Regular.ttf`, his file, TTSCardMaker b03b9975), so no font is
  missing: `font_check --all` over the clone reads 812 card PSDs against 274 files / 282 faces and finds nothing. R7's BabaUI
  waiver is gone. The same commit named every font file after the font's full name.

### L14. Stars
- Creatures, items, field cards: stars are the `N Stars` folder, so a stars edit moves PSD + PNG (aliases, `old_stats`, authors and the
  TTS save follow): the pipeline does it after round 1 (design question 4, S9). Minions, auxiliary items, N.M.E. items: star smart objects
  under a group named `Stars`; 2 cards read 0 (God Tamer: five "Delicate Flower" layers; Bubble: a starfish) and need them grouped
  (cards-reference §2). Check: `stars` unchanged except those two.

### L15. Renderer-blocked classes that only a PSD change can fix (RENDER_STATUS.md critic pass)
- Class B: Gaussian Blur smart filters whose mask is not in the file (62 cards, 47 B-only; worst Toriel, Yin_Yarn, Vivian): not
  reconstructible from the file, so rasterize the filtered art in Photoshop (look unchanged) or exclude them from the 2% target. Class D:
  several instances of one effect kind on a layer (9 cards / 34 layers: Jevil, Massive_Moss_Charger, Moss_Charger, Marx, Wham_Bam_Jewel,
  Necrodeus, Sans, Morpho_Knight): renderer fix first; dropping the duplicates is optional. Check: harness per card.
- **2026-10-21 (round 4 research): the class B statement above is withdrawn.** The mask is in the file: the document's `FEid`/`FMsk`
  blocks hold each smart filter's pre-filter source and its painted mask, so the fork can blend the two (branch `fix/smart-filter-mask`,
  planned; `~/Documents/Tools/psd-research/RENDER_ROUND4_PLAN.md` §2.1). 13 cards carry an active painted mask (Yin_Yarn, Toriel,
  Dragon_Mount, Vivian, Hive_Knight, Knight_Lost_A_Lot, Lorian, HR-D3, Defcon_Panda, Magnet-shroom, Flame_Bomb, Lothric, Indoctrination);
  48 other class B cards have white masks (no change). Class B becomes a renderer fix: once the branch lands, neither the Photoshop
  rasterizing nor the exclusion above is needed. The_Collector and Holy_Blade are stale exports (L16).

### L16. Stale official exports (RENDER_STATUS.md class H; 2026-10-21)
- The team re-exports Skullord, Restrained, Blood_Bomb, Dark_Mind_Eye, Asteroid_Field, PoD04Zebra_Bloon, The_Collector, Holy_Blade: their
  PNGs predate the PSD edits (Restrained's PNG reads "-1 Action", its PSD "(BOT) -1 Action"), so no renderer change can match them.
  A PNG write: on hold. Check: harness per card after the re-export (`RENDER_ROUND4_PLAN.md` §4).

## 4. Order and batching (one batch when the hold lifts)

1. **Before the day (allowed now).** Code S0-S4 (design §13) and `check_cards.py` (section 5); fonts (L13, done 2026-10-23); Thunderfury's entry (L8); the
   15 styles loaded in Photoshop. The rehearsal: copy every PSD an entry names (about 110 cards; 688 more with L1's header scope) to
   `/tmp/rehearsal/`, run each tool there in the day's order, then the three gates on the copies (parse of all 813 vs the pre-batch parse,
   harness regions vs `score/results/cards_v1_icc.json`, goldens) and write the expected per-card diff list.
   **2026-10-21: rehearsed** on copies of the manifest's 425 PSDs at 9f0c9ec: gates green or explained, 26-row diff list, timings and
   the day's needs in `STANDARDIZATION_REHEARSAL.md` (redo for the 115 cards a teammate re-pushed during it).
2. **The day.** Confirm no PSD commit in the last 30 min (the `recent` rule: never race an author), pull main. Photoshop batches first,
   since they re-export PNGs and reflow text: L7 (`Effect Layers - All Cards.jsx`), L1 migration, L11 pages, L5 blank content, the
   pixel-changing cases of L5/L6/L9. Then the fork scripts for 0-pixel structure: L2 rows and renames, L3 clean-ups, L5/L6 renames, L4 effect
   copies. Then the pipeline live run (L8 only), push trigger on. One commit per entry (PSD + PNG), e.g. `Standardize: L7 effect layers
   (40 cards)`; never pickles, `MDW/` templates or `TTS/` in those commits.
3. **Gates.** Parser gate on the live tree equals the rehearsal's diff list; harness: no region worse than before, touched regions within
   the bar; goldens green (refresh only cards whose change was intended, `tests/make_goldens.py`); the user's Photoshop check on 10 cards,
   one per entry kind. Any gate red: revert that entry's commit (PSDs do not merge; a revert is the only undo).
4. **After.** Jetson `/update_stats` (every touched PSD re-parses by blob sha; its old card goes to `old_stats` once); `diff_stats.py` lists
   exactly the touched cards with exactly the intended fields; `/push` lands the PICKLE commit; `site.yml` and `tts-save.yml` run. Then the
   code clean-ups: `EXCESSIVE_STAT_EXCLUSIONS` out, `ABILITY_EXCLUSIONS` to the certified list, the pipeline's refusals narrowed, this
   file's entries marked done with their commit shas.

## 5. The future guard: an Action on push that reports (then fixes)

Each rule is one TTSCardMaker Action on push that runs a bot module on the pushed PSDs by blob sha, writes the job summary, comments on the
commit and posts in the standards channel on Discord (`DISCORD_STANDARDS_WEBHOOK`; only the font check pings the card makers' role,
owner 2026-10-23; `DISCORD_FONT_WEBHOOK` is deleted), and never reverts or rejects.

- **Rule 1, the font check** (LIVE since 2026-10-21: TTSCardMaker f5a0e50, AutoCroissant ae1d2c5, secret set):
  `.github/workflows/font-check.yml` runs `commands/font_check.py` on every push that touches a PSD or `MDW/Fonts/`: the fonts a card asks
  for that `MDW/Fonts` lacks, per author and font (none since BabaUI was added 2026-10-23, TTSCardMaker b03b9975; L13).
- **Rule 2, the card check** (LANDED 2026-10-21: AutoCroissant df4a439 `commands/card_check.py`, TTSCardMaker a3451c8; the first
  whole-repo dispatch, run 39690899334, succeeded, job summary only):
  `.github/workflows/card-check.yml` runs `commands/card_check.py` (uncompiled; `python -m commands.card_check --help`) on every push that
  touches a PSD, `Types/` or `site/Card Edits.json`: for each pushed PSD the notes `/update_stats` would print (`NO STARS FOUND`,
  `MISSPELT TYPE`, `SEVERAL CREATURES`, `HP/DEF/ATK/SPD NOT FOUND`, `STATS TOO HIGH`, `ABILITY TEXT NOT FOUND`, `UNKNOWN TYPE`; the
  parser's own, the first three and `NO ABILITY LAYER`, are also the website's Parser notes, the validator's are printed only), after the
  card's edits-file entry (`problems_ok`, `allow_high_stats`, `creatures` ...) and `ABILITY_EXCLUSIONS`; `Types/`, the icon manifest and
  the edits file come from the tree at the pushed commit. Report-only, exit 0 always (an unreadable PSD or a failed API call is one error
  line, never a red push); one Discord line per card, "<login> has pushed <card> with: <note>; <note>."; a dispatch with `all` reads every
  card into the job summary. Today's notes (the clone at e4e0a58): Bubble and God Tamer `NO STARS FOUND`, 20 Creature Types
  `MISSPELT TYPE: tornado`, Tainted Lazarus, Pix and Twin Emperors `SEVERAL CREATURES`, Computer Virus `HP/DEF/ATK/SPD NOT FOUND`;
  `tests/test_card_check.py` pins them. Operations: autocroissant-run-and-operate §7.
- **Rules 3+, the push guards** (BUILT 2026-10-22 on AutoCroissant `work/push-guards` and TTSCardMaker `work/push-guards`, not
  landed; design and build notes `docs/Campaigns/PUSH_GUARDS.md`): `.github/workflows/card-guards.yml` runs
  `commands/card_guards.py` (`png_export.py`: a PSD pushed without its PNG gets the PNG the bake would make, the fork's
  render of the changed layers merged into the old PNG and gated, the stored composite only a cross-check (owner's ruling
  2026-10-22); `card_standards.py`: this ledger's rules R1-R11 on each pushed PSD (R3 and R4 without the template layout, dropped by the owner on 2026-10-23: several visible Ability layers, a hidden one with words the parser reads nowhere, a hidden or rasterized Name layer; kept unattended, for the owner to confirm), R9 Txt2 sync fixed with byte gates
  (R11's exact-name rename is built but reported until the owner says yes), the rest, fonts (R7) included, reported once
  per push in the standards channel after the 30-minute quiet window, `DISCORD_STANDARDS_WEBHOOK`, no role ping). Off until the repo variable
  `CARD_GUARDS` is `report`, then `on`; dispatches default to dry runs. At 6e50c3ee the rules reproduce this ledger's counts
  (PUSH_GUARDS.md section 16).
- **Later rules, deviations by class** (`site/build/check_cards.py` proposed beside the rulebook campaign's `check_rules.py`, or a bot
  module in the card check's shape): for each PSD whose blob changed on the push, report every deviation by class, the classes being this
  ledger's (the scratch `corpus_scan.py` + `aggregate.py` + `classify.py` ported into the bot next to `psd_edit.classify` and
  `psd_regions`, plus L10's composer and L2's bar-name checks), against a baseline that starts as today's inventory and shrinks to empty.
  Plain script by hand first, then its own Action (a red check must not stop the site). Report-only for a month.
- Fixing on push comes later and only for 0-pixel classes proven on every card of the batch (renames, effect copies), only when the PSD
  has been quiet about 30 min (Task 3c's rule), skipping its own commits, never a rulebook page, never while a hold stands. Pixel-changing
  classes stay report-only: PSDs cannot be merged and an author may still be editing.

## 6. Questions for the user (all nine answered 2026-10-21)

1. L1 scope: inline icons only, or the header type badges (688 cards) in the font as well?
   2026-10-21: inline icons only go to the font; the header type badges stay images, permanently (L1).
2. Multi-creature cards (Pix, Tainted Lazarus, Twin Emperors; the 5 several-Ability cards) and raid bosses: their own templates or deviations?
   2026-10-21: deviations, no extra templates; the pipeline keeps refusing them (`unsupported-card`).
3. The 37 drawn or pixel-bar cards (L2): rebuild as exposure rows (small look change) or allow as a variant the parser already reads?
   2026-10-21: rebuilt as template digit rows (L2).
4. The 85 `Text` outline holdouts (L9) and the 34 out-of-`Text` abilities (L5): apply the template outline (pixels change) or allow?
   2026-10-21: no forced `Text` outline (L9); the 34 out-of-`Text` Ability layers move under `Text` (L5).
5. Digit effects (L4): copy the block to all ten digits, or let the pipeline move it on toggle?
   2026-10-21: the pipeline moves it on toggle (L4; design §7 idea 12, in code).
6. Rasterized or pixel titles (Sonic and 5 more): retype as text (pixels change) or keep, with `display_name` from the website?
   2026-10-21: retyped as text, part of the batch (L6). The odd in-joke titles are never standardized; the six case-typo titles are
   asked card by card before any reprint.
7. The 6 fonts missing everywhere (30 layers): hunt the files, or re-set those runs to a face in `MDW/Fonts`?
   2026-10-21: the fonts are the user's; since TTSCardMaker a88b232 only BabaUI (`Baba.psd`) was missing, and the user deferred it (L13).
   2026-10-23: BabaUI added (TTSCardMaker b03b9975, the owner's file); nothing is missing now.
8. The guard: report-only for how long, and may it ever fix on push?
   2026-10-21: it reports first (no fixed term) and may later fix 0-pixel deviations (layer names, grouping) on push, under section 5's
   conditions.
9. One batch archives every touched card once in `old_stats.pkl` (about 110 cards, about 800 with the header scope): acceptable?
   2026-10-21: accepted (about 110 cards: the header scope is out).

## 7. Known wrong paths (fenced off)

- Editing any PSD or PNG while the hold stands, "small" renames included; dry runs on copies under `/tmp` only, never in the clone.
- Renaming PSD or PNG files to fix case or stems (L12): aliases, authors, edits keys, `old_stats` and the TTS save all key on the name.
- Fill-only recolouring of a styled status word (15 of 66): not Photoshop parity; an FX layer or a refusal.
- Sending paragraph styles back with `setd` without `layoutMode`: the composer resets to Every-line and lines re-break (L10).
- Collapsing gap whitespace in ability text (archaeology Entry 2): L1 replaces gaps with tokens, never collapses them.
- Renaming or deleting a `Types/` PNG to fix a token name: add an alias (companion §1.7). Hiding a rulebook text layer instead of
  deleting it: hidden text is parsed (rulebook campaign H1).
- `force=True` baselines and `layer.composite()` deltas (design §15); patching `stats.pkl` by hand after the batch (`/update_stats` +
  `/push`); letting an Action or an agent write a PSD while an author may be editing, or before the user lifts the hold.

## 8. Provenance (2026-10-20, read-only) and lists

Persistent copies of the `/tmp` evidence: `~/Documents/Tools/psd-research/evidence/psdmap-2026-10-20/` and `stdcamp-2026-10-20/`
(the `/tmp/psdmap` and `/tmp/stdcamp` paths below).

| Fact | Evidence |
|---|---|
| 813 PSDs; 651 bar-eligible; stat-row classes 371/235/45 and their lists; 107 effect cards / 396 rows (452 row-effect pairs); Ability 605/34/5/7/20 (`lists.py` prints the 17 visible with no text; `corpus.jsonl` `text_len` adds Galacta_Warrior and the 2 hidden); Name 552/91/3/5; Text group 566/37/12/25/11; 0 FX layers; 0 stoogedings | `/tmp/psdmap/corpus.jsonl` (`scripts/corpus_scan.py`, fork venv), `aggregate.txt`, `scripts/classify.py`; full lists: `/usr/bin/python3 -I /tmp/stdcamp/lists.py` |
| Composer 769/769 and 622/622 Single-line; icons above 688/875, below 54/115 raw = 42/71 icons + 44 bar layers on 12 cards (39/66 smart objects); 71 cards without icon layers | `/tmp/stdcamp/scan2.jsonl` by `PYTHONPATH=/tmp/wt-cards/src ~/Documents/Tools/psd-tools/.venv/bin/python /tmp/stdcamp/scan2.py`; `agg2.py` |
| Templates: `Text[DropShadow,Stroke]`, `Stats Bars[Stroke]`, text layer names | fork read of `MDW/Templates/*.psd` (2026-10-20); layer dumps `/tmp/psdmap/tmpl_*.txt`, `deep_Creature_Template.txt` |
| Class A 87 cards / 326 layers; critic 101 / 350; class B 62 / 47; class D 9 / 34 | `~/Documents/Tools/psd-research/RENDER_STATUS.md` ("What is still missing, by class", "Critic pass") |
| Fonts: 10 platform-only faces / 112 layers, 6 missing / 30 layers | `~/Documents/Tools/psd-research/text-design/score-runs/TRIAGE.md` "Exclusions"; companion skill §3 |
| 813/821 files, 5 case mismatches, 6 numbered duplicates, 2 PNGs without PSD | `git -C ~/Desktop/TTSCardMaker ls-files`, stems compared case-insensitively |
| Overlays 52 (53 on 40 cards), chart 12 + 54 / 15 styled, trial numbers, `layoutMode` | `aggregate.txt`; `.claude/skills/ttscardmaker-companion/SKILL.md` §5-6; `MDW/Scripts/Effect Layers.jsx` line 18 |
| Parser dispatch order, exclusion lists, FX check; rulebook rows; BUGS D1-D3, D6 | `commands/psd_parser.py` :465-476, :887-893, :32-33, :412-439; `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` §3; `docs/BUGS_AND_IMPROVEMENTS.md` |

**Text group lists** (stems; folders in the `lists.py` output). *Stroke only (37):* Pure_Vessel, Doc, The_Master, Mari, Clock, Dynamite,
Wooden_Spikes, Stone_Spikes, Crystal_Spikes, Iron_Spikes, Return_To_Sender, Copper_Sword, Shellphone, Iron_Sword, Titanium_Sword,
Bolts_and_Screws, Candy_Corn, Monster_Meat, Potato, Tomato, All_Purpose_Medical_Kit, Grapes, Pep_Drink, Pepperoni_Pizza, Regeneration_Potion,
Snow_Bunny, Strawberry_Shortcake, The_Chicken, Watermelon, Dumplings, Feast, Hadal_Stew, Maxim_Tomato, Wand_of_Obesity, Bag_of_Evil,
Bag_of_Good, Pile_Of_Swords. *DropShadow only (12):* Bulborb Larva, Dark_Tentacle, Skeleton_Mate, Frog, The_Radiance, SOMETHING, Stranger,
ABBI, Empress Bulblax, Grimora, Jevil, Spicy_Spray. *None (25):* Red_Knife, Regret, Grodus_X, Jack-In-The-Box_Zombie, Polevaulter_Zombie,
Tanjerin, Zigg, Green_Mage, Rasta_Beast, Traitor_Lord, Dark_Daroach, Butterfly, Crowned_Magolor, Necrodeus, Hooktail, Macho_Grubba,
Sir_Grodus, Little_Guy, Doctor_Zomboss, Gandhi, Leshy, Pandanoko, Apex_Predator, Flame_Trap, Spiky_Ball_Trap. *No top-level Text group
(11):* Dreamshield, Unbreakable_Strength, Tainted_Lazarus, Kel, Mr_Jawsum, Heavy_Blow, Fragile_Strength, Newspaper, Quick_Slash,
Stalwart_Shell, Kingsoul.
