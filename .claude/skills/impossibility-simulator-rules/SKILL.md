---
name: impossibility-simulator-rules
description: "Load to explain how the Impossibility Simulator card game is PLAYED: \"how does the game work\", \"explain the rules\", what a timing tag means ((1CD), (1GCD), (2U), (1/2CD), (OFS), (WOF); the old (1PGT)/(E2T) and what replaced them), cooldowns, uses, time based, turn order, global turn, turn phases, damage (ATK - DEF, combo damage, sneak attack, rounding), minions, primary/secondary field, item types, N.M.E. shop, star points / points / BRICKS, reactions and resolve order, negation, status effects (stacking, upgrades), creature types and immunities, damage types, death and revives, starting hand / constructed / ban list, \"which rulebook page says X\", or to check a /query_rulebook answer or wording in the game's own vocabulary. Contains the game in one page, every timing tag (the rulebook's old ones; the cards' cooldown and uses tags of 2026-10-21 with card evidence), attack arithmetic, items, reactions, 64 status effects, glossary, Known gaps, page citations. Card DATA encoding: impossibility-cards-reference."
---

# Impossibility Simulator: the rules of the game

Rulebook text verified 2026-10-20 and re-checked 2026-10-22 (unchanged by the cooldown update) against
`~/Desktop/TTSCardMaker/site/dist/data/rules.json` (rebuilt 2026-10-22 00:43; 48 pages, byte-identical to the 48
`rulebook` entries in `stats.pkl` at PICKLE `3888392` and in a fresh parse of TTSCardMaker `73904dbd`) and the PNGs in `~/Desktop/TTSCardMaker/Auxiliary/Rulebook/`. A
citation like (08_Global_Turn) is the page's PSD stem. Tag and status effect counts are over the cards.json shape of
2026-10-22: 773 rows, which is 767 distinct cards plus 6 `dup` copy rows, and 771 distinct names (`tags.py` drops the
dup rows and dedupes on psd/png path + name), built with build_site.py's own card loop from a forced sweep of
`73904dbd` (Provenance). Counts by kind, stars and type are `stats.pkl` PSD counts, unchanged from 2026-10-20 (cards.json
adds two PNG-only cards, Stubborn Queen and Gust of Trees). Card names here are file names; the bot's parser v2 stores
printed titles, which differ from the file name for 159 cards (165 rows with the copy rows): 130 changed with this
parse, 29 (the Clots, the Ingredients and others) were already renamed in `site/Card Edits.json`. Blockade prints "The
Wall By Ech".

**Live data.** PICKLE `3888392` (AutoCroissant, 2026-10-22 00:23) pushed this parse: the bot's `stats.pkl` matches the
fresh sweep exactly (819 entries, no ability, display_name or stars difference), and the website's cards.json (rebuilt
00:43) gives the same tag counts. So the bot and the site carry the new tags and printed titles: `/query_ability (1gcd)`
finds 82 cards and `(1pt)` none (only the rulebook page 11_Abilities2 still holds it, for `/query_rulebook`).

`MDW/Old Rules/` is history only. Where the rulebook is silent or contradicts itself this skill says so under "Known
gaps and ambiguities"; never fill a gap by guessing. Use the rulebook's words only (creatures, minions, items, item
types, field cards, N.M.E. items, status effects, buffs, debuffs, stars, types, abilities, timing tags, global turn, slots,
lanes, primary/secondary field, points, star points, BRICKS, shops, starting hand, facedown items).

## 1. The game in one page

- **Goal.** "Defeat all of your opponent's creatures, while making sure they don't defeat all of yours!", using
  "CREATURES and ITEMS to attack and protect" (01_Getting_Started). The "Don't be stupid" rule is named, never stated.
- **A team holds** a starting hand of 7 creatures and 10 items, 2 points and 7 star points (03_Setting_Up, 13_Items).
  Three counters per team (04_Table2): **star points** (reset to 7 each global turn; items cost their stars), **points** (start 2;
  gained equal to a creature's stars when it dies; pay abilities and the item shop), **BRICKS** (mining and 1 per team
  per global turn; buy field cards).
- **The table.** 6 slots, 3 per team, in three lanes; beside each slot HP/DEF/ATK/SPD counters and status effect
  markers, behind it a holder with one facedown item (16_The_Field, 04_Table3, 03_Setting_Up). Around it: decks sorted
  by stars, three shops, graveyard, discard pile, field card slots, auxiliary decks, markers, 0.5 chip, and for RNG the Redirection cube and Order Determinator (04_Table4; 08_Global_Turn
  breaks a last tie with "flip a dice, or roll a coin") (04_Table–04_Table5).
- **Flow.** Slots take turns one at a time, lane by lane, fastest first; once every slot has gone a **global turn** has
  passed: star points reset, +1 BRICK each (08_Global_Turn). A turn gives the creature one **action** (attack, mine,
  defense mode) plus (OYT) abilities and items (09_TakingYourTurn). "When a play occours [sic], a player may choose to react to
  it with one of their items or abilities" (12_Reactions): (AAT) abilities "at any time" (11_Abilities), facedown items
  "can be played at any time" (13_Items, 04_Table3), hand items "reactively only if it's currently your turn"
  (12_Reactions5). A creature at 0 HP goes to the graveyard and gives ITS OWNER points (17_Death); "You must immediately
  summon a creature into an empty slot if possible" (16_The_Field; the page does not say where the creature comes from).

## 2. Setup and the constructed starting hand

| Starting hand (03_Setting_Up) | 5★ | 4★ | 3★ | 2★ | 1★ | Total |
|---|---|---|---|---|---|---|
| Creatures | 1 | 2 | 3 | 1 | — | 7 |
| Items | — | 2 | 3 | 3 | 2 | 10 |

Each team fills its 3 slots with creatures face down and puts 3 "Facedown items" one behind each slot; creatures flip up
when everyone is ready; items are revealed only when played; only your side sees your facedown items (03_Setting_Up).

| Shop (05_Shops) | Stock | Price | Restock |
|---|---|---|---|
| Item shop | 2 items per star level, revealed | 4x stars in points (2★ = 8) | same star level; reshuffle once per global turn for 1 point |
| Field shop | 2 field cards per star level; 1 per level revealed, the rest at the end of global turn 1 | 2x stars in BRICKS | none |
| N.M.E shop | 4 random N.M.E. items, revealed | "written on each card" | random N.M.E. item |

**Where the hand comes from.** New players draw: the table holds a "simple version of the game", creature and item
decks with the "more complex creatures an items" removed and no field cards or N.M.E. items, and "New players will draw
their creatures and items from here" (04_Table5). Also on the table (04_Table): the DEVELOPER DECK, "a reduced version
of the full deck" of "newer or recently changed cards that require more playtesting", mixed with "Baseline" creatures
"considered balanced"; "For most players you can ignore this", it is for "a more wacky experience". How a hand is built
from the full decks is not stated (Known gaps). **Constructed** is defined only in
`~/Desktop/TTSCardMaker/site/Constructed Ban List.txt` (in `MDW/` until 2026-10-20): "choose whatever cards you want out of the entire deck for your
hand", "You may also only have one of each item in your hand", and each player may pick "one 3 star or lower field item
to be active in the shop at the start of the game". It bans 53 cards (creatures 8/8/9/4 for 5★–2★, items 8/8/3/1 for
4★–1★, 4 field items) for
"creature/ability copying", "antifun", "reliance on RNG", "too easy to build combos", "too powerful of a counter". The
website's deck builder (`site/assets/deck.js`) enforces the same counts (`HAND`), one of each card, no banned card, and
an optional field card with `FIELD_MAX_STARS = 3`; only creatures and items fill slots. List-vs-data
mismatches: the deck builder's one-copy rule covers creatures too, where the list says only "one of each item"; Tainted Bethany is 4★ (listed 3★), NME Stock 2★ (listed 1★), The Bank is an N.M.E. item (listed as a field
item), Cave of the Past and Error Room are 4★ field cards. Names: "The Wall (By Ech)" is Blockade.psd, printed title "The
Wall By Ech"; "Leeroy Jenkins" prints as "LEEEEEEEEEROY"; "Berserk" is "Berserk!".

## 3. Turn structure and every timing tag

**Global turn.** "When every slot has taken its turn once, that is called a GLOBAL TURN. When the global turn resets,
each team is set to 7 Star points. At the end of each GLOBAL TURN, each team gains 1 BRICK" (08_Global_Turn). The counter
starts at 1 (04_Table2). A dead slot or lane is skipped "but count time as if that slot/lane had gone" (16_The_Field).

**Turn order** (08_Global_Turn): lane by lane. Lane order = TOTAL SPEED of the fastest creature of each side (primary
field only); tie → least HEALTH LOST (MAX − CURRENT, summed over the lane in the example); tie → highest STAR LEVEL; tie →
RNG. Slots within a lane: same keys. "MOST SPEED > LEAST HEALTH LOST > HIGHEST STAR LEVEL > RNG" (worked in §12). (AAT)
speed changes and facedown items may change which lane goes next; as a slot's turn starts its creature may swap out for
2 points (08_Global_Turn), 1 less if WIND (20_Creature_Types).

| Phase (09_TakingYourTurn) | What happens |
|---|---|
| 1 | The slot's turn begins; (BOT) effects |
| 2 (main) | (OYT) abilities; items from hand; swap facedowns to/from hand; shops and item-shop shuffle; actions; swap primary/secondary fields if no action used yet |
| 3 | (ENDT) effects; nothing from phase 2 any more |
| 4 | Turn over; status effects deal damage/heal, durations tick down 1 (also 19_Status_Effects) |

**The action**: one per creature per turn — ATTACK any creature on the field (§4), MINE (BRICKS = stars; prime creatures
only, "Minions cannot mine", 15_Minions) or DEFENSE MODE (DEF doubled until its next turn begins; card turned 90°)
(09_TakingYourTurn). Every minion and parent creature in the primary field "may use their actions", in whatever order the player chooses
(15_Minions).

**Timing tags, old system (the rulebook's).** Defined on (02_Abbreviations) and explained on (11_Abilities)–(11_Abilities7). "X" is a variable;
cards print the number. 🕐 = "time based" (considered "refreshed" and become usable again in the specified timeframe); ⏳ = "multi
turn". Counts = distinct cards using the tag before the update (stats.pkl PICKLE 56b43b9; every tag: 606 uses in 352 of 767
cards; now 625 uses in 352). Since 2026-10-21 (XPT), (XPGT), (EXT), (FTEXT), (EXGT) appear only on the pages.

| Tag | Rulebook wording | Cards |
|---|---|---|
| (OYT) | On your turn: wait "until the creature with this ability's turn" (11_Abilities) | 38 (42 since 2026-10-21) |
| (AAT) | At any time; usable reactively (11_Abilities) | 13 |
| (XPT) 🕐 | X time(s) per turn; implies (OYT); "the most common ability type" (11_Abilities2) | (1PT) 72, (2PT) 6, (XPT) 2 |
| (XPGT) 🕐 | X time(s) per global turn; implies (AAT) (11_Abilities2) | (1PGT) 55, (2PGT) 6, (3PGT) 1, (4PGT) 1, (XPGT) 1 |
| (XPG) | X time(s) per game; always also (AAT) or (OYT) (11_Abilities2) | (1PG) 30, (2PG) 4, (3PG) 3, (XPG) 2 |
| (EXT) 🕐⏳ | Every X turns from the Xth: refreshed on the creature's Xth turn after game start or last use (turns 2, 4, 6…); (OYT) (11_Abilities4) | (E2T) 42, (E3T) 2, (EXT) 1 |
| (FTEXT) 🕐⏳ | Every X turns from the 1st (turns 1, 3, 5…) (11_Abilities4) | (FTE2T) 14 |
| (EXGT) 🕐⏳ | Every X global turns; (AAT); "refreshed when the global turn starts" (11_Abilities4) | (E2GT) 8, (EXGT) 1 |
| (BOT) 🕐 forced | Beginning of turn, "before it does anything else" (11_Abilities3); phase 1 | 24 |
| (ENDT) 🕐 forced | End of turn, "after everything else it can do" (11_Abilities3); phase 3 | 26 |
| (BOGT) 🕐 forced | Beginning of global turn, "before anything else in the global turn" (11_Abilities3) | 39 |
| (ENDGT) 🕐 forced | End of global turn (02_Abbreviations; 11_Abilities3 shows it only inside an example) | 37 |
| (OFS) forced | On first summon from the hand; cannot be negated or true negated; counts as (1PG) (11_Abilities5) | 34 |
| (OS) forced | On summon, every time (11_Abilities5) | 8 |
| (OD) forced | On death, forced (11_Abilities5); "Does not apply to creatures not on the field" (17_Death) | 23 |
| (WOF) forced | While on field: passives, most "in constant effect (WOF)"; "“(WOF)” is often only included when the ability effects other creatures as to signfy that it goes away when the creature exits the field" (11_Abilities7); in 02_Abbreviations' FORCED box with (OFS)/(OS)/(OD). A (WOF) line can carry a cooldown: Michael "(WOF) Discarded items may be bought for 3/4 shop price (1CD)." | 20 |
| (XCD), (XGCD), (XU) | **Not in the rulebook.** Before the update already on some cards; the new system below | cards, not uses (RULEBOOK U1's 22 is uses): (1GCD) 20, (2CD) 14, (1CD) 13, (2U) 6, (2GCD) 2, (1/2CD) 1 (Chris & Jordi), (-1/1CD) 1 (The Temple of Ahn'Qiraj), (3U) 1, (4U) 1 |

An unused time based ability: "it's saved so you can use it on the next turn, but its cooldown doesn't reset until it's
used"; abilities "do not count time passing while in the hand" (11_Abilities4). Forced tags are "non-optional and
assumed to happen automatically" (11_Abilities3). "You may react to time based [such as (BOGT)] effects with any (AAT)
effects or facedown items" (12_Reactions5): by 02_Abbreviations "time based" is every 🕐 tag ((XPT), (XPGT), (EXT),
(FTEXT), (EXGT) and the forced (BOT)/(ENDT)/(BOGT)/(ENDGT)); nothing is stated for (OFS)/(OS)/(OD); (OFS) abilities have
true negation immunity (22_Negation). These quotes are about the old tags; no page says yet whether they carry over.
Time Riddle lists cooldown tags as its examples of "time-based" ability usage ("(2CD) -> (1CD)"), so that card treats
them as time based; Energy Ritual says only "all time-based and (XPG) abilities" and names no cooldown tag, so for it
this is an inference. No page says these rules carry over.

**Timing tags, new system (from the cards, 2026-10-21; the rulebook pages still define the old tags).** The team
replaced the per-turn and every-N tags on every card: TTSCardMaker `df02b3b6` "cooldowns (2)" (50 PSDs), `09131c67` "(3)"
(49), `1b4f36a0` "(4)" (50) and `7cb4c867` "(5)" (29), all by Zomoroko on 2026-10-21, 178 PSDs with their PNGs (no
"(1)" exists); follow-ups `8ec1b925` (font badges), `c3f9d685` and `73904dbd` (website tooltips, old tags marked "Old
system"). The 178 are 141 creatures, 17 items, 14 field cards, 4 N.M.E. items and 2 aux items (no status effect card);
172 changed tags only, 6 also wording (data-facts §4). No page defines the new tags: the text of (02_Abbreviations), (11_Abilities2),
(11_Abilities4) and (22_Negation) still uses the old ones, and the pictures show old card text (below). Card text
gives evidence but no rule (below); 11_Abilities4 already calls a time based ability's timer its "cooldown". Meanings are
only what the letters say: CD cooldown, GCD global cooldown, U uses. Counts = uses / cards in the parsed ability text
(fresh sweep of TTSCardMaker `73904dbd`, 2026-10-22, the same data as PICKLE `3888392`; `tags.py`).

**Old card text in the page pictures.** Toy Bonnie's card on 12_Reactions, 12_Reactions2, 12_Reactions3 and
12_Reactions4 reads "(OPGT) Inflict up to 6 stacks of Softened for 1 turn ..." (an O, not a 1: at 4x zoom on 12_Reactions
the same crop shows a distinct "1" in "for 1 turn"); the card itself has read (1GCD) since before the update (PICKLE
`56b43b9`). Whispy Woods on 12_Reactions4 reads "(OPT) You can spend 2 points ..." (today "(1CD)(2 points) ..."). (OPGT)
and (OPT) belong to neither system. On 01_Getting_Started, OMORI reads "2+ counters: +1 SPD, first turn and (EOT)
attacks inflict 2 turns of Afraid." (today "(1/2CD) Attacks inflict 2T Afraid."), and Computer Virus "deal 2 piercing
(CD) to itself": that (CD) is the old counter damage (today "2 piercing CDMG"), not a cooldown tag.

| Family | Letters | On the cards (uses / cards) | Replaced, per PSD |
|---|---|---|---|
| (nCD) | cooldown, N | (1CD) 101/88, (2CD) 55/55, (3CD) 2/2, (XCD) 1/1 | (1PT) → (1CD); (EnT) → (nCD): (E2T) → (2CD), (E3T) → (3CD), (EXT) → (XCD) |
| (a/bCD) | cooldown, two numbers | (1/2CD) 15/15 | (FTE2T) → (1/2CD) |
| (nGCD) | global cooldown, N | (1GCD) 94/82, (2GCD) 11/11, (XGCD) 1/1 | (1PGT) → (1GCD); (E2GT) → (2GCD), (EXGT) → (XGCD) |
| (a/bGCD) | global cooldown, two numbers | (-1/1GCD) 1/1 (The Temple of Ahn'Qiraj, as (-1/1GCD)(OYT)) | (-1/1CD), itself undefined |
| (nU) | N uses | (2U) 18/17, (3U) 2/2, (4U) 2/2, (XU) 2/1 | the n of an (nPT)/(nPGT) with n ≥ 2: (2PT) → (1CD)(2U); (2PGT), (3PGT), (4PGT) → (1GCD) with (2U), (3U), (4U); (XPT)/(XPGT) → (1CD)/(1GCD) with (XU) (Michael's Blessing) |

Unchanged: (AAT), (BOT), (ENDT), (BOGT), (ENDGT), (OFS), (OS), (OD), (WOF) and (nPG) (cards only: (1PG) 32/30, (2PG)
4/4, (3PG) 3/3, (XPG) 2/2); (OYT) went from 38 to 42 cards because the 4 cards that moved to a global cooldown gained
it. 218 cards carry a new tag: 173 had only old tags, 45 already had one of the former undefined tags. The tag sets of
214 follow the last column exactly; 4 also moved from turns to global turns and gained (OYT): Water Totem (1CD) →
(1GCD)(OYT), The Gates of Ahn'Qiraj (2CD) → (2GCD)(OYT), Crock Pot (1PT) → (1GCD)(OYT), The Temple of Ahn'Qiraj
(-1/1CD) → (-1/1GCD)(OYT). The Bank (N.M.E., `7cb4c867`) lost its only cooldown-family tag: its line "Will store
individual uses of a multi-use (XPT) abilities." was deleted, and its cooldown line now reads "Cooldowns of stored
actions/abilities recharge as if they were used." It keeps (ENDGT), so it is not among the 218. "(OPT)" is in neither
system and on no visible card text: it survives only in a hidden `old` layer of Bobberlington and Michael (an earlier
ability) and in the 12_Reactions4 picture of Whispy Woods (the card reads (1CD)(2 points)); likewise "(OPGT)" only in
the Toy Bonnie pictures.

**What the cards show (card evidence, not rules).** Measured on the fresh parse (`lines.py`); each point is a reading
of card text. Verbatim texts and the per-line mapping: `reference/data-facts.md` §4.
- **Turn vs global, (OYT) and (AAT).** (nCD) and (a/bCD) never share a line with (OYT) or (AAT). Global cooldowns share
  a line with (OYT) on 10 lines (8 start with both, e.g. "(1GCD)(OYT)", "(1GCD)(4U)(OYT)", "(-1/1GCD)(OYT)"; plus
  Jetpack's mid-sentence "(OYT)" and a Michael's Blessing row), never with (AAT). Exactly the 4 turn tags that became
  global cooldowns gained (OYT). This matches 02_Abbreviations' "These imply (OYT)." for (XPT)/(EXT)/(FTEXT) and "These
  imply (AAT)." for (XPGT)/(EXGT). The website's tooltips (`site/assets/text.js`) agree: (nCD) and (a/bCD) under "On
  your turn", (nGCD) and (a/bGCD) under "At any time"; "(1/2CD) Cooldown: 1 turn, then 2", "(2U) 2 uses". They were
  written during the same update (`c3f9d685`, MichaelJSr, 2026-10-21), so they are a reading of the cards, not an
  independent or designer source. They group (nU) under "Per game & triggers", which could suggest uses per game; Time
  Riddle suggests uses per cooldown (below).
- **Uses.** (nU) comes right after (1CD) or (1GCD) on all 22 cards that carry it, never after (2CD) or longer. Line order
  is cooldown, uses, then cost or (OYT): "(1CD)(2U)(1.5 points)", "(1GCD)(4U)(OYT)", "(1CD)(3U)(2 points)". (nU) carries
  the n of an old (nPT)/(nPGT) with n of 2 or more, so 11_Abilities2's "(2PGT) Heal a creature by 2 HP" ("this creature
  can heal twice at any point during the global turn") would now be written (1GCD)(2U). Supreme Jeh'Oul: "this ability
  becomes and stays (1CD)(2U)". Items still write uses in words ("3 uses:", "Uses: ∞"): those are multi-use items
  (13_Items, 14_Item_Types), not the (nU) tag.
- **(a/bCD).** (1/2CD) is on 15 creatures, 14 from (FTE2T) (Chris & Jordi had it already). 8 pair it with a (2CD)
  ability (Chris & Jordi, Daroach, Dark Daroach, Doctor Zomboss, Koops, The Abyss Watchers, Vivian, Yoshi Kid):
  11_Abilities4's (E2T)/(FTE2T) pair, "This ability happens on turns where the above doesn't". So (1/2CD) reads as first
  usable on turn 1, then every 2 turns, and (2CD) as turns 2, 4, 6 (inference). The only (a/bGCD) is The Temple of
  Ahn'Qiraj's (-1/1GCD)(OYT); the -1 is unexplained.
- **Michael's Blessing** (N.M.E., 5★): "(OYT) -> (1GCD) / (BOT)/(ENDT) -> (1GCD) / (BOGT)/(ENDGT) -> (1GCD) / (XCD) ->
  (XGCD) / (1CD)(XU) -> (1GCD)(XU)". Its old last rows, "(XPT) -> (XPGT)" and "(EXT) -> (EXGT)", were in the other
  order, so (XPT) became (1CD)(XU) and (EXT) became (XCD). The cards' only explicit statement that a turn cooldown has a
  global cooldown counterpart.
- **Time Riddle** (4★ equip, banned in constructed): "Doubles all time-based and (XPG) ability usage, for example: (2CD)
  -> (1CD) / (1CD) -> (1CD)(2U) / (1PG) -> (2PG) / (1GCD) -> (1GCD)(2U)". Doubling halves a cooldown of 2 or more and
  doubles uses at cooldown 1: the best card evidence that (nCD) means usable once every n turns and (nU) uses per
  cooldown.
- **Recharge words.** Energy Ritual now "fully recharges and restores all usages of all time-based and (XPG)
  abilities"; Michael "(2GCD) Recharge 1 turn of charge to another creature's ability" (was "(E2GT) Refresh another
  creature's ability usage as though another turn has passed"). Older text with the same idea: Cave Of The Past "All
  🕐time-based effects and abilities do not refresh."; Hot Spring "refresh its own ability as if another turn has passed".
- **Tags mid-line.** 12 lines outside the two conversion tables carry a cooldown tag after other text: after a form or
  condition label (Warcraftian Druid "Bear: +1 DEF, (1GCD) Intercept ...", Pix, Yin Yarn, OMORI, Niko, Waddle Dee) or at
  a sentence's end (Corruption "Can only activate (1GCD).", Spice, Michael, Supreme Jeh'Oul). A tag at a sentence's end
  limits that clause (reading).
- **Multi-turn is gone from the letters.** (1CD) came from (1PT), which was not multi turn; (2CD), (3CD) and (1/2CD) came
  from (E2T), (E3T) and (FTE2T), which were. Rules that name multi-turn abilities (22_Negation "Multi turn abilities such
  as (E2T) abilities still count time passing even while they are negated"; 11_Abilities4's saved ability whose
  "cooldown doesn't reset until it's used") now need a mapping no page gives. Related: 11_Abilities7 "if the creature
  loses that ability then gains it back, it's not reset but instead incremented by a turn".

**Simultaneous effects** (11_Abilities6): higher-SPD creature first, then the higher-cumulative-SPD team's active items,
then field cards; ties use turn-order tie-breakers; the owner orders same-side conflicts; both teams setting one value
"remains random"; simultaneous HP-setting revives add up (to Max); directly conflicting effects: the player chooses.

## 4. Stats: HP, DEF, ATK, SPD

Bars show MAX values, counters beside the slot show CURRENT values (06_Reading_Cards). One marker is one unit of the real
stat ("1 DEF marker is 0.5 DEF, 1 HP marker is 1 HP"); "Flip them over for stat downs" (04_Table4); "For increments of
0.5, use a marker turned to its side" (07_Stats1, 07_Stats3, 07_Stats4). The 0.5 chip is for "a counter [that] uses half
integers" (04_Table4).

| Stat | One bar | Meaning |
|---|---|---|
| HEALTH | 2 HP ("8 bars … 16 HP") | "A creature perishes when its health reaches zero" (06_Reading_Cards, 07_Stats1) |
| DEFENSE | 0.5 DEF ("4 Bars = 2 DEF", 07_Stats3) | "Subtracted from any attack damage this creature takes" (06_Reading_Cards); "they take damage = ATK-DEF" (07_Stats3) |
| ATTACK | 1 ATK | damage dealt "minus DEF when attacking" (07_Stats4) |
| SPEED | 1 SPD | "determines turn order and combo damage" (07_Stats4) |

**MAX HP** (07_Stats1, 07_Stats2): regular healing stops at MAX; raising MAX raises current HP equally and "counts as
healing" (Mushroom +3.5: 16 → 19.5); lowering MAX "does not decrease their current HP, unless if their current HP would
become greater than their Max HP (in which case it is set to Max HP)" (16 → 14, but 10 stays 10) and "does not count as
damage"; two heals from one item/ability are "one instance".

**Rounding** (07_Stats5): "to the nearest half in the positive direction, after reductions or gains" (Chilled −25% on
5 SPD: 3.75 → 4; Enraged +25% on 5 ATK: 6.25 → 6.5). Additives first, then multiplicatives; multiplicatives add together
(−25% + −50% = −75%). The page's 120 computed cells (6 change columns x 20 base values) are exactly "nearest 0.5, halves up". Damage rounds up: 3.25 → 3.5 (10_Attacking2).

**Attacks** (10_Attacking): a creature's attack action, an attack item, or an ability dealing DMG separate from ATK.
"Attacks deal ATK - DEF in damage", primary field only by default. COMBO DAMAGE: attacker has "TWICE THE SPEED VALUE" of the
target → x1.5; "TRIPLE" → doubled (reading these as "at least 2x / 3x" is an inference, not page text). Examples: A (ATK 7, SPD 4) hits B (DEF 1, SPD 4): 7 − 1 = 6, B at 4/10. C (ATK 7, SPD 6) hits D (DEF 1, SPD 3): 6 x
1.5 = 9, D at 1/10. SNEAK ATTACK replaces a normal attack: piercing "25% of their attack stat +1", same on-hit effects.
Multi-hit attacks "count as seperate attacks for each hit, and each instance may be reacted to seperately"; separately,
when an ability or item attacks "multiple enemy creatures at once" the attacks go left-to-right from the attacker's
view and counter damage or on death effects wait until "all instances of the attack are resolved"; redirections retarget to a creature of choice, interceptions to the interceptor (leaving defense mode); inflictions
on offense or defense apply before the hit and, if from the attack itself, cannot be separately negated/reflected/
redirected/intercepted; only attack actions proc "on attack" effects (10_Attacking2).

## 5. Creatures, minions and the Auxiliary/Creatures cards

A creature card shows type icon (top left), stars (top right), four stat bars and the ability, "only active while on the
field" (06_Reading_Cards, 11_Abilities). Decks hold 2★–5★ creatures (04_Table; stats.pkl: 297 cards, 45/103/93/56).

**Minions** (15_Minions) are "small creatures that when summoned are attached onto a parent prime creature":

| | Creature ("prime creature", never defined) | Minion |
|---|---|---|
| Source | hand / creature decks | auxiliary minions deck, via abilities (04_Table4) |
| Own action | yes | yes |
| Can mine | yes (18_Field_Cards) | no |
| Points on death | its stars, to its owner (17_Death) | none; still goes to the graveyard |
| Attachment | — | always to the parent, even when moved or swapped to hand |
| Parent dies | — | its minions die, except persistent ones |

Offensive minions share the parent's field; protective minions (icon file `Defensive_Minion`) take the other field,
pushing the parent and its offensive minions to the secondary field; one summoned after the parent acted gets no action
that turn; **persistent** minions attach to the next prime creature in the slot and, alone, "must be killed for the game
to be won" (15_Minions).

**The `Auxiliary/Creatures/` cards** (Azamoth the Possessed Vessel, Igor Roballtowski, Mettaton EX, Mettaton NEO,
Militron (No Armor), Morpho Knight, Shadow Queen; bot type `creatures`, plural) have no rule in the book; Azamoth is only
a revive example (11_Abilities6). The rulebook has no name for them; their PNGs are the regular creature size.

## 6. Items, item types, N.M.E. items and shops

Items "are your main resource besides creatures"; one costs star points equal to its stars and is discarded after
resolving unless multi-use ("goes back facedown or to the hand"); the 3 facedown slots play at any time; hand↔facedown
swaps only on your turn; sell on your turn for points = stars (facedown ones at any time; never a used item, equips
included); random draws use a copy of the full deck (13_Items). Ability-summoned items come from the auxiliary items deck
(04_Table4). Star points are spent on play; an invalidated item returns and refunds (12_Reactions5). Item types are the
top-left symbol and "cannot be manipulated" (14_Item_Types); repo folder in brackets, counts from card data.

| Item type | Rule | Cards |
|---|---|---|
| HEALING ("food") [Heal] | heal creatures; not on machine creatures (page shows the machine icon; rules.json mis-reads it as `[support]`) | 35 |
| ATTACK [Attack] | "do damage, and count as an attack" | 16 |
| NEGATION [Negation] | "These items “Negate” something, taking it out of play" | 5 |
| SUPPORT [Support] | "general use items with various effects" | 30 |
| COUNTER [Counter] | conditional, "can only be used reactively"; often multi-use | 9 |
| INFLICTION [Infliction] | inflict a status effect; count as an infliction | 10 |
| EQUIP [Equip] | passive effect on your creature; unequip on your turn (a negative-effect equip locks until the next global turn); returns to hand if the creature dies; stays on a creature swapped to hand; a harmless one may go on an enemy, who then owns it; effects may "forcibly" unequip (14_Item_Types2) | 65 |
| FRAGILE EQUIP (icon, no folder) | cannot be unequipped; discarded when the creature dies (14_Item_Types2) | 11 by icon: 10 inside Equip's 65, 1 (Life Shroom) in Heal |
| ACTIVE [Active] | constant effect; has HP and DEF but is not a creature: no equips, healing items or status effects; discarded at 0 HP (14_Item_Types2) | 9 |
| DEPLOY [Deploy] | constant effect on a chosen slot (not "used on" that creature); cannot be attacked (14_Item_Types2) | 3 |

The first six "by default are discarded after being played" (14_Item_Types; COUNTER items "tend to have multiple uses
and are only discarded when all uses have been used"); the last four "do not get discarded immediately" (14_Item_Types2).
A healing item used to deal DMG counts as an
attack item, and vice versa (11_Abilities7). **N.M.E. items** (26) are "not in the standard deck" and cost what the card
says (05_Shops); "N.M.E." is never expanded. Items run 1★–4★ (stats.pkl: 182 PSDs).

## 7. Abilities, reactions, negation, and how ability text reads

An ability is "the main chunk of text at the bottom half of a creature"; when it can be used is "defined by
abbrviations [sic]" (11_Abilities), the tags of §3. Page examples: "(1PT) Summon a Mini Bee.", "(AAT)(3PG) Spend 7 points to
deal 10 piercing damage.", "(ENDT) +5 DEF until (ENDGT)", "(OFS) Summon a Koopa.", "(WOF) Friendly creatures gain +1 DEF."
(11_Abilities2–7; (1PT) is an old tag, §3). Points pay ability costs (04_Table2). Stolen/swapped abilities "refresh as brand new per creature";
damage-counting abilities ignore overkill (11_Abilities7).

Reading card text: a type icon in the text IS the type (stored as `[fire]`, rendered as an icon or emoji); "X" is "a
variable that can represent any number" (02_Abbreviations); "5T1 Poison" = 5 turns of 1 stack (19_Status_Effects2);
"(Action)" marks an ability costing the action (36 uses in 28 cards, 37 in 29 counting one "(action)"; card convention, not rulebook;
a grey badge, not a timing tag, since 2026-10-22); PF = primary field, CDMG =
counter damage (02_Abbreviations).

**Reactions** (12_Reactions–12_Reactions5): "When a play occours [sic], a player may choose to react to it with one of
their items or abilities … the reaction happens before whatever you reacted to." Nothing resolves until plays stop; then
reverse order (4>3>2>1), except "anything that deals damage resolves last" (Pebble 5 DMG → Hardened "giving it 3 defense" (the page says "5
stacks"; 6 at +0.5, see Known gaps) → Bomb 7 DMG → Metal Shield resolves 4>2>3>1, 12_Reactions3) and the original play "will always be resolved at the very end"
(5>4>2>3>1, 12_Reactions4). Example: Mighty Pebble (5 DMG); Toy Bonnie reacts with 6 Hardened (+3 DEF) and takes 2
(12_Reactions). A reaction that kills the target first refunds the attacker's item/action/ability; NO TAKE BACKS once
reacted to; hand items only on your turn, no shopping (12_Reactions5).

**Negation** (22_Negation): a negated attack "does not happen", action not refunded. A negated creature "is unable to use any
of its own abilities (unless otherwise stated), but may still use the abilities of any equipped items or minions, and can
still use actions"; multi-turn timers keep counting; things already summoned remain. "5+
star creatures are immune to negation"; True Negation beats that; True Negation Immunity beats even that — (OFS)
abilities and "upon being negated" effects have it. A negated item returns where it was, star points not regained,
unsellable while negated; negated equip/active/deploy items return to hand.

## 8. The field and field cards

Each slot splits into a PRIMARY FIELD, "the only field that can be targeted by default, and the only field creatures can
act in" ("everything in the Primary Field" = all 3 slots), and a SECONDARY FIELD that exists only when a protective minion
or protective active item pushes a creature back: untargetable by default, abilities but no actions; it comes forward
when the primary field is emptied; the parent may swap fields with its protectors once per turn if no action was used
(16_The_Field). All protectors share one field; minions summoned by minions attach to the slot's prime creature (16_The_Field2).

**Field cards** (18_Field_Cards) "have a constant effect". Bought with BRICKS (2x stars), playable at once for free, may
return to hand and be replayed for star points = stars; 1 in play per team. Counter-play at the start of the global turn
or on your turn: 1x stars in BRICKS negates it until the end of the global turn (not the turn it was played, not 2 turns
in a row); 3x destroys it (also on 22_Negation). Data: 50 field cards, 1★–4★.

## 9. Status effects

"A status effect, is either a buff or debuff" (19_Status_Effects). DMG/heal effects proc "in phase 4, after (ENDT) effects",
others immediately; durations count down in phase 4 (19_Status_Effects, 09_TakingYourTurn; reworded 2026-10-24);
status DMG is piercing, not an attack, one instance per unique stack ("5 stacks of Poison is one instance of 5
damage"); the owner orders them in phase 4; a creature that "started the global turn with a status effect"
whose global turn "ends without the creature having ended its turn" has the effect tick at the end of the global turn,
"even if the creature is in hand" (19_Status_Effects). **Standard** (icons `buff`/`debuff`): one
instance per creature; **stacking** (`buffstack`/`debuffstack`): any number; stack/upgrade/reapply → longest duration;
**upgradable** (`debuffupgrade`): Smouldering+Smouldering → Burning, +Burning → Blazing; Chilled+Chilled → Frostbite,
+Frostbite → 1 turn of Frozen; Sickness+Sickness → Plague; Frozen/Blazing/Plague block the prior debuffs except to
refresh the duration, and only an upgrade refreshes it ("need to upgrade to refresh duration", 19_Status_Effects2).
Debuffs carry 1★–4★ (the undefined "tier"); buffs have no stars. Effects below are
the 64 status effect cards' own text (card data, not rulebook); (k) stacking, (u) upgradable, unmarked standard; "not X"
= "can't affect [X]".

- **Buffs (12):** Diligent Watch inflictor gains +3 ATK when this creature dies; Empowered (k) +1 ATK; Enraged (k)
  +25% ATK; Hardened (k) +0.5 DEF; Hyper (k) +25% SPD; Inspired (k) +1 Action; Marked For Life (k) −25% DMG taken; Peppy (k)
  +1 SPD; Regen (k) heal 1 HP; Reinforced (k) +50% DEF; Thickened (k) +1 MAX HP; Vigor (k) +25% MAX HP.
- **1★ debuffs (12):** Blindness attacks hit a random enemy, not light; Burdened (k) −1 SPD; Burning (u) 3 DMG, not
  fire; Disgusted cannot consume healing items; Enfeebled (k) −1 ATK; Poison (k) 1 DMG, not machine; Sickness (u)
  −0.5 DEF −1 ATK −1 SPD, not undead/machine; Smeckeldorfed (k) 1 DMG; Smoldering (u) 2 DMG, not fire; Softened (k)
  −0.5 DEF; Stuffed (k) −1 SPD, not panda; Thinned (k) −1 MAX HP.
- **2★ (16):** Afraid −2 ATK −2 SPD, not dark; Blazing 4 DMG, not fire; Chained (k) −25% SPD; Chilled (u) −25% SPD,
  not ice; Confusion attacks hit random creatures, actions must attack; Edible (k) attackers gain +25% lifesteal, max
  4; Empty (k) −25% MAX HP; Engorged (k) −25% SPD, not panda; Exposed (k) −25% DEF; Frostbite (u) 2 DMG −25% SPD, not ice;
  Glued can't swap out; Loony (k) −25% ATK; Paralyzed −50% SPD, not electric; Plague −1 DEF −1 ATK −2 SPD, not
  machine/undead; Restrained (k) (BOT) −1 Action; Spirit Flame 2 DMG −1 DEF.
- **3★ (14):** Become Back Money (k) attackers gain 1 point; Cheap gives 1 point on death; Corrupted Blood moves
  to attacker/attacked, (BOT) 4 DMG; Corruption not darkmatter, 4 DMG when its ability resolves differently than if
  negated, (1GCD); Cursed Inferno 3 DMG, indefinite; Disarmed equips have no effect; Frozen no actions, can't
  switch out, not ice; Hexed at 3 Hexed creatures it cleanses and they take 2X DMG until their next turn ends, 2 DMG
  if in hand; Imprisoned no actions, can't switch out; Insane can't switch out, all actions hit a random friendly
  creature; Pessimism (k) −1 HP from healing; Resentful Glare take 3 DMG when damaging an enemy; Shell Shock −1 DEF
  +1 ATK, 50% to hit one slot clockwise; Suffocation (k) 1 DMG, not water.
- **4★ (10):** Condemned destroyed when it expires; Contracted "No effect. Lasts until the creature exits the field.";
  Curse (k) (BOGT) lose 1 SP; Despair −3 ATK −0.5 DEF −3 SPD; Frozen In Time no actions, swaps, items, equips,
  ability or healing; Hemorrhaging (k) 0.5 true DMG; Infected +2 ATK, negated; Marked For Death (k) +25% DMG taken;
  Recursion (BOT) double all stackable buffs/debuffs; Withering (BOT) 2 DMG −0.5 DEF −1 ATK −1 SPD.

Spelling drift: page "Smouldering"/"Regeneration" vs cards "Smoldering"/"Regen".

## 10. Creature types and damage types

"Every type has a unique function, typically an immunity to a debuff" (20_Creature_Types; the extracted text puts each
description before its name; this table is the page as laid out). Creatures = distinct stats.pkl creatures with the type (75 of 297 have
none; Pandora, Kyle and Z!!!!! list their type twice in stats.pkl, so icon occurrences would give panda 26, unknown 11,
girl 2); type tokens mark "if a creature type changes" (04_Table5).

| Type | Effect | Creatures |
|---|---|---|
| MACHINE | no food items; immune to Sickness, Plague, Poison | 50 |
| WIND | swap out/in for 1 less point | 25 |
| ROCK | +1 Brick when mining | 22 |
| PANDA | immune to Stuffed, Engorged | 25 |
| FIRE | immune to Smoldering, Burning, Blazing | 27 |
| ICE | immune to Chilled, Frostbite, Frozen | 9 |
| WATER | immune to Suffocation | 18 |
| ELECTRIC | immune to Paralyzed | 15 |
| UNDEAD | immune to Sickness, Plague | 23 |
| DARK | immune to Afraid | 25 |
| LIGHT | immune to Blindness | 40 |
| DARK MATTER | immune to Corruption | 10 |
| UNKNOWN | once per game on your turn, permanently change into another type | 10 |
| GIRL | +1 point (OD); hidden overflow text: "may only be an innate type in special cases" | 1 |

| Damage type (21_Damage_Types) | Rule |
|---|---|
| NORMAL | "DMG - (1/2*DEF)" — contradicts "ATK - DEF" (07/10); see Known gaps |
| PIERCING | "Ignores DEF" (status DMG and sneak attacks are piercing) |
| COUNTER (CDMG) | only as retaliation to an attack; not an attack; bypasses abilities and protective minions |
| LIFE STEAL | DMG after DEF is healed before the attack resolves; 100% = 1:1, 50% = 1:2, 200% = 2:1 |
| SHOCKWAVE | all other enemy targets on the primary field plus the secondary slot behind the target; not an attack |
| TRUE DAMAGE | ignores DEF; cannot be negated, redirected or reduced |

## 11. Death and winning

A friendly creature at 0 HP dies, goes to your graveyard, "your team gains points equal to that creatures star level",
and its (OD) effects proc (only on the field) (17_Death). Revived by its own On Death effect: points only when fully
killed; revived otherwise: points already paid, the revived form gives 0; multi-target creatures pay when all targets die;
"destroyed" = enough damage to kill "regardless of healing or any possible factors" (17_Death). GIRL gives +1 point (OD)
(20_Creature_Types); minions give none (15_Minions). Winning = all enemy creatures defeated (01_Getting_Started),
including a lone persistent minion (15_Minions). Creatures still in hand: not stated.

## 12. A worked turn

Assembled from the rulebook's own examples; the joins are this skill's.

1. **Global turn begins**: star points → 7; (BOGT) abilities fire, higher SPD first (08_Global_Turn, 11_Abilities6);
   BRICKS may be paid to negate/destroy a field card (18_Field_Cards).
2. **Order** (08_Global_Turn's example): A (8/10 HP, SPD 5, 3★) vs B (8/9, SPD 5, 5★) in lane 1; C (8/8, SPD 6, 4★) vs D
   (6/6, SPD 4, 4★) in lane 2; E (7/7, SPD 3, 5★) vs F (8/8, SPD 3, 3★) in lane 3. Lanes 1 and 2 total 10 speed; lane 2
   lost 0 HP so it goes first; C before D; B before A (less health lost); E before F (stars) → "C > D > B > A > E > F".
3. **C, phase 1**: (BOT) effects (09_TakingYourTurn).
4. **Phase 2**: the attack, using 10_Attacking's own C/D pair, which is a different pair from step 2's (08's C and D
   print no ATK/DEF, and 08's D has SPD 4, so 6 is not double it: no combo damage): C (ATK 7, SPD 6) attacks D (DEF 1,
   SPD 3, 10/10 HP): 7 − 1 = 6, "C's speed is DOUBLE that of Creature D's" → x1.5 = 9, D at 1 HP.
5. **Reaction**: D's owner may answer with a facedown item or an (AAT) ability. That includes an ability usable per
   global turn: the old (XPGT), which implies (AAT) (11_Abilities2), and on today's cards (nGCD), whose (AAT) reading is
   inferred (§3, Known gaps). The Toy Bonnie pictured on 12_Reactions–12_Reactions4 reads (OPGT), in neither system;
   its card has read (1GCD) since before the update. DEF gains
   resolve before the attack, damage items after other plays, the attack last (12_Reactions3, 12_Reactions4); C cannot
   take it back (12_Reactions5).
6. **Phase 3**: C's (ENDT) effects, e.g. "+5 DEF until (ENDGT)" (11_Abilities3).
7. **Phase 4**: C's status effects proc and tick (09_TakingYourTurn, 19_Status_Effects). Had D died (10's D survives at 1 HP),
   D's owner gains points equal to D's stars (08's D is 4★) and refills the slot (17_Death, 16_The_Field).
8. **D, B, A, E, F** likewise; then (ENDGT) effects, +1 BRICK each, the counter ticks (08_Global_Turn, 04_Table2).

## 13. Glossary

| Term | Meaning (page) |
|---|---|
| Action | one per creature per turn: attack, mine, defense mode (09_TakingYourTurn) |
| BRICKS / points / star points | the three counters: BRICKS buy field cards; points pay abilities and the item shop; star points play items (04_Table2) |
| Facedown items | 3 items behind the slots, playable any time (13_Items) |
| Prime creature | a non-minion creature (15_Minions; never defined) |
| Primary / secondary field | front and back of a slot (16_The_Field) |
| Proc / instance | trigger / one application of damage or healing (never defined) |
| Reaction | a play that resolves before what it answers (12_Reactions) |
| Swap out / in | creature back to hand, another in; 2 points as its turn starts (08_Global_Turn) |
| Tier (debuff) | its star level (inference) |
| Cooldown (nCD) | card term, not in the rulebook; reading (inference): turns before the ability is usable again (cards; website tooltip "Cooldown: 2 turns") |
| Global cooldown (nGCD) | card term; reading: the same in global turns |
| a/b cooldown (a/bCD, a/bGCD) | card term; reading: a first wait, then a repeating wait (website, "Cooldown: 1 turn, then 2") |
| Uses (nU) | card term; reading: uses per cooldown (Time Riddle); the website groups it under "Per game & triggers"; when uses refill is open |
| Charge, recharge, usages | the cards' words for cooldown progress and remaining uses (Michael, Energy Ritual, The Bank) |
| Time based | every 🕐 tag (02_Abbreviations); Time Riddle also applies it to the cooldown tags (Energy Ritual by inference) |

## 14. How to answer a rules question

1. **Find the page.** 01 Getting Started, 02 Abbreviations, 03 Setting Up, 04 Table(2–5), 05 Shops, 06 Reading
   Cards(2), 07 Stats1–5, 08 Global Turn, 09 TakingYourTurn, 10 Attacking(2), 11 Abilities(2–7), 12 Reactions(2–5),
   13 Items, 14 Item Types(2), 15 Minions, 16 The Field(2), 17 Death, 18 Field Cards, 19 Status Effects(2), 20 Creature
   Types, 21 Damage Types, 22 Negation. Sleepy Kirby marks "more technical rules" (00_Table_of_Contents).
2. **Quote and cite** as (NN_Stem), in the rulebook's words. If pages disagree or the book is silent, say so and point to
   Known gaps. Status effect and type effects are card text — say so.
3. **Where the text lives.** Source of truth: the PSD text layers in `Auxiliary/Rulebook/` (PNG beside each); published
   as `site/dist/data/rules.json` (`{"rules": [{id, name, ability, …}]}`) and the website's Rulebook tab; `stats.pkl`
   holds the same 48 entries. Verbatim extract with hidden/overflow markers: `.claude/skills/impossibility-simulator-rules/reference/rulebook.md` (beside it: `glossary.md`, every defined term and abbreviation with its page; `data-facts.md`, the card-data counts behind this file; `header.md`/`footer.md`, build_rulebook.py's inputs).
4. **The bot.** `/query_rulebook search_text [limit]`: case-insensitive substring search over the 48 pages, sorted by page
   name, lowercased, icons as emojis; `search_text` is required by the slash command (main.py), so only a whitespace-only
   search reaches the engine's all-pages path. `/export_rulebook`: all pages as a text file.
   `/query_ability` searches card text (`type==debuff & stars==3`, `(1gcd)`; DSL in impossibility-cards-reference §7).
   It searches the bot's `stats.pkl`, which carries the new tags since PICKLE `3888392` (header).
5. **Text players never see** is in rules.json/stats.pkl/`/query_rulebook`: hidden duplicates (09), a hidden PASSIVES
   block (11_Abilities), an off-canvas "ABILITIES 7" header (11_Abilities6), an older reactions draft with order 4>2>1>3
   (12_Reactions2; the rule is 4>2>3>1), an older ITEMS block (22_Negation), overflow text (12_Reactions "of 5.",
   14_Item_Types "if you find this text dont delete it", 20 GIRL sentence); "[support]" where the page shows the machine
   icon (14); stray icons after page numbers (14, 14.2, 15, 16, 19.2, 20); lost ★ glyphs (08) and minion-icon gaps (15).
   The 11_Abilities example box is an image ("(OYT) Spend 3 points to cure a tier 2 debuff." / "(AAT) Spend a point to
   heal this creature by 1 HP.").
6. **Vocabulary.** "creatures" not units, "minions" not summons, "items"/"item types" not spells or subtypes (`subtype`
   is only the bot's field name), and likewise field cards, status effects, buffs, debuffs, stars, types, abilities,
   timing tags, global turn, starting hand.

## Known gaps and ambiguities

Contradictions:
- NORMAL "DMG - (1/2*DEF)" (21_Damage_Types) vs "Attacks deal ATK - DEF" (10_Attacking, 07_Stats3): an old-rules leftover
  (`MDW/Old Rules/17_Damage_Types`, when DEF was the bar value); current pages and examples use ATK − DEF.
- 12_Reactions3 "5 stacks of hardened, giving it 3 defense": Hardened is +0.5/stack, so 6 (as 12_Reactions says).
- 07_Stats3 text "-1 DEF BAR --> 1"; its picture shows 1.5.
- Hidden drafts (12_Reactions2 order 4>2>1>3; 22_Negation ITEMS) disagree with the visible pages, which win.
- Resolved 2026-10-24 (owner): status DMG/heal and durations tick in phase 4, after (ENDT) effects; 19_Status_Effects
  now says so, as 09_TakingYourTurn did.

Undefined: the cards' new tags (nCD), (a/bCD), (nGCD), (a/bGCD), (nU) since 2026-10-21: undefined by the rulebook; card
evidence in §3. The text of 02_Abbreviations, 11_Abilities2, 11_Abilities4 and 22_Negation still uses the old tags, and
the pictures show old card text: Toy Bonnie's (OPGT) on 12_Reactions–12_Reactions4, Whispy Woods' (OPT) on 12_Reactions4,
OMORI's (EOT) and Computer Virus' "(CD)" (counter damage, as in Old Rules) on 01_Getting_Started. Open within that: how a cooldown counts (the cards
imply once every n turns), whether (nCD) implies (OYT) and (nGCD) implies (AAT) (the cards imply so), when uses refill,
what the -1 of (-1/1GCD) means, and which new tags the multi-turn rules (11_Abilities4, 22_Negation) now cover. Also
undefined: "prime creature"; "tier"; "proc", "instance", "on-hit", "reactively", "cleanse", "innate actions/type";
"N.M.E."; the "Don't be stupid" rule.

Silent: how a hand is built from the full or developer/baseline decks (new players draw from the simple-version decks,
04_Table5; 04_Table describes the developer and "Baseline" decks); hand limits; summoning from hand except to fill an empty slot;
whether creatures in hand must be defeated to win; starting BRICKS; which star level breaks a lane tie; whether combo
damage applies to sneak attacks or ability DMG; how "25% +1" rounds; unspent star points (implied lost); "one ACTION" (09)
vs "2 innate actions" (11_Abilities7) and Inspired (+1 Action); "5+ star creatures" vs decks topping out at 5★; COUNTER as
item type (14) vs damage type (21); the `Auxiliary/Creatures/` cards; status effects
are defined only on cards (the book names ~22 of 64).

## When NOT to use this skill

| Task | Use instead |
|---|---|
| How card DATA is encoded: folders, PSD layers, stats extraction, type injection, pickle schemas, the /query DSL | impossibility-cards-reference |
| The icon font, [type] tokens, timing-tag badges, the card website or deck builder, anything in ~/Desktop/TTSCardMaker | ttscardmaker-companion |
| Running the bot, /update_stats, pushing pickles (which also republishes rules.json) | autocroissant-run-and-operate |
| A card whose ability text or type parsed wrong | autocroissant-psd-extraction-campaign, autocroissant-diagnostics-and-tooling |
| Help text, describe strings, skill-writing conventions | autocroissant-docs-and-style |
| An AI that answers rules questions (RAG, /ask) as a future feature | autocroissant-research-frontier |

## Provenance and maintenance

| Fact | Re-verify with |
|---|---|
| 48 pages, PSD per PNG | `ls ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/*.psd \| wc -l` (and `*.png`) |
| rules.json count and any page's text | `python3 -I -c "import json;r=json.load(open('$HOME/Desktop/TTSCardMaker/site/dist/data/rules.json'))['rules'];print(len(r));print(r[0])"` |
| stats.pkl == rules.json | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/loadpkl.py ~/Desktop/AutoCroissant/stats.pkl /tmp/s.json` (stub loader, no psd_tools needed) then compare `ability` per name |
| Hidden / off-canvas layers | `python .claude/skills/impossibility-simulator-rules/scripts/alllayers.py <page.psd>` from a venv with psd-tools (the system `python3` lacks it; autocroissant-build-and-env §1) |
| Tag counts (§3, 2026-10-22) | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/tags.py <cards.json or loadpkl.py dump>` (matches (1/2CD) and (-1/1GCD); prints a total line). `site/dist/data/cards.json` is rebuilt only by a PICKLE push: before trusting it, check that tags.py shows no (1PT)/(1PGT)/(E2T). Between `/update_stats` and the push, run it on the loadpkl.py dump of the bot's `stats.pkl` (same tag counts; the dump lacks only the two PNG-only cards, which have no text) |
| Card evidence (§3 "What the cards show") | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/lines.py <cards.json or dump>`: lines per family with (OYT)/(AAT), the tag before each (nU), line starts, cooldown tags after other text, (a/bCD) cards |
| The 2026-10-22 inputs and the per-card old → new mapping | scratch under `/tmp/rules_refresh` (gone on reboot). Sandbox = `git archive` of AutoCroissant origin/main plus config.py and the three pickles; from the psd-tools venv `python local_sweep.py <sandbox> ~/Desktop/TTSCardMaker` runs the bot's own `update_stats(force_update=True)` over the clone's HEAD blobs with the network blocked (about 2 min; 813 PSDs, 819 entries, 0 errors) → `stats_now.pkl`; `python build_cards.py <stats.pkl> <old_stats.pkl> <aliases.pkl> <out.json>` → `cards_now.json` (and `cards_before.json` from PICKLE 56b43b9) with build_site.py's card loop, offline; `changes.py` → `changes.md`/`changes.json` (`line_tag_mappings`, `tags_removed`, `tags_added`). Michael's Blessing reorders its rows, so read its mapping by meaning, not by line position |
| Status effect texts, mentions | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/mentions.py <cards.json or dump>`; or `/query_ability type==debuff` |
| Card counts by kind/stars/type (cards.json: 773 rows, 298 creature rows, 46/103/93/56; 2026-10-22) | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/counts.py <cards.json or dump>`, `python3 -I .claude/skills/impossibility-simulator-rules/scripts/agg.py <cards.json or dump>`; on the dump they give the stats.pkl counts (297 = 45/103/93/56). `scripts/rows.py` reads both shapes for all four |
| stats.pkl counts used above (297 = 45/103/93/56, 75 untyped, per-type distinct creatures, 182 item PSDs) | after the loadpkl.py line above: `python3 -I -c "import json,collections as c;s=json.load(open('/tmp/s.json'));cr=[v for v in s.values() if v['card_type']=='creature'];print(len(cr),sorted(c.Counter(v['stars'] for v in cr).items()),sum(1 for v in cr if not v['types']),c.Counter(t for v in cr for t in set(v['types'])),len({v['path'] for v in s.values() if v['card_type']=='item'}))"` (`set()` per creature, not icon occurrences) |
| Ban list (53); moved from `MDW/` to `site/` 2026-10-20 | `grep -c ' - ' ~/Desktop/TTSCardMaker/site/Constructed\ Ban\ List.txt` |
| Website hand rule | `grep -n "HAND\|FIELD_MAX_STARS" ~/Desktop/TTSCardMaker/site/assets/deck.js` |
| Slash commands | `grep -n 'name="query_rulebook"\|name="export_rulebook"' main.py`; `grep -n rulebook_dff commands/query_card.py` |
| Rounding table (07_Stats5: 20 bases, then 6 change columns of 20 cells, then the page number) | `python3 -I -c "import json,re,math;t=[p for p in json.load(open('$HOME/Desktop/TTSCardMaker/site/dist/data/rules.json'))['rules'] if p['id']=='07-stats5'][0]['ability'].split('+75%')[1];n=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',t)];b=n[:20];c=n[20:140];print(sum(math.floor(2*x*(1+k)+0.5)/2!=v for (k,x),v in zip([(k,x) for k in(-.25,-.5,-.75,.25,.5,.75) for x in b],c)),'mismatches')"` (prints `0 mismatches`) |
| Old rules | `ls ~/Desktop/TTSCardMaker/MDW/Old\ Rules/` (22 PSD+PNG pairs) |
| TTS save facts (data-facts §7) | copy `~/Desktop/TTSCardMaker/TTS/audit.py` to a scratch folder, set `OUT` to that folder, `REPO` to the clone and `RULES` to `OUT/decks.json`, run it with `python3 -I` (it writes only to `OUT`); the committed `TTS/AUDIT.md` is the 2026-10-20 run |

When a rulebook PSD changes: rebuild the site, run `/update_stats`, regenerate `reference/rulebook.md` with
`python3 .claude/skills/impossibility-simulator-rules/scripts/build_rulebook.py` (no `-I`: it imports the sibling `notes.py`; reads `reference/header.md`, `footer.md` and `SRC`, the Mac clone's `site/dist/data/rules.json`), re-check Known gaps, update dates. When cards change: after `/update_stats`, dump the bot's `stats.pkl` with
loadpkl.py and rerun tags.py, lines.py and mentions.py on the dump (or on the site's cards.json once pushed); refresh §3
and data-facts §4. If the ban list moves: update §2.

2026-10-20 (later): the scratch folder `~/Documents/rules-skill/` moved into this skill: scripts to `scripts/` (notes.py holds
build_rulebook.py's per-page annotations; layers.py lists the hidden text layers of several pages, `python scripts/layers.py <dir> <page.psd>...`
from the psd-tools venv; the one-shot patch scripts that shaped this file were not kept), rulebook.md, glossary.md,
data-facts.md, header.md and footer.md to `reference/`.

2026-10-22: re-measured after the cooldown update (TTSCardMaker `73904dbd`, AutoCroissant `39da9ed`); PICKLE `3888392`
pushed the same data, and the scripts give the same counts on its `stats.pkl` dump and on the site's cards.json rebuilt
at 00:43 (TTSCardMaker `2291b5de`, a site-only commit; no PSD changed after `73904dbd`). The scripts read
cards.json or a loadpkl.py dump through `rows.py`; `lines.py` is new; `tags.py` matches (a/bCD) tags; `loadpkl.py`
dumps stat lines as dicts.

2026-10-22 (Action): (Action) became a grey `neutral` badge (TTSCardMaker `f6ce500e`); §"Timing tags (badge_tags)", the
glossary count and the "Other parenthesised words" line of reference/data-facts.md, and the reading note above, updated.
