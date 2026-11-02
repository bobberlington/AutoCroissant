---
name: impossibility-simulator-rules
description: "Load to explain how the Impossibility Simulator card game is PLAYED: \"how does the game work\", \"explain the rules\", what a timing tag means ((1CD), (1GCD), (2U), (1/2CD), (OFS), (WOF); the old (1PGT)/(E2T) and what replaced them), cooldowns, charge, uses, time based, multi turn, turn order, pair turns, global turn, turn phases, damage (ATK - DEF, combo damage, sneak attack, counter damage, rounding), minions, prime creature, primary/secondary field, field cards, item types, N.M.E. shop, star points / points / BRICKS, reactions and resolve order, negation, status effects (stacking, upgrades), creature types and immunities, damage types, death, defeat and revives, starting hand / constructed / ban list, \"which rulebook page or chapter says X\", rule changes from Discord (pending-updates), or to check a /query_rulebook answer or wording in the game's own vocabulary. Contains the game in one page, every timing tag and the charge rules with card evidence, attack arithmetic, items, reactions, 64 status effects, glossary, Known gaps, chapter citations; the authority is the website's chapters, site/rulebook. Card DATA encoding: impossibility-cards-reference."
---

# Impossibility Simulator: the rules of the game

The game's rules live only in the website's chapters, `~/Desktop/TTSCardMaker/site/rulebook/NN-slug.md` (24 files:
Contents + 23). They are the authority: quote them. This file was checked against them at TTSCardMaker `c6b1e144`
(2026-11-01). A citation like (08) or (ch 08) is the chapter number. The Photoshop pages are history in `MDW/Old Rules/`
(`Rulebook/` the 48 pages, `Very Old Rulebook/` the first rules). Where the chapters are silent or disagree this skill says
so under "Known gaps and ambiguities"; never fill a gap by guessing. Use the rulebook's words only (creatures, minions,
items, item types, field cards, N.M.E. items, status effects, buffs, debuffs, stars, types, abilities, timing tags, global
turn, pair turns, slots, lanes, primary/secondary field, points, star points, BRICKS, shops, starting hand, facedown items).

**Card data.** Tag and status effect counts are over the cards.json shape of 2026-10-22: 773 rows, which is 767 distinct
cards plus 6 `dup` copy rows, and 771 distinct names (`tags.py` drops the dup rows and dedupes on psd/png path + name), from
a forced sweep of TTSCardMaker `73904dbd`. Counts by kind, stars and type are `stats.pkl` PSD counts, unchanged from
2026-10-20 (cards.json adds two PNG-only cards, Stubborn Queen and Gust of Trees). Today's numbers differ by a few cards:
rerun the scripts (Provenance). Card names here are file names; the bot's parser v2 stores printed titles, which differ from
the file name for 159 cards (165 rows with the copy rows): 130 changed with this parse, 29 (the Clots, the Ingredients and
others) were already renamed in `site/Card Edits.json`. Blockade prints "The Wall By Ech". The bot and the website carry the
same tags and titles: `/query_ability (1gcd)` finds 82 cards and `(1pt)` none (in the rulebook, old tags are only in ch 23's
conversion table).

## 1. The game in one page

- **Goal.** "Defeat all of your opponent's creatures, while making sure they don't defeat all of yours!", using
  "CREATURES and ITEMS to attack and protect" (01). The "Don't be stupid" rule is named, never stated.
- **A team holds** a starting hand of 7 creatures and 10 items, 2 points and 7 star points (03, 13).
  Three counters per team (04): **star points** (set to 7 each global turn; items cost their stars), **points** (start 2;
  gained equal to a creature's stars when it is defeated; pay abilities and the item shop), **BRICKS** (mining and 1 per
  team per global turn; buy field cards).
- **The table.** 6 slots, 3 per team, in three lanes; beside each slot HP/DEF/ATK/SPD counters and status effect
  markers, behind it a holder with one facedown item (16, 04, 03). Around it: decks sorted by stars, three shops, graveyard,
  discard pile, field card slots, auxiliary decks, markers, 0.5 chip, and for RNG the Redirection cube and Order Determinator
  (04; 08 breaks a last tie with "flip a dice, or roll a coin").
- **Flow.** Slots take turns one at a time in *pair turns*, "a slot from one team, then a slot from the opposing team",
  fastest first; once every slot has gone a **global turn** has passed: star points reset, +1 BRICK each (08). A turn gives
  the creature one **action** (attack, mine, defense mode) plus (OYT) abilities and items (09). "When a play occours [sic], a
  player may choose to react to it with one of their items or abilities" (12): (AAT) abilities "at any time" (11), facedown
  items "can be played at any time" (13, 04), hand items "reactively only if it's currently your turn" (12). A creature at
  0 HP dies; unless revived on the spot it is **defeated**: it goes to the graveyard and a prime creature gives ITS OWNER
  points (17). "You must immediately summon a creature into an empty slot if possible" (16; the chapter does not say where
  the creature comes from).

## 2. Setup and the constructed starting hand

| Starting hand (03) | 5★ | 4★ | 3★ | 2★ | 1★ | Total |
|---|---|---|---|---|---|---|
| Creatures | 1 | 2 | 3 | 1 | — | 7 |
| Items | — | 2 | 3 | 3 | 2 | 10 |

Each team fills its 3 slots with creatures face down and puts 3 "Facedown items" one behind each slot; creatures flip up
when everyone is ready; items are revealed only when played; only your side sees your facedown items (03).

| Shop (05) | Stock | Price | Restock |
|---|---|---|---|
| Item shop | 2 items per star level, revealed | 4x stars in points (2★ = 8) | same star level; reshuffle once per global turn for 1 point |
| Field shop | 2 field cards per star level; 1 per level revealed, the rest at the end of global turn 1 | 2x stars in BRICKS | none |
| N.M.E shop | 4 random N.M.E. items, revealed | "written on each card" | random N.M.E. item |

**Where the hand comes from.** New players draw: the table holds a "simple version of the game", creature and item
decks with the "More complex creatures an items" removed and no field cards or N.M.E. items, and "New players will draw
their creatures and items from here" (04). Also on the table (04): the DEVELOPER DECK, "a reduced version of the full deck"
of "newer or recently changed cards that require more playtesting", mixed with "Baseline" creatures "considered balanced";
"For most players you can ignore this", it is for "a more wacky experience". How a hand is built from the full decks is not
stated (Known gaps). **Constructed** is defined only in `~/Desktop/TTSCardMaker/site/Constructed Ban List.txt`: "choose
whatever cards you want out of the entire deck for your hand", "You may also only have one of each item in your hand", and
each player may pick "one 3 star or lower field card to be active in the shop at the start of the game". It bans 53 cards
(creatures 8/8/9/4 for 5★–2★, items 8/8/3/1 for 4★–1★, 4 in its group "Field Cards") for "creature/ability copying",
"antifun", "reliance on RNG", "too easy to build combos", "too powerful of a counter". The website's deck builder
(`site/assets/deck.js`) enforces the same counts (`HAND`), one of each card, no banned card, and an optional field card with
`FIELD_MAX_STARS = 3`; only creatures and items fill slots. List-vs-data mismatches: the deck builder's one-copy rule covers
creatures too, where the list says only "one of each item"; Tainted Bethany is 4★ (listed 3★), NME Stock 2★ (listed 1★), The
Bank is an N.M.E. item (listed under Field Cards), Cave of the Past and Error Room are 4★ field cards. Names: "The Wall (By
Ech)" is Blockade.psd, printed title "The Wall By Ech"; "Leeroy Jenkins" prints as "LEEROOOOOOOOY JENKINS"; "Berserk" is "Berserk!".

## 3. Turn structure and every timing tag

**Global turn.** "When every slot has taken its turn once, that is called a GLOBAL TURN. When the global turn resets,
each team is set to 7 Star points. At the end of each GLOBAL TURN, each team gains 1 BRICK" (08). The counter starts at 1 and
"ticks up by 1 once every filled slot on the field has taken a turn" (04). A dead slot is skipped "but count time as if that
slot had gone" (16).

**Turn order** (08): "Turns are taken in *pair turns*: a slot from one team, then a slot from the opposing team." The fastest
creature goes first, then the opposing team's fastest; then the next fastest, then the opposing team's next fastest; then the
fastest remaining, then the opposing team's fastest remaining. "Only the PRIMARY FIELD counts. A slot's speed is its fastest
creature's." Ties: the slot with the least HEALTH LOST (MAX − CURRENT); then the highest STAR LEVEL; then RNG. "MOST SPEED >
LEAST HEALTH LOST > HIGHEST STAR LEVEL > RNG" (worked in §12). A lane is still a place on the table (04), not a turn. (AAT)
speed changes and facedown items may change which slot goes next. As a slot's turn is about to start its creature may swap
out for 2 points, 1 less if WIND (20): "This cannot change turn order." A creature in the middle of its turn when the order
changes finishes it. **No stalling:** "During a full global turn, if none of a team's creatures attack, those creatures take 1
TRUE DAMAGE at the end of the global turn" (08).

| Phase (09) | What happens |
|---|---|
| 1 | The slot's turn begins; (BOT) effects |
| 2 (main) | (OYT) abilities; items from hand; swap facedowns to/from hand; shops and item-shop shuffle; actions; swap primary/secondary fields if no action used yet |
| 3 | (ENDT) effects; nothing from phase 2 any more |
| 4 | Turn over; status effects deal damage/heal, durations tick down 1 (also 19) |

**The action**: one per creature per turn, "Actions can only be used (OYT)": ATTACK any creature on the field (§4), MINE
(BRICKS = stars; prime creatures only, "Minions cannot mine", 15, 18) or DEFENSE MODE (DEF doubled until its next turn begins;
card turned 90°) (09). Every minion and parent creature in the primary field "may use their actions", in whatever order the
player chooses (15).

**Timing tags** (02, 11, 23). "X" "can represent any number"; cards print the number. 🕐 = "time based", ⏳ = "multi turn" (below).
Counts = distinct cards before the 2026-10-21 tag update (stats.pkl PICKLE 56b43b9; every tag: 606 uses in 352 of 767 cards, 625 uses in 352 after it) for the unchanged tags; the families' counts
are in the table after this one.

| Tag | Rule | Cards |
|---|---|---|
| (OYT) | On your turn: wait "until the creature with this ability's turn" (11) | 42 |
| (AAT) | At any time; usable reactively (11) | 13 |
| (XCD) 🕐 | Cooldown: needs X turn(s) of charge; implies (OYT); "the most common ability type" (02, 11) | below |
| (Y/XCD) 🕐 | The same, starting with Y turn(s) of charge; Y "can also be 0 or −1" (02, 11, 23) | below |
| (XGCD) 🕐 | Global cooldown: needs X turn(s) of charge; implies (AAT); gains charge "at any point during the global turn" (02, 11) | below |
| (Y/XGCD) 🕐 | The same, starting with Y turn(s) of charge (02, 23) | below |
| (XU) | "X use(s) per full charge" (02, 23) | below |
| (XPG) | X time(s) per game, "ever"; always also (AAT) or (OYT) (02, 11) | (1PG) 30, (2PG) 4, (3PG) 3, (XPG) 2 |
| (BOT) 🕐 forced | Beginning of turn, "before it does anything else" (11); phase 1 | 24 |
| (ENDT) 🕐 forced | End of turn, "after everything else it can do" (11); phase 3 | 26 |
| (BOGT) 🕐 forced | Beginning of global turn, "before anything else in the global turn" (11) | 39 |
| (ENDGT) 🕐 forced | End of global turn (02; 11 shows it inside an example) | 37 |
| (OFS) forced | On first summon from the hand; cannot be negated or true negated; counts as (1PG) (11) | 34 |
| (OS) forced | On summon, every time (11) | 8 |
| (OD) forced | On death, forced (11); "does not apply to creatures not on the field" (17) | 23 |
| (WOF) forced | While on field: passives, most "in constant effect (WOF)"; "“(WOF)” is often only included when the ability effects other creatures as to signfy that it goes away when the creature exits the field" (11). A (WOF) line can carry a cooldown: Michael "(WOF) Discarded items may be bought for 3/4 shop price (1CD)." | 20 |
| (Action) | "Costs the creature's action" (23); a grey badge | §7 |

**Charge** (11). "***X*** is how many turns of charge the ability needs to be used (it can't go over ***X***). Abilities start
with no charge. 1 turn of charge is gained at any point during phases 1-3 of the creature's turn, if the ability hasn't gained
charge yet that turn and hasn't been fully charged at any point during that turn." So (1CD) may be used once on each turn the
creature takes. (XGCD) "gains charge at any point during the global turn, under the same conditions". (2CD): "on its second
turn, and every 2 turns after that. (Turns 4, 6, etc.)". (1/2CD): "starts with 1 of 2 turns of charge, so it may be summoned on
the first turn, then every 2 turns after. (Turns 3, 5, etc.)". **Uses:** "(XU) is how many uses the ability has when charged. If
even one use is spent the ability can gain charge again; charged again without spending every use, it only goes back up to
***X*** uses. No (XU): 1 use per full charge." **Saved and paused:** "If you forget to use your time based ability, it's saved so
you can use it on the next turn, but its cooldown doesn't reset until it's used"; abilities "do not count time passing while in
the hand" (11); "Cooldown abilities such as (2CD) still gain charge even while they are negated" (22); a lost and regained
stolen or swapped ability "gains 1 turn of charge" (11); a revive resets charge (17).

**Time based, multi turn** (02). "Abilites marked with 🕐 are considered "time based". Time based abilities with a cooldown
become usable again once they are fully charged." "⏳ (XCD), (Y/XCD), (XGCD) and (Y/XGCD) abilities with ***X*** of 2 or more are
considered "multi turn" abilities." Forced tags are "non-optional and assumed to happen automatically" (11). "You may react to
time based [such as (BOGT)] effects with any (AAT) effects or facedown items" (12); nothing is stated for (OFS)/(OS)/(OD), and
(OFS) abilities have true negation immunity (22).

**Old tags.** Ch 23's OLD SYSTEM table is their only place in the rulebook, and on no card since 2026-10-21. The team
replaced them on every card: TTSCardMaker `df02b3b6` "cooldowns (2)" (50 PSDs), `09131c67` "(3)" (49), `1b4f36a0` "(4)" (50) and
`7cb4c867` "(5)" (29), all by Zomoroko on 2026-10-21, 178 PSDs with their PNGs (no "(1)" exists): 141 creatures, 17 items, 14
field cards, 4 N.M.E. items and 2 aux items (no status effect card); 172 changed tags only, 6 also wording (data-facts §4).
Counts below = uses / cards in the parsed ability text (fresh sweep of `73904dbd`, 2026-10-22; `tags.py`).

| Family | Meaning (23) | On the cards (uses / cards) | Replaced, per PSD |
|---|---|---|---|
| (nCD) | needs n turn(s) of charge | (1CD) 101/88, (2CD) 55/55, (3CD) 2/2, (XCD) 1/1 | (1PT) → (1CD); (EnT) → (nCD): (E2T) → (2CD), (E3T) → (3CD), (EXT) → (XCD) |
| (a/bCD) | starts with a of b turns of charge | (1/2CD) 15/15 | (FTE2T) → (1/2CD) |
| (nGCD) | global; needs n turn(s) of charge | (1GCD) 94/82, (2GCD) 11/11, (XGCD) 1/1 | (1PGT) → (1GCD); (E2GT) → (2GCD), (EXGT) → (XGCD) |
| (a/bGCD) | global; starts with a of b turns of charge | (-1/1GCD) 1/1 (The Temple of Ahn'Qiraj, as (-1/1GCD)(OYT)); (1/2GCD) on no card | (-1/1CD) ("1PT except first turn" in 23); FGTE2GT → (1/2GCD) in 23 |
| (nU) | n use(s) per full charge | (2U) 18/17, (3U) 2/2, (4U) 2/2, (XU) 2/1 | the n of an (nPT)/(nPGT) with n ≥ 2: (2PT) → (1CD)(2U); (2PGT), (3PGT), (4PGT) → (1GCD) with (2U), (3U), (4U); (XPT)/(XPGT) → (1CD)/(1GCD) with (XU) (Michael's Blessing) |

Unchanged by the update: (AAT), (BOT), (ENDT), (BOGT), (ENDGT), (OFS), (OS), (OD), (WOF) and (nPG) (cards only: (1PG) 32/30, (2PG)
4/4, (3PG) 3/3, (XPG) 2/2); (OYT) went from 38 to 42 cards because the 4 cards that moved to a global cooldown gained it. 218
cards carry a new tag: 173 had only old tags, 45 already had one of the former undefined tags. The tag sets of 214 follow the
last column exactly; 4 also moved from turns to global turns and gained (OYT): Water Totem (1CD) → (1GCD)(OYT), The Gates of
Ahn'Qiraj (2CD) → (2GCD)(OYT), Crock Pot (1PT) → (1GCD)(OYT), The Temple of Ahn'Qiraj (-1/1CD) → (-1/1GCD)(OYT). The Bank (N.M.E.,
`7cb4c867`) lost its only cooldown-family tag: its line "Will store individual uses of a multi-use (XPT) abilities." was deleted,
and its cooldown line now reads "Cooldowns of stored actions/abilities recharge as if they were used." It keeps (ENDGT), so it is
not among the 218. Website tooltips are ch 23's meanings (`build_site.py` GAME_TAGS and `tag_meanings`, `site/assets/text.js`):
(1/2CD) "Cooldown: starts with 1 of 2 turns of charge", (2CD) "Cooldown: needs 2 turns of charge", (2U) "2 uses per full charge".

**Old card text in the pictures.** Ch 12's pictures still show Toy Bonnie as "(OPGT) Inflict up to 6 stacks of Softened for 1 turn
..." (an O, not a 1; the card has read (1GCD) since before the update) and Whispy Woods as "(OPT) You can spend 2 points ..." (today
"(1CD)(2 points) ..."). (OPGT) and (OPT) belong to neither system; "(OPT)" otherwise survives only in a hidden `old` layer of
Bobberlington and Michael.

**What the cards show (card evidence; the chapters now state the rules).** Measured on the fresh parse (`lines.py`). Verbatim
texts and the per-line mapping: `reference/data-facts.md` §4.
- **Turn vs global, (OYT) and (AAT).** (nCD) and (a/bCD) never share a line with (OYT) or (AAT). Global cooldowns share a line
  with (OYT) on 10 lines (8 start with both, e.g. "(1GCD)(OYT)", "(1GCD)(4U)(OYT)", "(-1/1GCD)(OYT)"; plus Jetpack's mid-sentence
  "(OYT)" and a Michael's Blessing row), never with (AAT). Exactly the 4 turn tags that became global cooldowns gained (OYT).
- **Uses.** (nU) comes right after (1CD) or (1GCD) on all 22 cards that carry it, never after (2CD) or longer. Line order is
  cooldown, uses, then cost or (OYT): "(1CD)(2U)(1.5 points)", "(1GCD)(4U)(OYT)", "(1CD)(3U)(2 points)". Supreme Jeh'Oul: "this
  ability becomes and stays (1CD)(2U)". Items still write uses in words ("3 uses:", "Uses: ∞"): those are multi-use items (13,
  14), not the (nU) tag.
- **(a/bCD).** (1/2CD) is on 15 creatures, 14 from (FTE2T) (Chris & Jordi had it already). 8 pair it with a (2CD) ability (Chris
  & Jordi, Daroach, Dark Daroach, Doctor Zomboss, Koops, The Abyss Watchers, Vivian, Yoshi Kid): ch 11's pair, "This ability
  happens on turns where the above doesn't".
- **Michael's Blessing** (N.M.E., 5★): "(OYT) -> (1GCD) / (BOT)/(ENDT) -> (1GCD) / (BOGT)/(ENDGT) -> (1GCD) / (XCD) -> (XGCD) /
  (1CD)(XU) -> (1GCD)(XU)". Its old last rows, "(XPT) -> (XPGT)" and "(EXT) -> (EXGT)", were in the other order, so (XPT) became
  (1CD)(XU) and (EXT) became (XCD).
- **Time Riddle** (4★ equip, banned in constructed): "Doubles all time-based and (XPG) ability usage, for example: (2CD) -> (1CD)
  / (1CD) -> (1CD)(2U) / (1PG) -> (2PG) / (1GCD) -> (1GCD)(2U)". Its first line, "Has no effect until 2 lanes have gone.", still
  counts lanes (Known gaps).
- **Recharge words.** Energy Ritual "fully recharges and restores all usages of all time-based and (XPG) abilities"; Michael
  "(2GCD) Recharge 1 turn of charge to another creature's ability" (was "(E2GT) Refresh another creature's ability usage as though
  another turn has passed"). Older text with the same idea: Cave Of The Past "All 🕐time-based effects and abilities do not
  refresh."; Hot Spring "refresh its own ability as if another turn has passed".
- **Tags mid-line.** 12 lines outside the two conversion tables carry a cooldown tag after other text: after a form or condition
  label (Warcraftian Druid "Bear: +1 DEF, (1GCD) Intercept ...", Pix, Yin Yarn, OMORI, Niko, Waddle Dee) or at a sentence's end
  (Corruption "Can only activate (1GCD).", Spice, Michael, Supreme Jeh'Oul). A tag at a sentence's end limits that clause (reading).

**Simultaneous effects** (11): higher-SPD creature first, then the higher-cumulative-SPD team's active items, then field cards;
ties use turn-order tie-breakers; the owner orders same-side conflicts; both teams setting one value "remains random";
simultaneous HP-setting revives add up (to Max); directly conflicting effects: the player chooses.

## 4. Stats: HP, DEF, ATK, SPD

Bars show MAX values, counters beside the slot show CURRENT values (06). One marker is one unit of the real stat ("1 DEF marker is
0.5 DEF, 1 HP marker is 1 HP"); "Flip them over for stat downs" (04); "For increments of 0.5, use a marker turned to its side" (07).
The 0.5 chip is for a counter that "uses half integers" (04). "A creature's base stats are the stats printed on its card, before
any changes. Stat changes affect its current stats, not its base stats, unless an effect says it changes a base stat" (07).

| Stat | One bar | Meaning |
|---|---|---|
| HEALTH | 2 HP ("8 bars … 16 HP") | "A creature perishes when its health reaches zero" (06); at 0 "it dies and, unless it is revived on the spot, goes to the graveyard" (07) |
| DEFENSE | 0.5 DEF ("4 Bars = 2 DEF", 07) | "Subtracted from any attack damage this creature takes" (06); "they take damage = ATK-DEF" (07) |
| ATTACK | 1 ATK | damage dealt "minus DEF when attacking" (07) |
| SPEED | 1 SPD | "determines turn order and combo damage" (07) |

**MAX HP** (07): regular healing stops at MAX; raising MAX raises current HP equally and "counts as healing" (Mushroom +3.5: 16 →
19.5); lowering MAX "does not decrease their current HP, unless if their current HP would become greater than their Max HP (in
which case it is set to Max HP). If its current HP was already greater than its Max HP, the decrease lowers its current HP by the
same amount." (16 → 14, but 10 stays 10) and "does not count as damage"; two heals from one item/ability are "one instance".

**Rounding** (07): "to the nearest half in the positive direction, *after* reductions or gains" (Chilled −25% on 5 SPD: 3.75 → 4;
Enraged +25% on 5 ATK: 6.25 → 6.5). Additives first, then multiplicatives; multiplicatives add together (−25% + −50% = −75%). The
table's 120 computed cells (6 change columns x 20 base values) are exactly "nearest 0.5, halves up". "SPD can't go below 1. ATK and
DEF can't go below 0. MAX HP can't be reduced below 1 bar (2 HP)." Damage rounds up: 3.25 → 3.5 (10).

**Attacks** (10): a creature's attack action, an attack item, or an ability dealing DMG separate from ATK. "Attacks deal ATK - DEF in
damage", primary field only by default. "Creatures with 0 ATK cannot declare an attack (of any type)." "Damage from abilities and
item effects cannot trigger counter damage as retaliation. Attacks from abilities cannot trigger counter items." COMBO DAMAGE:
attacker has "TWICE THE SPEED VALUE" of the target → x1.5; "TRIPLE" → doubled (reading these as "at least 2x / 3x" is an inference,
not chapter text). Examples: A (ATK 7, SPD 4) hits B (DEF 1, SPD 4): 7 − 1 = 6, B at 4/10. C (ATK 7, SPD 6) hits D (DEF 1, SPD 3): 6
x 1.5 = 9, D at 1/10. SNEAK ATTACK replaces a normal attack: piercing "25% of their attack stat +1", same on-hit effects; "Minion
sneak attacks are only 25% of their attack stat, without the +1." Multi-hit attacks "do not count as separate attacks. Each instance
of damage cannot be reacted to on its own, only the entire attack can"; "To make 3 separate attacks, a creature must attack 3
separate times." When an ability or item attacks "multiple enemy creatures at once" the attacks go left-to-right from the attacker's
perspective and on death effects wait until "all instances of the attack are resolved"; redirections retarget to a creature of
choice, interceptions to the interceptor (leaving defense mode); inflictions on offense or defense apply before the hit and, if from
the attack itself, cannot be separately negated/reflected/redirected/intercepted; only attack actions proc "on attack" effects (10).

## 5. Creatures, minions and the Auxiliary/Creatures cards

A creature card shows type icon (top left), stars (top right), four stat bars and the ability, "only active while on the field"
(06, 11). Decks hold 2★–5★ creatures (04; stats.pkl: 297 cards, 45/103/93/56). **Prime creatures** "are the creatures that fill
the slots: every creature that isn't a minion. A card with several forms or made of multiple creatures (like Mantis Lords) is one
prime creature. Prime creatures yield points when defeated." (16)

**Minions** (15) are "small creatures that when summoned are attached onto a parent prime creature"; "For most purposes, Minions are
considered creatures":

| | Prime creature | Minion |
|---|---|---|
| Source | hand / creature decks | auxiliary minions deck, via abilities (04) |
| Own action | yes | yes |
| Can mine | yes (18) | no |
| When defeated | its stars in points, to its owner; graveyard (17) | no points; still goes to the graveyard |
| Attachment | — | always to the parent, even when moved or swapped to hand |
| Parent defeated | — | "all minions in the slot die (excluding persistent minions)" |

Offensive minions share the parent's field; protective minions (icon file `Defensive_Minion`) take the other field, pushing the parent
and its offensive minions to the secondary field; a protective one summoned after the parent acted gets no action that turn;
**persistent** minions attach to the next prime creature in the slot and, alone, "must be killed for the game to be won by the
opponent". "5 star minions cannot be targeted by minion effects." "If something just says “summon”, it means onto the creature that
the ability is from." (15)

**The `Auxiliary/Creatures/` cards** (Azamoth the Possessed Vessel, Igor Roballtowski, Mettaton EX, Mettaton NEO, Militron (No Armor),
Morpho Knight, Shadow Queen; bot type `creatures`, plural) have no rule in the book; Azamoth is only a revive example (11). The
rulebook has no name for them; their PNGs are the regular creature size.

## 6. Items, item types, N.M.E. items and shops

Items "are your main resource besides creatures"; one costs star points equal to its stars and is discarded after resolving unless
multi-use ("goes back facedown or to the hand"); the 3 facedown slots play at any time; hand↔facedown swaps only on your turn; sell on
your turn for points = stars (facedown ones at any time; never a used item, equips included); random draws use a copy of the full deck;
"You cannot use items on creatures in the hand or graveyard. You can still unequip items from creatures in the hand." (13)
Ability-summoned items come from the auxiliary items deck (04). Star points are spent on play; an invalidated item returns and refunds
(12). Item types are the top-left symbol and "cannot be manipulated" (14); repo folder in brackets, counts from card data.

| Item type | Rule | Cards |
|---|---|---|
| HEALING ("food") [Heal] | heal creatures; not on machine creatures | 35 |
| ATTACK [Attack] | "do damage, and count as an attack" | 16 |
| NEGATION [Negation] | "These items “Negate” something, taking it out of play" | 5 |
| SUPPORT [Support] | "general use items with various effects" | 30 |
| COUNTER [Counter] | conditional, "can only be used reactively"; often multi-use | 9 |
| INFLICTION [Infliction] | inflict a status effect; count as an infliction | 10 |
| EQUIP [Equip] | passive effect on your creature, "Creatures do not inherit the abilities of equip items, they are only affected by them" (separate from innate abilities, and a negated creature's items are not negated: 14, 22); unequip on your turn (a negative-effect equip locks until the next global turn); returns to hand if the creature dies; stays on a creature swapped to hand; a harmless one may go on an enemy, who then owns it; effects may "Forcibly" unequip (14) | 65 |
| FRAGILE EQUIP (icon, no folder) | cannot be unequipped; discarded when the creature dies (14) | 11 by icon: 10 inside Equip's 65, 1 (Life Shroom) in Heal |
| ACTIVE [Active] | "placed in a slot" with a constant effect until destroyed, "considered on the primary field"; has HP and DEF but is not a creature: no equips, healing items or status effects; discarded at 0 HP (14). "something that hits everything on the PF hits them too. Something that hits all enemy creatures on the PF does not." (16) | 9 |
| DEPLOY [Deploy] | constant effect on a chosen slot (not "used on" that creature); cannot be attacked (14) | 3 |

The first six "by default are discarded after being played" (14; COUNTER items "tend to have multiple uses and are only discarded when
all uses have been used"); the last four "do not get discarded immediately" (14). A healing item used to deal DMG counts as an attack
item, and vice versa; effects that increase healing increase that damage (Chef Saltbaker, Toy Chica; 11). **N.M.E. items** (26) are
"not in the standard deck" and cost what the card says (05); "N.M.E." is never expanded. Items run 1★–4★ (stats.pkl: 182 PSDs).

## 7. Abilities, reactions, negation, and how ability text reads

An ability is "the main chunk of text at the bottom half of a creature"; when it can be used is "defined by abbrviations [sic]" (11), the
tags of §3. Chapter examples: "(1CD) Summon a Mini Bee.", "(1GCD)(2U) Heal a creature by 2 HP.", "(AAT)(3PG) Spend 7 points to deal 10
piercing damage.", "(ENDT) +5 DEF until (ENDGT)", "(OFS) Summon a Koopa.", "(WOF) Friendly creatures gain +1 DEF." (11). Points pay
ability costs (04). Stolen/swapped abilities "start as brand new per creature"; damage-counting abilities ignore overkill (11).

Reading card text: a type icon in the text IS the type (stored as `[fire]`, rendered as an icon or emoji); "X" is "a variable that can
represent any number" (02); "5T1 Poison" = 5 turns of 1 stack (19); "(Action)" marks an ability costing the action (36 uses in 28
cards, 37 in 29 counting one "(action)"; 23, a grey badge since 2026-10-22); PF = primary field, CDMG = counter damage (02).

**Reactions** (12): "When a play occours [sic], a player may choose to react to it with one of their items or abilities … the reaction
happens before whatever you reacted to." Nothing resolves until plays stop; then reverse order (4>3>2>1), except "anything that deals
damage resolves *last*" (Pebble 5 DMG → 6 Hardened "giving it 3 defense" → Bomb 7 DMG → Metal Shield resolves 4>2>3>1) and the original
play "will always be resolved at the very end" (5>4>2>3>1). Example: Mighty Pebble (5 DMG); Toy Bonnie reacts with 6 Hardened (+3 DEF)
and takes 2. A reaction that kills the target first refunds the attacker's item/action/ability; NO TAKE BACKS once reacted to; hand items
only on your turn, equips may be reactively moved on your turn, no shopping; a chain resolves before a new one begins. "If something is
targeted for negation (or destruction, in the case of field cards), its owner cannot reactively activate any (AAT) effects on that card."
(12, 22)

**Negation** (22): a negated attack "does not happen", action not refunded. A negated creature "is unable to use any of its own abilities
(unless otherwise stated), but may still use the abilities of any equipped items or minions, and can still use actions"; cooldowns keep
gaining charge; things already summoned remain; two sources negating each other at once are both negated. "5+ star creatures are immune
to negation"; True Negation beats that; True Negation Immunity beats even that — (OFS) abilities and "upon being negated" effects have
it. A negated item returns where it was, star points not regained, unsellable while negated; negated equip/active/deploy items return to
hand.

## 8. The field and field cards

Each slot splits into a PRIMARY FIELD, "the only field opponents can target by default, and the only field creatures can act in"
("everything in the Primary Field" = all 3 slots), and a SECONDARY FIELD that exists only when a protective minion or protective active
item pushes a creature back: "By default this field cannot be targeted by opponents", abilities but no actions; it comes forward when no
protective minions or protective active items are left in the primary field; the parent may swap fields with its protective minions once
per turn if no action was used in the slot (16). All protectors share one field; minions summoned by minions attach to the slot's prime
creature (16). Each slot with creatures "takes one turn during each global turn" (16).

**Field cards** (18) "have a constant effect". Bought with BRICKS (2x stars), playable at once for free, may return to hand and be replayed
for star points = stars; 1 in play per team; "cannot be sold, and items cannot be used on them". "Outside of using star points to play them
from the hand, field cards do not behave like items in any way. Anything meant for items cannot be used for field cards (such as Stopwatch
or facedown slots)." Counter-play at the start of the global turn or on your turn: 1x stars in BRICKS negates it until the end of the global
turn (not the turn it was played, not 2 turns in a row); 3x destroys it (also on 22). Data: 50 field cards, 1★–4★.

## 9. Status effects

"A status effect, is either a buff or debuff" (19). DMG/heal effects proc "in phase 4, after (ENDT) effects", others immediately; durations
count down in phase 4 (19, 09); status DMG is piercing, not an attack, one instance per unique stack ("5 stacks of Poison is one instance of
5 damage"); the owner orders them in phase 4; conflicting debuffs: "the highest star level takes precedence"; a creature that "started the
global turn with a status effect" whose global turn "ends without the creature having ended its turn" has the effect tick at the end of the
global turn, "even if the creature is in hand" (19). A creature that dies "loses all status effects" (17). **Standard** (icons
`buff`/`debuff`): one instance per creature; **stacking** (`buffstack`/`debuffstack`): any number; stack/upgrade/reapply → longest duration;
**upgradable** (`debuffupgrade`): Smouldering+Smouldering → Burning, +Burning → Blazing; Chilled+Chilled → Frostbite, +Frostbite → 1 turn of
Frozen; Sickness+Sickness → Plague; Frozen/Blazing/Plague block the prior debuffs except to refresh the duration, and only an upgrade
refreshes it ("need to upgrade to refresh duration", 19). Debuffs carry 1★–4★ (the undefined "tier"); buffs have no stars. Effects below are
the 64 status effect cards' own text (card data, not rulebook); (k) stacking, (u) upgradable, unmarked standard; "not X" = "can't affect [X]".

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

Spelling drift: chapter "Smouldering"/"Regeneration" vs cards "Smoldering"/"Regen".

## 10. Creature types and damage types

"Every type has a unique function, typically an immunity to a debuff" (20). Creatures = distinct stats.pkl creatures with the type (75 of 297
have none; Pandora, Kyle and Z!!!!! list their type twice in stats.pkl, so icon occurrences would give panda 26, unknown 11, girl 2); type
tokens mark "if a creature type changes" (04).

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
| GIRL | +1 point (OD); "may only be an innate type in special cases, but may still be added to creatures through effects" | 1 |

"Multiple damage types can be applied to an instance of damage, combining their effects." (21)

| Damage type (21) | Rule |
|---|---|
| NORMAL | "DMG - DEF" |
| PIERCING | "Ignores DEF" (status DMG and sneak attacks are piercing) |
| COUNTER (CDMG) | "Can only be dealt as retaliation to an attack, or by something that deals damage when it dies"; not an attack; bypasses abilities and protective minions; "Damage from abilities and item effects cannot trigger counter damage as retaliation" |
| LIFE STEAL | DMG after DEF is healed before the attack resolves; 100% = 1:1, 50% = 1:2, 200% = 2:1 |
| SHOCKWAVE | all other enemy targets on the primary field plus the secondary slot behind the target; not an attack |
| TRUE DAMAGE | ignores DEF; cannot be negated, redirected or reduced |
| OVERKILL | only when a card says so: carries to the next target the card names, reduced by its DEF; not an attack; other effects of the attack do not proc |

## 11. Death and winning

(17) **Dying.** A friendly creature at 0 HP dies: its "On Death" effects proc (only on the field), it "loses all status effects and all of
its items are unequipped". "If something revives it on the spot (such as a Life Shroom, or an (OD) ability that revives it or switches it to
its next form), it is not defeated." **Defeat.** A creature that dies and isn't revived on the spot is defeated and goes to your graveyard.
When a prime creature is defeated, "your team gains points equal to that creature's star level, all non-persistent minions in that slot die,
and the slot is empty so a new prime creature can be placed in it". "Outside of “On Death” effects, anything that says a prime creature “dies”
means it is defeated." **Revives.** Revived by an "On Death" effect: points only when defeated; revived by other means: points kept, the revived
version awards 0; a revive resets HP, stats, abilities, charge and counters to base unless otherwise specified. **Specifics.** Multi-target
creatures pay when all targets are defeated; "A creature being “destroyed” instantly by any effect takes true damage equal to its current HP";
a creature replacing one defeated mid-turn after using an action "can't use actions, but may use an ability". GIRL gives +1 point (OD) (20);
minions give none (15). Winning = all enemy creatures defeated (01), including a lone persistent minion (15). Creatures still in hand: not
stated.

## 12. A worked turn

Assembled from the rulebook's own examples; the joins are this skill's.

1. **Global turn begins**: star points → 7; (BOGT) abilities fire, higher SPD first (08, 11); BRICKS may be paid to negate/destroy a field
   card (18).
2. **Order** (08's example): team A, C, E against team B, D, F. A (8/10 HP, SPD 5, 3★), B (8/9, SPD 5, 5★) in lane 1; C (8/8, SPD 6, 4★), D
   (6/6, SPD 4, 4★) in lane 2; E (7/7, SPD 3, 5★), F (8/8, SPD 3, 3★) in lane 3. C is fastest, then the opposing team's fastest, B; next A,
   the fastest remaining, then the opposing team's fastest remaining, D; last E and F tie on speed and health lost, so E (stars) → "C > B >
   A > D > E > F".
3. **C, phase 1**: (BOT) effects; C's cooldowns may gain charge from here to phase 3 (09, 11).
4. **Phase 2**: the attack, using 10's own C/D pair, which is a different pair from step 2's (08's C and D print no ATK/DEF, and 08's D has
   SPD 4, so 6 is not double it: no combo damage): C (ATK 7, SPD 6) attacks D (DEF 1, SPD 3, 10/10 HP): 7 − 1 = 6, "C's speed is DOUBLE that
   of Creature D's" → x1.5 = 9, D at 1 HP.
5. **Reaction**: D's owner may answer with a facedown item or an (AAT) ability, which includes a charged (nGCD) ability (11). DEF gains resolve
   before the attack, damage items after other plays, the attack last (12); C cannot take it back (12). Had the attack defeated D when it resolved (10's D survives at 1 HP), D's owner
   would gain points equal to D's stars (08's D is 4★), its non-persistent minions would die and the slot would refill (17, 16).
6. **Phase 3**: C's (ENDT) effects, e.g. "+5 DEF until (ENDGT)" (11).
7. **Phase 4**: C's status effects proc and tick (09, 19).
8. **B, A, D, E, F** likewise; then (ENDGT) effects, +1 BRICK each, the counter ticks (08, 04); if none of a team's creatures attacked, those creatures take
   1 TRUE DAMAGE (08).

## 13. Glossary

| Term | Meaning (chapter) |
|---|---|
| Action | one per creature per turn, only (OYT): attack, mine, defense mode (09) |
| Base stats | "the stats printed on its card, before any changes" (07) |
| BRICKS / points / star points | the three counters: BRICKS buy field cards; points pay abilities and the item shop; star points play items (04) |
| Charge | turns of progress a cooldown needs; gained in phases 1-3, or during the global turn for a global cooldown (11) |
| Cooldown (XCD) | needs X turn(s) of charge; implies (OYT) (02, 11, 23) |
| Global cooldown (XGCD) | needs X turn(s) of charge; implies (AAT); charges during the global turn (02, 11, 23) |
| Y/X cooldown (Y/XCD, Y/XGCD) | starts with Y turn(s) of charge; Y can be 0 or −1 (02, 11, 23) |
| Dies / defeated | dies at 0 HP; defeated if not revived on the spot: graveyard, points, the slot empties. Outside (OD) effects, a prime creature that "dies" is defeated (17) |
| Destroyed (creature) | takes true damage equal to its current HP (17) |
| Facedown items | 3 items behind the slots, playable any time (13) |
| Field card | a card from the Field shop; not an item apart from paying star points (18). "Field item" was the old word |
| Lane | a place on the table: three lanes, a team on each side (04). Not a turn unit since pair turns |
| Multi turn ⏳ | (XCD), (Y/XCD), (XGCD), (Y/XGCD) with X of 2 or more (02) |
| Pair turn | a slot from one team, then a slot from the opposing team (08) |
| Prime creature | a creature that fills a slot: every creature that isn't a minion; a card with several forms or made of multiple creatures is one (16) |
| Primary / secondary field | front and back of a slot (16) |
| Proc / instance | trigger / one application of damage or healing (never defined) |
| Reaction | a play that resolves before what it answers (12) |
| Swap out / in | creature back to hand, another in; 2 points as its turn is about to start; cannot change turn order (08) |
| Tier (debuff) | its star level (inference) |
| Time based 🕐 | every 🕐 tag: (XCD), (Y/XCD), (XGCD), (Y/XGCD), (BOT), (ENDT), (BOGT), (ENDGT) (02) |
| Uses (XU) | X use(s) per full charge; no (XU) = 1 (02, 11, 23) |
| Recharge, usages | the cards' words for charge and remaining uses (Michael, Energy Ritual, The Bank) |

## 14. How to answer a rules question

1. **Find the chapter.** 00 Contents, 01 Getting Started, 02 Abbreviations, 03 Setting Up, 04 Table, 05 Shops, 06 Reading Cards, 07 Stats, 08 The
   Global Turn, 09 Taking Your Turn, 10 Attacking, 11 Abilities, 12 Reactions, 13 Items, 14 Item Types, 15 Minions, 16 The Field, 17 Death, 18
   Field Cards, 19 Status Effects, 20 Creature Types, 21 Damage Types, 22 Negation, 23 Timing Tags. Sleepy Kirby marks "more technical rules" (00).
2. **Quote and cite** as (NN) or (ch NN), in the rulebook's words. If chapters disagree or the book is silent, say so and point to Known gaps.
   Status effect and type effects are card text — say so.
3. **Where the text lives.** `site/rulebook/NN-slug.md` in the card repo, written on the website; the website's Rulebook tab shows them, Card sync
   writes the bot's `db/rulebook.json` and prints `TTS/Rulebook/Rulebook.pdf`. Beside this file: `reference/glossary.md` (every defined term and tag
   by chapter) and `reference/data-facts.md` (the card-data counts behind this file).
4. **The bot.** `/query_rulebook search_text [limit]`: case-insensitive substring search over the chapters, icons as emojis; `search_text` is required
   by the slash command (main.py); a whitespace-only search only gets "Give a word or phrase to search the rulebook for.". `/export_rulebook`: all chapters as a
   text file. `/query_ability` searches card text (`type==debuff & stars==3`, `(1gcd)`; DSL in impossibility-cards-reference §7) in the bot's card
   data.
5. **Text players never see.** Only ch 14's `[if you find this text dont delete it]{.secret}`, which is also in `db/rulebook.json` and so in
   `/query_rulebook`. The Photoshop pages' hidden drafts stayed behind in `MDW/Old Rules/Rulebook/`.
6. **Vocabulary.** "creatures" not units, "minions" not summons, "items"/"item types" not spells or subtypes (`subtype` is only the bot's field name),
   "field cards" not field items, "pair turns" not lane turns, and likewise status effects, buffs, debuffs, stars, types, abilities, timing tags,
   global turn, starting hand.

## Rule changes from Discord

Messages marked 📖 in pending-updates (channel 1087579710428565526) are implemented in `site/rulebook` as one commit, then forwarded to
finished-updates (1091208792575844444) and deleted from pending.

## Known gaps and ambiguities

**Resolved 2026-10-30 (owner rulings, in the chapters):** NORMAL damage is DMG − DEF (the ½DEF leftover fixed) and damage types combine (21); OVERKILL
defined (21); attacks from abilities can't trigger counter items (10); revived creatures reset HP, stats, abilities, charge and counters (17); a
creature replacing one defeated after acting can't act but may use an ability (17); SPD can't go below 1, ATK and DEF not below 0, MAX HP not below 1
bar (07); two sources negating each other at once are both negated (22); conflicting debuffs: the higher star level wins (19); a creature finishes its
turn if turn order changes mid-turn, and NO STALLING (08); field cards can't be sold or targeted by items (18); a reaction chain resolves before a new
one begins (12); typos fixed: Reactions "6 stacks", Setting Up "Two 1-star items ★ ★".

**Resolved 2026-11-01 (Discord pending updates, TTSCardMaker `c3eb51a6`; `2fecc66d` for the TTS save):** pair turns replace lane turns, a lane is still a
place (08, 04); swapping out cannot change turn order (08); dying vs being defeated (17); "destroyed" = true damage equal to current HP (17); prime
creature (16) and base stats (07) defined; the cooldown tags and charge, so how a cooldown counts, (XCD) implying (OYT), (XGCD) implying (AAT), when uses
refill, the -1 of (-1/1GCD), and which tags are time based and multi turn (02, 11, 23); multi-hit attacks are one attack (10); active items sit in a slot
on the primary field (14, 16); damage from abilities and item effects cannot trigger counter damage as retaliation (10, 21); damage dealt when something
dies IS counter damage (21; owner ruling, the poll's result was lost); 0 ATK cannot declare an attack (10); minion sneak attacks 25% without +1, which also
groups the "+1" outside the 25% (10); Max HP decrease rules (07); items not on creatures in hand or graveyard (13); secondary field untargetable by
opponents (16); 5★ minions not targetable by minion effects, bare "summon" (15); no inheriting equip abilities (14); targeted for negation or field card
destruction: no (AAT) reaction with that card's effects (12, 22); field cards are not items apart from paying star points (18); actions only (OYT) (09);
Toy Chica + Chef Saltbaker (11); "field item" is now "field card" in the chapters, the ban list and the TTS save.

**Owner rulings 2026-11-01, later (now in 14, 15, 22):** minions are tied to their parent creature (the creature they are on), not their summoner, and a summoner can occasionally summon onto another creature; a defeated parent's minions die unless persistent. A creature's innate abilities and its equip's are separate (Sword of Aeons grants lifesteal; the creature has none innately), and negating a creature does not negate its equipped items.

Tensions left in the text:
- Cards still counting lanes as turns (card text is not changed by rulebook work): Delicate Flower "3 lane turns", The Last Cabbage Merchant "each time a
  lane turn ends", Time Riddle "until 2 lanes have gone", Asteroid Field "Every time a lane ends its turn", Forget Me Now "after the current lane ends", NME
  Salesman "3 lanes' turns must end", Lost Kin "always go first in its lane". Other cards' "lane" is the place (Dimension D, Inner Eye, Mom's Knife).
- (02) lists (XU) under PER GAME though it is per full charge; (23) calls the group "Per game & triggers".
- Ch 12's pictures show old card text (§3).

Undefined: "tier"; "proc", "instance", "on-hit", "reactively", "innate actions/type"; "cleanse" (card word); "N.M.E."; the "Don't be stupid" rule.

Silent: how a hand is built from the full or developer/baseline decks (new players draw from the simple-version decks, 04); hand limits; summoning from hand
except to fill an empty slot; whether creatures in hand must be defeated to win; starting BRICKS; how health lost and star level count for a slot with several
creatures on the primary field (08); whether combo damage applies to sneak attacks or ability DMG; unspent star points (implied lost); "one ACTION" (09) vs "2
innate actions" (11) and Inspired (+1 Action); "5+ star creatures" vs decks topping out at 5★; COUNTER as item type (14) vs damage type (21); the
`Auxiliary/Creatures/` cards; status effects are defined only on cards (the book names ~22 of 64).

## When NOT to use this skill

| Task | Use instead |
|---|---|
| How card DATA is encoded: folders, PSD layers, stats extraction, type injection, card data schemas, the /query DSL | impossibility-cards-reference |
| The icon font, [type] tokens, timing-tag badges, the card website or deck builder, anything in ~/Desktop/TTSCardMaker | ttscardmaker-companion |
| Running the bot, /update_stats (the card data, which also carries the chapters as db/rulebook.json) | autocroissant-run-and-operate |
| A card whose ability text or type parsed wrong | autocroissant-psd-extraction-campaign, autocroissant-diagnostics-and-tooling |
| Help text, describe strings, skill-writing conventions | autocroissant-docs-and-style |
| An AI that answers rules questions (RAG, /ask) as a future feature | autocroissant-research-frontier |

## Provenance and maintenance

Source: `site/rulebook/` in `~/Desktop/TTSCardMaker`, plus the card data on its `data` branch. To re-verify a rule, read its chapter and compare; to re-verify a
count, run the row below on the current card data (`git -C ~/Desktop/TTSCardMaker show origin/data:db/stats.json > /tmp/s.json`, or the site's built `cards.json`).

| Fact | Re-verify with |
|---|---|
| 24 chapters | `ls ~/Desktop/TTSCardMaker/site/rulebook/*.md \| wc -l` |
| Old tags only in ch 23 | `grep -nE '\b(E[0-9X]+G?T\|FG?TE[0-9X]+G?T\|[0-9X]+PG?T)\b' ~/Desktop/TTSCardMaker/site/rulebook/*.md` |
| The old words are gone | `grep -rni "field item\|lane turn" ~/Desktop/TTSCardMaker/site/rulebook/*.md ~/Desktop/TTSCardMaker/site/Constructed\ Ban\ List.txt` (prints nothing) |
| The bot's copy of the chapters | `git -C ~/Desktop/TTSCardMaker show origin/data:db/rulebook.json \| python3 -I -c "import json,sys;print(len(json.load(sys.stdin)['chapters']))"` (24) |
| Tag counts (§3, 2026-10-22) | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/tags.py /tmp/s.json` (matches (1/2CD) and (-1/1GCD); prints a total line) |
| Card evidence (§3 "What the cards show") | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/lines.py /tmp/s.json`: lines per family with (OYT)/(AAT), the tag before each (nU), line starts, cooldown tags after other text, (a/bCD) cards |
| Status effect texts, mentions | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/mentions.py /tmp/s.json`; or `/query_ability type==debuff` |
| Card counts by kind/stars/type (cards.json: 773 rows, 298 creature rows, 46/103/93/56; 2026-10-22) | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/counts.py /tmp/s.json`, `.../scripts/agg.py /tmp/s.json`; `scripts/rows.py` reads both shapes for all four |
| stats.pkl counts used above (297 = 45/103/93/56, 75 untyped, per-type distinct creatures, 182 item PSDs) | `python3 -I -c "import json,collections as c;s=json.load(open('/tmp/s.json'))['cards'];cr=[v for v in s.values() if v['card_type']=='creature'];print(len(cr),sorted(c.Counter(v['stars'] for v in cr).items()),sum(1 for v in cr if not v['types']),c.Counter(t for v in cr for t in set(v['types'])),len({v['path'] for v in s.values() if v['card_type']=='item'}))"` (`set()` per creature, not icon occurrences) |
| Cards that say "lane" | `python3 -I -c "import json;[print(k,'\|',v['ability']) for k,v in json.load(open('/tmp/s.json'))['cards'].items() if 'lane' in (v.get('ability') or '').lower()]"` |
| Ban list (53) | `grep -c ' - ' ~/Desktop/TTSCardMaker/site/Constructed\ Ban\ List.txt` |
| Website hand rule | `grep -n "HAND\|FIELD_MAX_STARS" ~/Desktop/TTSCardMaker/site/assets/deck.js` |
| Tag tooltips | ch 23; `grep -n "GAME_TAGS\|tag_meanings" ~/Desktop/TTSCardMaker/site/build/build_site.py` |
| Slash commands | `grep -n 'name="query_rulebook"\|name="export_rulebook"' main.py`; `grep -n rulebook_dff commands/query_card.py` |
| Rounding table (07: 20 rows, 6 change columns) | `python3 -I -c "import re,os,math;R=[[float(x) for x in re.findall(r'\d+(?:\.\d+)?',l)] for l in open(os.path.expanduser('~/Desktop/TTSCardMaker/site/rulebook/07-stats.md')) if re.match(r'\| \d',l)];print(len(R),sum(math.floor(2*r[0]*(1+k)+0.5)/2!=v for r in R for k,v in zip((-.25,-.5,-.75,.25,.5,.75),r[1:])),'mismatches')"` (prints `20 0 mismatches`) |
| Old rules | `ls ~/Desktop/TTSCardMaker/MDW/Old\ Rules/Rulebook/*.psd \| wc -l` (48; `MDW/Old Rules/README.md`) |
| TTS save facts (data-facts §7) | copy `TTS/audit.py` from TTSCardMaker `843a4a99` (removed 2026-10-25: `git show 843a4a99:TTS/audit.py`) to a scratch folder, set `OUT` to that folder, `REPO` to the clone and `RULES` to `OUT/decks.json`, run it with `python3 -I` (it writes only to `OUT`); `TTS/AUDIT.md` (the 2026-10-20 run) is at `843a4a99` too |

When a chapter changes: update this file (the section stating the rule, its citation, Known gaps). When cards change: after the Card sync data commit, rerun
tags.py, lines.py and mentions.py on its db/stats.json (or on the site's cards.json); refresh §3 and data-facts §4. If the ban list moves: update §2.

2026-10-22: counts re-measured after the cooldown update (TTSCardMaker `73904dbd`). 2026-10-25: scripts read db/stats.json. 2026-11-01: restated from the chapters; the
verbatim Photoshop extract and its build scripts removed (a copy would drift; they remain in git history).
