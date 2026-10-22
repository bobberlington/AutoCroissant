NOTES = {}
N = NOTES

N["0000 Cover"] = """
Diagram: front cover art. Title "Impossibility Simulator" over a white double-infinity logo and a collage of creatures. No rules.
"""
N["0000 Cover Book"] = """
Diagram: the 1920x1920 texture for the Tabletop Simulator rulebook object: the cover (title, logo, collage, "Rulebook" with a Yoshi sprite, a Kirby ball) and a SpongeBob-flower spine strip. No rules.
"""
N["00 Table of Contents"] = """
Icons: sleepy Kirby (the 8-space gap before "If you see sleepy Kirby"). It is a picture, not a type icon.
Notes: sleepy Kirby sits beside, for example, 10 Attacking2 REDIRECTIONS, the 11 Abilities6, 11 Abilities7 and 12 Reactions5 titles, 13 Items MISC, 17 Death SPECIFICS and 22 Negation CREATURES; those boxes are the technical rules. Other Kirby poses (open-mouthed, waving, starry) are just decoration or a speaker for tips.
"""
N["01 Getting Started"] = """
Diagram: three example creature cards (Computer Virus, Grey Prince Zote and OMORI) beside the CREATURES box. A Dr. Eggman-style face speaks the "Don't be stupid" line. The final "1" is the page number.
Notes: the pictured cards show old text. Computer Virus: "attacks inflict Poison for 5 turns and deal 2 piercing (CD) to itself", where (CD) is the old counter damage (today "2 piercing CDMG"), not the cooldown tag. OMORI: "2+ counters: +1 SPD, first turn and (EOT) attacks inflict 2 turns of Afraid." (today "(1/2CD) Attacks inflict 2T Afraid.").
"""
N["02 Abbreviations"] = """
Icons: 🕐 (clock, "time based") and ⏳ (hourglass, "multi turn") are real characters in the text and on the page. Kirby faces mark the notes "These imply (OYT).", "These imply (AAT)." and "All of these must happen automatically.".
Diagram: the page is a grid of boxes, which the extracted text runs together:
- GENERAL: SP, DMG, HP, ATK, DEF, SPD.
- TIME FRAME: (OYT), (AAT). Beside it a second GENERAL box: PF, CDMG.
- 🕐ON YOUR TURN: (XPT), (EXT)⏳, (FTEXT)⏳. Note: "These imply (OYT)."
- 🕐AT ANY TIME: (XPGT), (EXGT)⏳. Note: "These imply (AAT)."
- 🕐FORCED: (BOT), (ENDT), (BOGT), (ENDGT). Note: "All of these must happen automatically."
- FORCED (no clock): (OFS), (OS), (OD), (WOF).
- PER GAME: (XPG).
The X in each tag is bold italic: it stands for the number printed on the card, e.g. (1PT), (2PGT), (E2T), (FTE2T), (E2GT), (3PG).
Since 2026-10-21 the cards print cooldown and uses tags instead, such as (1CD), (1GCD), (1/2CD) and (2U) (SKILL.md §3); this page still lists only the old ones.
Notes: in the extracted text the second FORCED heading and PER GAME are run together before (OFS). On the page they are two boxes: FORCED holds OFS/OS/OD/WOF, PER GAME holds XPG.
"""
N["03 Setting Up"] = """
Diagram: the STARTING HAND list in two columns (creatures left, items right), each with its star count drawn. Below that is a screenshot of a hand (Items and Creatures). A Sonic ring box says "Each team starts with 2 points, and 7 star points." The table screenshot shows both teams' 3 slots with creatures face down and stat counters beside them, and the 3 facedown item holders, one behind each slot. A Dr. Eggman face says "Evil Quote!".
Notes: the starting hand is 7 creatures (1 five-star, 2 four-star, 3 three-star, 1 two-star) and 10 items (2 four-star, 3 three-star, 3 two-star, 2 one-star), which matches "You start with only 10 of them" on 13 Items. The extracted text shows two stars for each one-star item ("⭐⭐ ⭐⭐"); the page draws one star each. "Each team starts with:" appears twice because the Sonic box is a separate layer.
"""
N["04 Table"] = """
Diagram: a screenshot of the Tabletop Simulator table. Upper right: (1) the four creature decks (5★, 4★, 3★, 2★ from left), (2) the item decks, (3) the field card decks, (4) the N.M.E. deck. Upper left: (5) the developer creature decks and (6) the baseline creatures. The digits 5 1 4 2 6 3 at the top of the text are these callouts.
Notes: the page prints items 1-4 as "1. The CREATURE DECKS ..." and so on. The extracted text drops the numbers and keeps only a leading space.
"""
N["04 Table2"] = """
Diagram: the counter row and the shops. In the middle is (1) the global turn counter. On each side of it are each team's (2) star point counter (yellow), (3) BRICKS counter (red) and (4) points counter (green). Below are (5) the item shop (8 slots priced 4★ 4★ 3★ 3★ 2★ 2★ 1★ 1★), (6) the field card shop (8 slots in the same star pattern, pictured as trees) and (7) the N.M.E. shop (4 slots behind a blue curtain). The digits 4 3 2 1 5 6 7 are these callouts. Each list item is numbered 1-7 on the page; the text drops numbers 1-4.
"""
N["04 Table3"] = """
Diagram: the playing field seen from above. There are three horizontal lanes. Each team's creatures sit on its own side (left and right), each slot with four coloured counters and red marker squares. Each team's facedown item holder sits at the outer edge of each lane. Inset: (4) status effect markers on top of the slot, (5) the two graveyards (tombstones), (6) the discard pile (a pixel bin) and (7) the two field card slots (tree banners).
"""
N["04 Table4"] = """
Diagram: (1) the auxiliary item deck (Spaghetti on top), (2) the auxiliary minion deck, (3) the minion card holder with four counters, (4) a green d6, the "Redirection cube", (5) a coin, the "Order Determinator", (6) the +0.5 chip, (7) a black counter, (8) the HP marker (green, heart), (9) the DEF marker (blue, shield), (10) the ATK marker (red, sword) and (11) the SPD marker (yellow, lightning). Last, (12) the active item holder: a radio-tower card with two counters (green HP and blue DEF), because active items have HP and DEF.
Notes: counter colours used everywhere in the book are HP green, DEF blue, ATK red, SPD yellow. One marker is one unit of the real stat value (1 DEF marker = 0.5 DEF, 1 HP marker = 1 HP), not one bar.
"""
N["04 Table5"] = """
Diagram: the "simple version" deck block (1, upper right, creature and item decks) and the "extra stuff" (upper left): (2) the copy of the full deck, (3) a bag of sorted item decks and (4) a bag of creature type tokens. Item 1 is numbered on the page; the text drops the "1.".
"""
N["05 Shops"] = """
Diagram: the three shop layouts. ITEM SHOP has 8 item slots labelled 4★ 4★ 3★ 3★ 2★ 2★ 1★ 1★. FIELD SHOP has 8 tree slots with 4, 4, 3, 3, 2, 2, 1 and 1 stars. N.M.E SHOP has 4 TV-host slots behind a curtain. A Morshu picture says the "Sorry Link" quote; the N.M.E. host icon says "My items are situational. Buy them wisely!".
Notes: the numbering repeats ("2)" twice in each list) on the page too.
"""
N["06 Reading Cards"] = """
Diagram: a creature card with callouts. (1) The type icon at the top left. (2) The star level at the top right. (3) Four counters beside the card, from top to bottom green HEALTH 14, blue DEFENSE 1, red ATTACK 9, yellow SPEED 5, each with a red marker square to its right. (4) The four stat bars on the card. (5) The ability text box ("This creature takes 3 counter damage each time it attacks. When this creature has 50% or less HP, its attacks strike all creatures on the PF"). On the page items 1-5 are numbered, and 4 has sub-items "A. The health value is double of what is shown." and "B. The defense value is half of what is shown.".
Notes: the stat bars show the MAX value (bars x2 for HP, x0.5 for DEF, x1 for ATK and SPD). The counters show the CURRENT value.
"""
N["06 Reading Cards2"] = """
Diagram: the item card Sacred Heart. (1) The item type icon at the top left (the equip sword icon). (2) The star level at the top right. (3) The effect box: "+50% ATK / Attacks can hit any field and ignore intercepts, but will not receive the ATK boost if this is done."
"""
N["07 Stats1"] = """
Diagram: one creature's stat bars (10 cells per stat). HEALTH has 8 lit cells (counter 16), DEFENSE 4 (counter 2), ATTACK 4 (counter 4), SPEED 6 (counter 6). The stray numbers 16 2 4 6 in the text are these counters. The healing rows read:
- 16 + "Heal 4 HP" --> 16 (already at MAX HP).
- 14 + "Heal 4 HP" --> 16 (capped at MAX).
- 16 + "+3.5 Max HP" --> 19 plus a +0.5 chip (19.5). MAX HP rises, shown by three heart markers and one sideways heart marker ("+3.5").
- 16 + "Heal 3 HP (Can go above max)" --> 19. Note: "If this goes below 16, it can only heal up to 16 again with regular healing, as MAX HP was not increased."
A pixel Alphys-style speaker gives the "2 different healing effects" note.
"""
N["07 Stats2"] = """
Diagram: the same stat bars (HP 16, DEF 2, ATK 4, SPD 6). The rows read 16 + "-2 MAX HP" --> 14, and 12 + "-2 MAX HP" --> 12. Captions: "1 Bar of HP is 2 MAX HP" and "Decreases to MAX HP are indicated with these markers" (a heart with a minus).
"""
N["07 Stats3"] = """
Diagram: the stat bars with DEFENSE highlighted (4 bars, counter 2). The rows read 2 + "-1 DEF" --> 1, and 2 + "-1 DEF BAR" --> 1 plus a +0.5 chip (that is 1.5). The DEF-down marker is a shield with a minus. Link and a fox with a shield decorate the page.
Notes: the text says "-1 DEF BAR --> 1", but the picture shows the counter at 1 with the +0.5 chip, i.e. 1.5. One DEF bar is 0.5 DEF.
"""
N["07 Stats4"] = """
Diagram: ATTACK row 4 + "1.5 ATK" --> 5 plus a +0.5 chip (5.5). The ATK-up marker is a sword with a red plus. SPEED row 6 + "1.5 SPD" --> 7 plus a +0.5 chip (7.5). The SPD-up marker is a yellow lightning bolt with a plus. A flexing Kirby and Sonic decorate the page.
"""
N["07 Stats5"] = """
Diagram: a lookup table. The text has its seven columns one after another; on the page they read across. Each cell is the base value with the change applied, rounded to the nearest 0.5 with halves rounded up:

| base (±0%) | -25% | -50% | -75% | +25% | +50% | +75% |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.5 | 0 | 0.5 | 1 | 1 |
| 1 | 1 | 0.5 | 0.5 | 1.5 | 1.5 | 2 |
| 1.5 | 1 | 1 | 0.5 | 2 | 2.5 | 2.5 |
| 2 | 1.5 | 1 | 0.5 | 2.5 | 3 | 3.5 |
| 2.5 | 2 | 1.5 | 0.5 | 3 | 4 | 4.5 |
| 3 | 2.5 | 1.5 | 1 | 4 | 4.5 | 5.5 |
| 3.5 | 2.5 | 2 | 1 | 4.5 | 5.5 | 6 |
| 4 | 3 | 2 | 1 | 5 | 6 | 7 |
| 4.5 | 3.5 | 2.5 | 1 | 5.5 | 7 | 8 |
| 5 | 4 | 2.5 | 1.5 | 6.5 | 7.5 | 9 |
| 5.5 | 4 | 3 | 1.5 | 7 | 8.5 | 9.5 |
| 6 | 4.5 | 3 | 1.5 | 7.5 | 9 | 10.5 |
| 6.5 | 5 | 3.5 | 1.5 | 8 | 10 | 11.5 |
| 7 | 5.5 | 3.5 | 2 | 9 | 10.5 | 12.5 |
| 7.5 | 5.5 | 4 | 2 | 9.5 | 11.5 | 13 |
| 8 | 6 | 4 | 2 | 10 | 12 | 14 |
| 8.5 | 6.5 | 4.5 | 2 | 10.5 | 13 | 15 |
| 9 | 7 | 4.5 | 2.5 | 11.5 | 13.5 | 16 |
| 9.5 | 7 | 5 | 2.5 | 12 | 14.5 | 16.5 |
| 10 | 7.5 | 5 | 2.5 | 12.5 | 15 | 17.5 |

Checked: all 140 cells equal floor(2*base*(1+change) + 0.5) / 2, i.e. the nearest half with exact quarters rounded up. The final "7.5" in the text is the page number.
"""
N["08 Global Turn"] = """
Icons: the star glyph (★ = star level) was lost from the extracted example cards; it is restored above as [icon: ★].
Diagram: the six example creatures in three lanes. A lane is a column here: the top creature is one team, the bottom the other.

| lane | top creature | bottom creature |
|---|---|---|
| L1 | A: HP 8/10, SPD 5, 3★ | B: HP 8/9, SPD 5, 5★ |
| L2 | C: HP 8/8, SPD 6, 4★ | D: HP 6/6, SPD 4, 4★ |
| L3 | E: HP 7/7, SPD 3, 5★ | F: HP 8/8, SPD 3, 3★ |

The resulting order is "C > D > B > A > E > F".
Notes: Lane 1 "missing 3 HP" is A's 2 plus B's 1: health lost is summed over the lane. The lane's speed is the fastest creature of each side added up: 5+5 = 10, 6+4 = 10, 3+3 = 6.
"""
N["09 TakingYourTurn"] = """
Diagram: decoration only (a fox and Steve with a pickaxe). The section order on the page is ATTACK, MINING, DEFENSE MODE, then TURN PHASES. In the extracted text the DEFENSE MODE heading comes after its paragraph.
Notes: rules.json repeats the ATTACK paragraph and the DEFENSE MODE paragraph. The second copy of each is a HIDDEN layer in the PSD (groups "Group 2/Attack" and "Defense Mode/Defense Mode", checked with psd-tools) and is left out above.
"""
N["10 Attacking"] = """
Diagram: two worked examples set out as cards. A (HP 10/10, DEF 2, ATK 7, SPD 4) attacks B (HP 10/10, DEF 1, ATK 6, SPD 4): 7 - 1 = 6 damage, B at 4 HP. C (HP 10/10, DEF 2, ATK 7, SPD 6) attacks D (HP 10/10, DEF 1, ATK 6, SPD 3): 6 x 1.5 = 9, D at 1 HP. Both pairs are in lane L1. A Kirby-like ninja sprite decorates SNEAK ATTACKS.
Notes: these examples subtract the full DEF (B and D have DEF 1). Compare 21 Damage Types "NORMAL: DMG - (1/2*DEF)" (see Notes there).
"""
N["10 Attacking2"] = """
Diagram: decoration only (sleepy Kirby by REDIRECTIONS, an infliction icon by INFLICTIONS).
"""
N["11 Abilities"] = """
Diagram: the example ability box is an IMAGE; its text is not in rules.json. Transcribed from the PNG:
1. "(OYT) Spend 3 points to cure a tier 2 debuff."
2. "(AAT) Spend a point to heal this creature by 1 HP."
The "1" and "2" in the text are these callouts.
Notes: rules.json ends with a PASSIVES heading and paragraph that are HIDDEN layers in the PSD ("Group 2/PASSIVES", "Group 2/Note copy 2"). They are left out above. The visible version of that rule is on 11 Abilities7. "Tier 2 debuff": debuffs carry star levels (the debuff cards sit in Auxiliary/Status Effects/Debuffs/<N> Stars), and "tier" means that star level. The rulebook never says so.
"""
N["11 Abilities2"] = """
Diagram: the example ability box shows "(1PT) Summon a Mini Bee.", "(2PGT) Heal a creature by 2 HP." and "(AAT)(3PG) Spend 7 points to deal 10 piercing damage.", with callouts 1-3. A Dr. Eggman face quotes "(1PG)(AAT) Kill an enemy creature" as a joke.
"""
N["11 Abilities3"] = """
Diagram: the example box shows "(BOGT) +1 SPD", "(BOT) +1 ATK" and "(ENDT) +5 DEF until (ENDGT)". Dr. Eggman says "Don't give this ability to sonic.".
"""
N["11 Abilities4"] = """
Diagram: the example box shows "(E2T) Summon a Mega Bee.", "(FTE2T) Summon Brombus." and "(E2GT) Deal 3 piercing damage.". Two tips: Sonic's ring ("If you forget to use your time based ability ...") and an Alphys-style face ("Abilties do not count time passing while in the hand.").
"""
N["11 Abilities5"] = """
Diagram: the example box shows "(OFS) Summon a Koopa.", "(BOGT)/(OS) Declare which slot this creature will be attacking and it will attack all targets in that lane." and "(OD) Deal 5 damage to everything on the PF.". Decoration: Mettaton (box form) and another character with the line "Summon me any time!".
"""
N["11 Abilities6"] = """
Notes: the first line of rules.json, "ABILITIES 7", is a header layer placed off the canvas (y = -162), so it does not show. The page title is ABILITIES 6. Sleepy Kirby sits in the title, so the whole page is "technical rules".
"""
N["11 Abilities7"] = """
Diagram: the PASSIVES example box shows "(WOF) Friendly creatures gain +1 DEF.", "This creature attacks inflict Burning.", "This creature is immune to Poison." and "This creature has 2 innate actions.". The MISC healing-item sentence is a side note by an Alphys-style face. Sleepy Kirby sits in the title.
"""
N["12 Reactions"] = """
Diagram: (1) the Mighty Pebble item ("Deal 5 damage") and (2) Toy Bonnie (2★, machine type). Toy Bonnie's stats go from "Stats before" HP 8, DEF 0, ATK 5, SPD 5 to "Stats after" HP 6, DEF 3, ATK 5, SPD 5. Six Hardened markers sit above it. Toy Bonnie's pictured card reads "(OPGT) Inflict up to 6 stacks of Softened for 1 turn ..." (an O, not a 1; a pre-rulebook tag in neither system); the card today reads (1GCD), as it did before the cooldown update.
Notes: the trailing "of 5." in rules.json overflows its text box and is not drawn.
"""
N["12 Reactions2"] = """
Diagram: (1) Mighty Pebble, (2) Toy Bonnie, (3) Metronome (a negation item) and (4) Stopwatch. Toy Bonnie goes from HP 8/DEF 0/ATK 5/SPD 5 to HP 6/DEF 3/ATK 5/SPD 5. Toy Bonnie's pictured card reads "(OPGT) Inflict up to 6 stacks of Softened for 1 turn ..." (an O, not a 1; a pre-rulebook tag in neither system); the card today reads (1GCD), as it did before the cooldown update.
Notes: rules.json also contains a second block, a HIDDEN layer ("INTRODUCTION/Item shop copy"), shown above for reference. It is an older draft of 12 Reactions3 that adds "Attacks also resolve forward, meaning the first damage item played resolves first", and so gets the order 4>2>1>3 (pebble 0.5, then bomb 2.5). The visible 12 Reactions3 gets 4>2>3>1 instead: the later damage resolves first, i.e. damage also resolves in reverse. The visible page wins.
"""
N["12 Reactions3"] = """
Diagram: (1) Mighty Pebble, (2) Toy Bonnie, (3) Bomb ("Deal 7 damage") and (4) Metal Shield, whose card reads "+1.5 DEF -.5 SPD". Toy Bonnie goes from HP 8, DEF 0, ATK 5, SPD 5 to HP 5, DEF 4 + 0.5 chip (4.5), ATK 5, SPD 4 + 0.5 chip (4.5). The SPD drop comes from Metal Shield's card; the text does not mention it. Toy Bonnie's pictured card reads "(OPGT) Inflict up to 6 stacks of Softened for 1 turn ..." (an O, not a 1; a pre-rulebook tag in neither system); the card today reads (1GCD), as it did before the cooldown update.
Notes: the text says "5 stacks of hardened, giving it 3 defense". Hardened is +0.5 DEF per stack (Auxiliary buff card), so 3 DEF needs 6 stacks, as on 12 Reactions/12 Reactions2. The 4.5 total = 3 + 1.5 also needs 6 stacks. The page number is printed "123" (meaning 12.3).
"""
N["12 Reactions4"] = """
Diagram: (1) Whispy Woods (counters HP 21, DEF 2, ATK 4, SPD 3; ability upside-down "(OPT) You can spend 2 points to heal all friendly creatures by 3 HP"), (2) Toy Bonnie (HP 2, DEF 0, ATK 5, SPD 5), (3) Bomb, (4) Pep Drink ("Heal a creature by 2.5 HP and give it 3 turns of 1 Peppy, or heal [icon] a by 4 HP") and (5) Amy Rose's Hammer. Softened (-0.5 DEF, counter 2) and Hardened (+0.5 DEF) markers are shown. Three "after" columns of counters show HP -2/DEF 3, then HP 2/DEF 3, then HP 0/DEF 2: the state if the chain ended after step 3, after step 4, and the final result. Toy Bonnie's card ability reads "(OPGT) Inflict up to 6 stacks of Softened for 1 turn distributed across any creatures, and this creature inflicts an equal number of stacks of Hardened on itself for 1 turn." (small print; clearly (OPGT) on 12 Reactions, the same card). (OPGT) and (OPT) are pre-rulebook tags in neither system. On the cards today Toy Bonnie reads (1GCD) (as before the cooldown update) and Whispy Woods "(1CD)(2 points) Heal all friendly creatures by 3 HP.". The page number is printed "124" (12.4).
"""
N["12 Reactions5"] = """
Notes: sleepy Kirby sits in the title, so this is "technical rules". The page number is printed "125" (12.5).
"""
N["13 Items"] = """
Diagram: a row of item art at the top. Mr. Krabs (money) marks SELLING ITEMS. A Dr. Eggman face says the "Chaos Emerald" quote. Sleepy Kirby marks MISC.
"""
N["14 Item Types"] = """
Icons: each heading carries its item type icon: HEALING (Food, a tomato), ATTACK (a bomb), NEGATION (a metronome), SUPPORT (a blue ring), COUNTER (spiky grass), INFLICTION (Mr. Krabs on fire). On the page the HEALING sentence ends with the machine icon (gears): "May not be used on [machine]." This matches 20 Creature Types (MACHINE: "Cannot have food items be used on them").
Notes: rules.json says "May not be used on [support]." That is a parser mis-injection: the machine icon is above the card midline, so it went to the page's types, and the gap got the next icon in reading order (support). The trailing "[infliction]" in rules.json is a leftover icon appended after the page number. The sentence "if you find this text dont delete it" overflows its text box and is not drawn.
"""
N["14 Item Types2"] = """
Icons: EQUIP (a sword with a red ribbon), FRAGILE EQUIP (the same sword with an alarm clock), ACTIVE (a radio tower) and DEPLOY (a Peashooter). rules.json appended the leftover heading icons as " [fragile equip] [active] [deploy]" after the page number; they are left out above.
"""
N["15 Minions"] = """
Icons: under TYPES, the offensive minion icon (a brown bunny-eared head, Types/Offensive_Minion.png) starts the offensive line, and the protective minion icon (a green Koopa shell, Types/Defensive_Minion.png) starts the protective line. rules.json lost both gaps ("TYPES:Offensive", "):Protective") and appended "[defensive_minion]" after the page number. Restored above.
Notes: the rulebook says "protective minion" and the icon file is named Defensive_Minion. They are the same thing.
"""
N["16 The Field"] = """
Icons: "designated by this symbol:" is followed by the protective minion icon (the green shell). In rules.json the icon is appended after the page number instead.
Diagram: a slot with (1) the parent creature (an "Agent" card with counters) at the front-bottom, and (2) and (3) two protective minions above it. The page calls (1) the parent creature in the secondary field. A Dr. Eggman face says the joke line.
"""
N["16 The Field2"] = """
Diagram: one slot in two coloured boxes. The green box (1, PRIMARY FIELD) holds two protective minions with green shells, (5) Yarn and (6) Grey Prince Zote. The red box (2, SECONDARY FIELD) holds (4) an offensive minion, Gulka, and (3) the parent creature Holo Defense MK, with counters 0/2 +0.5/5/5. The digits 5 6 1 4 3 2 are these callouts.
"""
N["17 Death"] = """
Diagram: a large tombstone, a knocked-out character and a ghost by REVIVES. Sleepy Kirby marks SPECIFICS. Sonic's ring speaks the quote.
"""
N["18 Field Cards"] = """
Icons: the field type icon (a Terraria tree) decorates the page; rules.json lists types: field. A Dryad sprite speaks the quote.
Notes: the COUNTER-PLAY bullets are repeated word for word under FIELD CARDS on 22 Negation.
"""
N["19 Status Effects"] = """
Diagram: Panda (panda type, 2★) with the Softened card (-0.5 DEF, counter 2) and the Smouldering card (counter 2) above it. Its counters are HP 16, DEF 0 + 0.5 chip, ATK 7, SPD 6. The Mr. Krabs pictures are the status effect cards: thriving Krabs for buffs, suffering Krabs for debuffs.
"""
N["19 Status Effects2"] = """
Icons: the buff (single orange up arrow) and debuff (single blue down arrow) icons follow the Standard sentence. The buffstack (double orange arrows) and debuffstack (double blue arrows) icons follow the Stacking sentence. The debuffupgrade icon (a down arrow with a second arrow) sits by UPGRADABLE. rules.json put "[debuff]" after the page number; restored above.
Diagram: examples. Peppy (buffstack, "+1 SPD") beside "This buff can be applied many times ...". Withering (debuff, 4★, "(BOT) 2 DMG -0.5 DEF -1 ATK -1 SPD.") beside "This debuff cannot be applied more than once.". The upgrade chains drawn as cards:
- Smouldering (1★, "2 DMG, can't affect [fire]") -> Burning (1★, "3 DMG, can't affect [fire]") -> Blazing (2★, "4 DMG, can't affect [fire]").
- Chilled (2★, "-25% SPD can't affect [ice]") -> Frostbite (2★, "2 DMG. -25% SPD, can't affect [ice]") -> Frozen (3★, "Can't affect [ice]. Lose all actions, cannot switch out.").
- Sickness (1★, "-0.5 DEF -1 ATK -1 SPD can't affect [undead] [machine]") -> Plague (2★, "-1 DEF -1 ATK -2 SPD Can't affect [machine] [undead]").
(Card texts checked against cards.json.) The card file is spelled "Smoldering"; this page spells it "Smouldering".
"""
N["20 Creature Types"] = """
Icons: each row starts with its creature type icon: machine (gears), wind (a tornado; the PSD layer is named "tornado", hence the problem MISSPELT TYPE: tornado), rock, panda, fire, ice, water, electric, undead (a skull), dark, light, darkmatter (a dark orb), unknown (a magenta cube), girl (♀). rules.json appended " [water] [electric] [undead] [light] [unknown] [girl]" after the page number; left out above.
Diagram: the text above lists each type's description BEFORE its name; on the page the name is the heading of its row. As laid out on the page:

| creature type | effect |
|---|---|
| MACHINE | Cannot have food items be used on them. Immune to Sickness, Plague, Poison |
| WIND | Can swap out/in for 1 less point |
| ROCK | +1 Brick when mining for Bricks |
| PANDA | Immune to Stuffed, Engorged |
| FIRE | Immune to Smoldering, Burning, Blazing |
| ICE | Immune to Chilled, Frostbite, Frozen |
| WATER | Immune to Suffocation |
| ELECTRIC | Immune to Paralyzed |
| UNDEAD | Immune to Sickness, Plague |
| DARK | Immune to Afraid |
| LIGHT | Immune to Blindness |
| DARK MATTER | Immune to Corruption |
| UNKNOWN | Once per game on your turn may permanently change this type into other creature type. |
| GIRL | Gain 1 additional point (OD) |

Notes: the GIRL sentence "This may only be an innate type in special cases ..." is in the PSD text but overflows its one-line box, so the page does not show it. Types/ also has type-like icons with no row here: lightshadow and null.
"""
N["21 Damage Types"] = """
Notes: NORMAL "DMG - (1/2*DEF)" appears to be left over from the old rules. MDW/Old Rules/17_Damage_Types has the same sentence; there DEF was the shown bar value (the old DEF sheet uses whole numbers 1-10). Today's 06/07/10 pages halve the DEF bars into the DEF value and then subtract the full DEF ("Attacks deal ATK - DEF", the 10 Attacking examples). Both give the same damage if 21's "DEF" means the bars shown on the card. The rulebook never says this.
"""
N["22 Negation"] = """
Diagram: decoration only (a rock, sleepy Kirby by CREATURES, a bullet character by ITEMS).
Notes: rules.json contains a second ITEMS block from a HIDDEN layer ("Items/Item shop copy"), shown above for reference. It differs from the visible text: a negated item "returns to the hand, (or back to being face down ...)" instead of "where it was before being played"; star points "are still expended" (the visible text says "are not regained", the same outcome); and it lists only "Equip and active items", without Deploy. The visible text wins. The FIELD CARDS section repeats 18 Field Cards COUNTER-PLAY word for word.
"""
