# Impossibility Simulator rulebook: verbatim extract

Extracted 2026-10-20 from `TTSCardMaker/site/dist/data/rules.json` (built locally that day). Checks:
- rules.json has 48 entries, and `Auxiliary/Rulebook/` has 48 PSDs with 48 PNGs, one-to-one.
- AutoCroissant `stats.pkl` holds the same 48 `rulebook` entries. Their `ability` strings are byte-identical to rules.json (compared that day).
- Every page PNG was looked at. Every PSD was scanned with psd-tools for hidden text layers.

How to read this file:
- Each page is `## <rules.json name>`, its PSD path, then the page text in a `text` block, verbatim from rules.json. The only changes are the bracketed markers below.
- `[icon: X]` is an icon drawn on the page where the text has a gap or lost glyph. X is the `TTSCardMaker/Types/` file stem, or a description if it is not a type icon.
- `[hidden ... omitted]`, `[off-canvas ... omitted]` and `[overflow, not shown on page: ...]` mark text that is in the PSD (and so in rules.json, stats.pkl and /query_rulebook answers) but that a reader of the page never sees. Hidden blocks that differ from the visible text are reproduced under the page for reference.
- Trailing ` [type] [type]` tokens after a page number are the bot's leftover-icon append (type injection, see the impossibility-cards-reference skill §4). They are removed from the text and listed under "Icons:".
- Stray digits in the text (e.g. `1 2 3`, `16 2 4 6`) are callout numbers or counter values from the page's pictures. The "Diagram:" note explains them.
- The last line of most pages is the printed page number (`1`, `4.2`, `11.6`, ...). Pages 12.3-12.5 are printed "123", "124", "125".
- Page order is book order: the two cover images, then table of contents, then 01 to 22. rules.json itself lists `00 Table of Contents` before the covers. The book's own numbering (Table of Contents) is 1. Getting Started ... 22. Negation. Several chapters span more pages (04 x5, 06 x2, 07 x5, 10 x2, 11 x7, 12 x5, 14 x2, 16 x2, 19 x2).
- Old rulebook versions are in `TTSCardMaker/MDW/Old Rules/` (22 PSDs, history only; today's pages win). They are cited here only where they explain a leftover in today's text.

## 0000 Cover

`Auxiliary/Rulebook/0000_Cover.psd` (id `0000-cover`)

```text
Impossibility 
Simulator
```

Diagram: front cover art. Title "Impossibility Simulator" over a white double-infinity logo and a collage of creatures. No rules.

## 0000 Cover Book

`Auxiliary/Rulebook/0000_Cover_Book.psd` (id `0000-cover-book`)

```text
Impossibility 
Simulator
```

Diagram: the 1920x1920 texture for the Tabletop Simulator rulebook object: the cover (title, logo, collage, "Rulebook" with a Yoshi sprite, a Kirby ball) and a SpongeBob-flower spine strip. No rules.

## 00 Table of Contents

`Auxiliary/Rulebook/00_Table_of_Contents.psd` (id `00-table-of-contents`)

```text
TABLE OF CONTENTS
1. Getting Started
2. Abbreviations
3. Setting Up
4. Table
5. Shops
6. Reading Cards
7. Stats
8. The Global Turn
9. Taking Your Turn
10. Attacking
11. Abilities
12. Reactions
13. Items
14. Item Types
15. Minions
16. The Field
17. Death
18. Field Cards
19. Status Effects
20. Creature Types
21. Damage Types
22. Negation
[icon: sleepy Kirby] If you see sleepy Kirby, that means the textbox or card is explaining more technical rules. Don’t worry about understanding them too much; just refer to them if it comes up.
```

Icons: sleepy Kirby (the 8-space gap before "If you see sleepy Kirby"). It is a picture, not a type icon.

Notes: sleepy Kirby sits beside, for example, 10 Attacking2 REDIRECTIONS, the 11 Abilities6, 11 Abilities7 and 12 Reactions5 titles, 13 Items MISC, 17 Death SPECIFICS and 22 Negation CREATURES; those boxes are the technical rules. Other Kirby poses (open-mouthed, waving, starry) are just decoration or a speaker for tips.

## 01 Getting Started

`Auxiliary/Rulebook/01_Getting_Started.psd` (id `01-getting-started`)

```text
GETTING STARTED
WELCOME TO IMPOSSIBILITY SIM!
Thank you so much for playing! We’re very happy to have you with us today. 
This handy little book will run through the rules to get you started with our little card game. We hope you have fun!
TO WIN:
Defeat all of your opponent’s creatures, while making sure they don’t defeat all of yours!
Players accomplish this by using their own CREATURES and ITEMS to attack and protect.
CREATURES:
This game lives and breathes by them. Defeat your opponent’s while protecting your own. 
Above all else, make sure to follow the “Don’t be stupid” rule!
While reading is helpful, this card game can often seem like it gets complicated, fast. Rather than trying to memorize all of the mechanics and concepts before your first game, it is both encouraged and expected to jump straight into a test match against an experienced player. Make sure to ask questions about mechanics and interactions that are unclear. This game is still in beta and is always being tested, so your feedback is highly appreciated!
1
```

Diagram: three example creature cards (one of them Grey Prince Zote) beside the CREATURES box. A Dr. Eggman-style face speaks the "Don't be stupid" line. The final "1" is the page number.

## 02 Abbreviations

`Auxiliary/Rulebook/02_Abbreviations.psd` (id `02-abbreviations`)

```text
Abbreviations
 “X” is a variable that can represent any number!
GENERAL
SP = Star Points. DMG = Damage. HP = Health stat.
ATK = Attack stat. DEF = Defense stat. SPD = Speed stat.
TIME FRAME
GENERAL
(OYT) = On your turn.
(AAT) = At any time.
 PF  = Primary field.
CDMG = Counter damage.
🕐ON YOUR TURN
(XPT) = X time(s) per turn.
(EXT) = Every X turns (starting on the Xth turn).⏳
(FTEXT) = Every X turns (starting on the 1st turn).⏳
 These imply (OYT).
🕐AT ANY TIME
(XPGT) = X time(s) per global turn.
(EXGT) = Every X global turns.⏳
These imply
(AAT).
🕐FORCED
(BOT) = Beginning of turn.
(ENDT) = End of turn.
(BOGT) = Beginning of global turn.
(ENDGT) = End of global turn.
All of these must happen automatically.
FORCED
PER GAME
(OFS) = On first summon.
(OS) = On summon.
(OD) = On death.
(WOF) = While on field.
(XPG) = Ability can only be activated X time(s) per game.
Abilities marked with ⏳ are considered "multi turn" abilities.
Abilites marked with 🕐 are considered "time based".
Time based abilities are considered "refreshed" and become usable again in the specified timeframe.
2
```

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

Notes: in the extracted text the second FORCED heading and PER GAME are run together before (OFS). On the page they are two boxes: FORCED holds OFS/OS/OD/WOF, PER GAME holds XPG.

## 03 Setting Up

`Auxiliary/Rulebook/03_Setting_Up.psd` (id `03-setting-up`, rules.json types field)

```text
SETTING UP
STARTING HAND:
Each team starts with:
Each team starts with 2 points, and 7 star points.
One 5-star creature
⭐⭐⭐⭐⭐
Two 4-star creatures
⭐⭐⭐⭐ ⭐⭐⭐⭐
Three 3-star creatures 
⭐⭐⭐ ⭐⭐⭐ ⭐⭐⭐
One 2-star creature
⭐⭐
Two 4-star items
⭐⭐⭐⭐ ⭐⭐⭐⭐
 Three 3-star items
⭐⭐⭐ ⭐⭐⭐ ⭐⭐⭐
Three 2-star items 
⭐⭐ ⭐⭐ ⭐⭐
Two 1-star items
⭐⭐ ⭐⭐
Refer to “SHOPS” for setting up the shops!
Each team fills each of the 3 slots with a creature face down to start. They also start with 3 “Facedown items” placed one behind each slot. When everyone is ready, you flip up your creatures. Items are only revealed when they are played.
“Evil Quote!”
Only your side can see your "Facedown Items" while they\'re in the holders, even if they\'re face up. Pull them out of the holders to reveal and play them!
3
```

Diagram: the STARTING HAND list in two columns (creatures left, items right), each with its star count drawn. Below that is a screenshot of a hand (Items and Creatures). A Sonic ring box says "Each team starts with 2 points, and 7 star points." The table screenshot shows both teams' 3 slots with creatures face down and stat counters beside them, and the 3 facedown item holders, one behind each slot. A Dr. Eggman face says "Evil Quote!".

Notes: the starting hand is 7 creatures (1 five-star, 2 four-star, 3 three-star, 1 two-star) and 10 items (2 four-star, 3 three-star, 3 two-star, 2 one-star), which matches "You start with only 10 of them" on 13 Items. The extracted text shows two stars for each one-star item ("⭐⭐ ⭐⭐"); the page draws one star each. "Each team starts with:" appears twice because the Sonic box is a separate layer.

## 04 Table

`Auxiliary/Rulebook/04_Table.psd` (id `04-table`)

```text
TABLE
5
1
4
2
6
3
MAIN DECK
 The CREATURE DECKS. 5 star Creatures are in the leftmost deck, 4 Stars are next, then 3 Stars, then 2 Stars.
 The ITEM DECKS. 4 Star Items are in the leftmost deck, 3 Stars are next, then 2 Stars, then 1 Stars.
 FIELD CARDS for the Field card Shop (See: “SHOPS”). 4 Stars are in the leftmost deck, 3 Stars are next, then 2 Stars, then 1 Stars.
 N.M.E. Items for the N.M.E. Shop. (See: “SHOPS”)
DEVELOPER DECK
5. The Developer Version of the creature decks. This is a reduced version of the full deck, meant to only included newer or recently changed cards that require more playtesting. For most players you can ignore this but you can play with it if you want a more wacky experience!
6. “Baseline” creatures that we mix into the Developer version. These cards are considered balanced, so they serve as a way to test the less balanced cards.
4
```

Diagram: a screenshot of the Tabletop Simulator table. Upper right: (1) the four creature decks (5★, 4★, 3★, 2★ from left), (2) the item decks, (3) the field card decks, (4) the N.M.E. deck. Upper left: (5) the developer creature decks and (6) the baseline creatures. The digits 5 1 4 2 6 3 at the top of the text are these callouts.

Notes: the page prints items 1-4 as "1. The CREATURE DECKS ..." and so on. The extracted text drops the numbers and keeps only a leading space.

## 04 Table2

`Auxiliary/Rulebook/04_Table2.psd` (id `04-table2`)

```text
TABLE 2
4
3
2
1
5
6
7
 The Global Turn counter. Starts at 1, and ticks up by 1 once every filled slot on the field has taken a turn. (See: “THE GLOBAL TURN”)
 Star Point counter. At the beginning of each global turn, this is set to 7. These are spent in order to play items. Items cost their star value in star points in order to play them. (See: “ITEMS”)
 BRICKS counter. BRICKS are used to buy Field cards. Field cards cost 2x their star level in BRICKS. BRICKS are obtained when creatures spend an action to mine, gaining BRICKS equal to their star level. 1 BRICK is awarded to each team at the end of every global turn. (See: “FIELD CARDS”)
 Points counter. Each team starts the game with 2 points. You gain points equal to your creature's star level when it dies. Points are spent on abilities, and buying items from the item shop for 4x cost their star level.
The Item shop. (See: “SHOPS”)
 The Field Card shop (See: “SHOPS”)
 The N.M.E. shop (See: “SHOPS”)
Trying to understand all of the mechanics at once can get 
overwhelming. Take it slow and ask your opponent for help!
4.2
```

Diagram: the counter row and the shops. In the middle is (1) the global turn counter. On each side of it are each team's (2) star point counter (yellow), (3) BRICKS counter (red) and (4) points counter (green). Below are (5) the item shop (8 slots priced 4★ 4★ 3★ 3★ 2★ 2★ 1★ 1★), (6) the field card shop (8 slots in the same star pattern, pictured as trees) and (7) the N.M.E. shop (4 slots behind a blue curtain). The digits 4 3 2 1 5 6 7 are these callouts. Each list item is numbered 1-7 on the page; the text drops numbers 1-4.

## 04 Table3

`Auxiliary/Rulebook/04_Table3.psd` (id `04-table3`, rules.json types field)

```text
TABLE 3
1
2
3
1. A LANE on the PLAYING FIELD. There are three lanes. Your creatures occupy one side of the lanes, while your opponent’s creatures occupy the other.
2. Your FACE-DOWN ITEMS. These items can be used at any time. (See the page on ITEMS for more info).
3. A Slot. When occupied by a creature, this takes a turn.
5
4
4. STATUS EFFECT MARKERS. (See: “STATUS EFFECTS”)

5. The GRAVEYARD. Your perished creatures go in the graveyard on your side of the playing field. (See: “DEATH”)

6. The DISCARD PILE. Used items go here. 

7. FIELD CARD SLOTS. (See “FIELD CARDS”).
6
7
TIP: Mousing over anything will tell you what it is.
Useful for if you forget!
4.4
```

Diagram: the playing field seen from above. There are three horizontal lanes. Each team's creatures sit on its own side (left and right), each slot with four coloured counters and red marker squares. Each team's facedown item holder sits at the outer edge of each lane. Inset: (4) status effect markers on top of the slot, (5) the two graveyards (tombstones), (6) the discard pile (a pixel bin) and (7) the two field card slots (tree banners).

## 04 Table4

`Auxiliary/Rulebook/04_Table4.psd` (id `04-table4`)

```text
TABLE 4
1
2
3
4
1. Auxillary items deck. Items summoned by specific abilities. These items are not in the main item pool, but a few items from the main pool are in this deck as they are summoned by specific creatures.
2. Auxillary Minions deck. Where all the minions summoned by specific abilities are drawn from. (See: Minions)
3. The Minion card holder. Bring this over to the playing field and put your minion on it to keep track of summoned minion stats.
4. The Redirection cube. Use this to determine RNG.
5. The Order Determinator. Use this to determine RNG if the odds are 1 in 2.
6. 0.5 Chip incase a counter uses half integers. (See: 7.)
7. Counters, used to represent the duration of status effects, or for ability counters.
8-11:
8. Health 9. Defense 10. Attack 11. Speed markers. Use these to indicate when a creature has a stat up or down. (Flip them over for stat downs). One of these represents the actual value of a stat, not the bar. (1 DEF marker is 0.5 DEF, 1 HP marker is 1 HP)
12. Active item holder. Bring this over to the playing field and put your active item on it to keep track of its stats.
5
6
7
8
9
10
11
12
```

Diagram: (1) the auxiliary item deck (Spaghetti on top), (2) the auxiliary minion deck, (3) the minion card holder with four counters, (4) a green d6, the "Redirection cube", (5) a coin, the "Order Determinator", (6) the +0.5 chip, (7) a black counter, (8) the HP marker (green, heart), (9) the DEF marker (blue, shield), (10) the ATK marker (red, sword) and (11) the SPD marker (yellow, lightning). Last, (12) the active item holder: a radio-tower card with two counters (green HP and blue DEF), because active items have HP and DEF.

Notes: counter colours used everywhere in the book are HP green, DEF blue, ATK red, SPD yellow. One marker is one unit of the real stat value (1 DEF marker = 0.5 DEF, 1 HP marker = 1 HP), not one bar.

## 04 Table5

`Auxiliary/Rulebook/04_Table5.psd` (id `04-table5`)

```text
TABLE 5
2
1
3
4
 This is the simple version of the game. New players will draw their creatures and items from here. More complex creatures an items are removed from these decks, so it’s easier to get used to the general rules and how to play. This version of the game does not included Field Cards, or N.M.E. items. However if new players wish to play with those, they can set them up using the main deck, though it’s not recommended for learning the game.
EXTRA STUFF
2. This is a copy of the deck, kept around for various reasons. In case of accidental deletion of a card, if players want to look at the full deck and discuss the game, or for certain abilities that require you to draw from a copy of the deck.
3. This bag contains specific sorted decks of items  for certain abilities that draw from specific item pools. 
4. This bag contains tokens for each creature type, to signify if a creature type changes.
4.5
```

Diagram: the "simple version" deck block (1, upper right, creature and item decks) and the "extra stuff" (upper left): (2) the copy of the full deck, (3) a bag of sorted item decks and (4) a bag of creature type tokens. Item 1 is numbered on the page; the text drops the "1.".

## 05 Shops

`Auxiliary/Rulebook/05_Shops.psd` (id `05-shops`)

```text
SHOPS
ITEM SHOP
1) Lay out the item shop with 2 items for each star level from the item deck.
2) Reveal all the items while setting up.
2) Items in the shop cost 4x their star level in points. (Ex: a 2 star item would cost 8 points)
3) Once per global turn, each team may reshuffle this shop for 1 point by returning all items to the deck and redrawing.
4) After an item is bought, restock the slot with an item of the same star level.
FIELD SHOP
1) Lay out the field card shop with 2 field cards for each star level from the field card decks, only reveal 1 for each star level before the game stars.
2) At the end of global turn 1, reveal the remaining 4 field cards.
2) Field Cards cost 2x their star level in BRICKS.
3) The Field Shop does not restock after a card is bought.
N.M.E SHOP
1) Lay out an N.M.E. shop, with 4 random items from the N.M.E. deck.
2) Reveal all N.M.E. items before the game starts.
2) After an N.M.E. item is, restock the slot with another random N.M.E. item.
3) N.M.E items purchase cost is written on each card.
4) N.M.E items are not in the standard deck.
“Sorry Link, I can’t give credit. Come back when you’re a little MMMMMM... richer!”
My items are situational. Buy them wisely!
5
```

Diagram: the three shop layouts. ITEM SHOP has 8 item slots labelled 4★ 4★ 3★ 3★ 2★ 2★ 1★ 1★. FIELD SHOP has 8 tree slots with 4, 4, 3, 3, 2, 2, 1 and 1 stars. N.M.E SHOP has 4 TV-host slots behind a curtain. A Morshu picture says the "Sorry Link" quote; the N.M.E. host icon says "My items are situational. Buy them wisely!".

Notes: the numbering repeats ("2)" twice in each list) on the page too.

## 06 Reading Cards

`Auxiliary/Rulebook/06_Reading_Cards.psd` (id `06-reading-cards`)

```text
READING CREATURES
1
2
3
4
5
TYPES: Special markers that give creatures certian effects. (See: Creature Types)
STAR LEVEL: A creature’s star level. A higher star level means the creature is more powerful. 
From top to bottom: HEALTH, DEFENSE, ATTACK, and SPEED counters. These counters are responsible for showing the current value of a creature’s stats.
Max Health, Defense, Attack, and Speed bars. 
The health value is double of what is shown.
The defense value is half of what is shown.
ABILITIES: Text explaining what a creature is capable of doing while they are on the field. 

HEALTH: A creature perishes when its health reaches zero.
DEFENSE: Subtracted from any attack damage this creature takes.
ATTACK: The amount of damage this creature deals to opponents.
SPEED: Determines the order that creatures take their turn in (More about turn order in GLOBAL TURNS).
6
```

Diagram: a creature card with callouts. (1) The type icon at the top left. (2) The star level at the top right. (3) Four counters beside the card, from top to bottom green HEALTH 14, blue DEFENSE 1, red ATTACK 9, yellow SPEED 5, each with a red marker square to its right. (4) The four stat bars on the card. (5) The ability text box ("This creature takes 3 counter damage each time it attacks. When this creature has 50% or less HP, its attacks strike all creatures on the PF"). On the page items 1-5 are numbered, and 4 has sub-items "A. The health value is double of what is shown." and "B. The defense value is half of what is shown.".

Notes: the stat bars show the MAX value (bars x2 for HP, x0.5 for DEF, x1 for ATK and SPD). The counters show the CURRENT value.

## 06 Reading Cards2

`Auxiliary/Rulebook/06_Reading_Cards2.psd` (id `06-reading-cards2`)

```text
READING ITEMS
1
2
3
1. TYPES: Dictate how the item functions. (See: “ITEM TYPES”)
2. STAR LEVEL: An items star level. The more stars means the item is more powerful. The star level is how much the item costs to play in star points. (See: “ITEMS”)
3. EFFECTS: Text explaining the effect the item has when it is played.
6.2
```

Diagram: the item card Sacred Heart. (1) The item type icon at the top left (the equip sword icon). (2) The star level at the top right. (3) The effect box: "+50% ATK / Attacks can hit any field and ignore intercepts, but will not receive the ATK boost if this is done."

## 07 Stats1

`Auxiliary/Rulebook/07_Stats1.psd` (id `07-stats1`, rules.json types attack)

```text
STATS
HEALTH
16
2
4
6
- Each bar of health represents 2 HP. For example, this creature with 8 bars of health has 16 HP.
- When attacked, HP goes down by ATK value after DEF. When a creature’s HP reaches 0, it dies and goes to the graveyard. (See: “ATTACKING”)

- A creature’s MAX HP is the HP a creature starts with. They cannot be healed above this amount naturally. 
- An increase to the MAX HP of a creature increases their current HP by the same amount (This counts as healing). For instance, if a Mushroom (+3.5 HP) is used on this creature, the MAX and current HP would go both from 16 HP to 19.5 HP. Then, if it’s later damaged, it can be healed back up to 19.5 HP through regular means of healing. 
- If a creature’s current HP becomes greater than its MAX HP, but its MAX HP is not increased, regular healing cannot go above MAX.
-->
Increases to MAX HP are indicated with these markers. For increments of 0.5, use a marker turned to its side.
16
+
16
14
+
-->
16
->
16
+
-->
19
If this goes below 16, it can only heal up to 16 again with regular healing, as MAX HP was not increased.
+3.5
16
+
-->
19
->
If an item or ability has 2 different healing effects, it is counted as one instance of healing. For instance, if an item healed +2 MAX HP, and then +1 HP, that would only be one instance of 3 healing.
7
```

Diagram: one creature's stat bars (10 cells per stat). HEALTH has 8 lit cells (counter 16), DEFENSE 4 (counter 2), ATTACK 4 (counter 4), SPEED 6 (counter 6). The stray numbers 16 2 4 6 in the text are these counters. The healing rows read:

- 16 + "Heal 4 HP" --> 16 (already at MAX HP).
- 14 + "Heal 4 HP" --> 16 (capped at MAX).
- 16 + "+3.5 Max HP" --> 19 plus a +0.5 chip (19.5). MAX HP rises, shown by three heart markers and one sideways heart marker ("+3.5").
- 16 + "Heal 3 HP (Can go above max)" --> 19. Note: "If this goes below 16, it can only heal up to 16 again with regular healing, as MAX HP was not increased."

A pixel Alphys-style speaker gives the "2 different healing effects" note.

## 07 Stats2

`Auxiliary/Rulebook/07_Stats2.psd` (id `07-stats2`, rules.json types attack)

```text
STATS 2
HEALTH REDUCTION
16
2
4
6
- A decrease to the MAX HP of a creature does not decrease their current HP, unless if their current HP would become greater than their Max HP (in which case it is set to Max HP).
For instance, A stat stealing creature steals 1 bar (2 MAX HP) of HP. This creature would fall from 16 HP to 14 HP. If the creature was damaged and only at 10 HP, it would not lose HP, but would not be able to heal past 14.
- A decrease to the Max HP of a creature does not count as damage.
16
+
-2 MAX HP
-->
14
12
+
-2 MAX HP
-->
12
1 Bar of HP is 2 MAX HP
->
Decreases to MAX HP are indicated with these markers. 
->
7.2
```

Diagram: the same stat bars (HP 16, DEF 2, ATK 4, SPD 6). The rows read 16 + "-2 MAX HP" --> 14, and 12 + "-2 MAX HP" --> 12. Captions: "1 Bar of HP is 2 MAX HP" and "Decreases to MAX HP are indicated with these markers" (a heart with a minus).

## 07 Stats3

`Auxiliary/Rulebook/07_Stats3.psd` (id `07-stats3`, rules.json types attack)

```text
STATS 3
DEFENSE
16
2
6
4
- Each bar of Defense represents 0.5 DEF. 4 Bars = 2 DEF.
- When a creature is attacked, they take damage = ATK-DEF (See: “ATTACKING”)
2
+
-1 DEF
-->
1
2
+
-1 DEF BAR
-->
1
1 Bar of DEF is 0.5 DEF
->
Decreases to DEF are indicated with these markers. For increments of 0.5, use a marker turned to its side. 
->
7.3
```

Diagram: the stat bars with DEFENSE highlighted (4 bars, counter 2). The rows read 2 + "-1 DEF" --> 1, and 2 + "-1 DEF BAR" --> 1 plus a +0.5 chip (that is 1.5). The DEF-down marker is a shield with a minus. Link and a fox with a shield decorate the page.

Notes: the text says "-1 DEF BAR --> 1", but the picture shows the counter at 1 with the +0.5 chip, i.e. 1.5. One DEF bar is 0.5 DEF.

## 07 Stats4

`Auxiliary/Rulebook/07_Stats4.psd` (id `07-stats4`, rules.json types attack)

```text
STATS 4
ATTACK
16
2
4
6
- Each bar of Attack represents 1 ATK.
- This is how much damage a creature does minus DEF when attacking. (See: “ATTACKING”)
4
+
1.5 ATK
-->
5
Increases to ATK are indicated with these markers. For increments of 0.5, use a marker turned to its side. 
->
SPEED
16
2
4
6
- Each bar of Speed represents 1 SPD.
- This stat determines turn order and combo damage. (See: “GLOBAL TURN”, “ATTACKING”)
6
+
1.5 SPD
-->
7
Increases to SPD are indicated with these markers. For increments of 0.5, use a marker turned to its side. 
->
7.4
```

Diagram: ATTACK row 4 + "1.5 ATK" --> 5 plus a +0.5 chip (5.5). The ATK-up marker is a sword with a red plus. SPEED row 6 + "1.5 SPD" --> 7 plus a +0.5 chip (7.5). The SPD-up marker is a yellow lightning bolt with a plus. A flexing Kirby and Sonic decorate the page.

## 07 Stats5

`Auxiliary/Rulebook/07_Stats5.psd` (id `07-stats5`)

```text
STATS 5
ROUNDING + STAT CHANGES
- Stats are rounded to the nearest half in the positive direction, after reductions or gains are applied. For example, inflicting Chilled (-25% SPD) on a 5 SPD creature (5 SPD - 1.25 SPD = 3.75 SPD) rounds to 4 SPD, and inflicting Enraged on a 5 ATK creature (5 ATK + 1.25 ATK = 6.25 ATK) rounds to 6.5 ATK.

- When calculating stat changes or damage dealt, Additives are applied first and multiplicatives second.
- Multiplicative effects are applied additively i.e. Chilled (-25% SPD) + Paralyzed (-50% SPD) = -75% SPD
±0%
-25%
-50%
-75%
+25%
+50%
+75%
0.5
1
1.5
2
2.5
3
3.5
4
4.5
5
5.5
6
6.5
7
7.5
8
8.5
9
9.5
10
0.5
1
1
1.5
2
2.5
2.5
3
3.5
4
4
4.5
5
5.5
5.5
6
6.5
7
7
7.5
0.5
0.5
1
1
1.5
1.5
2
2
2.5
2.5
3
3
3.5
3.5
4
4
4.5
4.5
5
5
0
0.5
0.5
0.5
0.5
1
1
1
1
1.5
1.5
1.5
1.5
2
2
2
2
2.5
2.5
2.5
0.5
1.5
2
2.5
3
4
4.5
5
5.5
6.5
7
7.5
8
9
9.5
10
10.5
11.5
12
12.5
1
1.5
2.5
3
4
4.5
5.5
6
7
7.5
8.5
9
10
10.5
11.5
12
13
13.5
14.5
15
1
2
2.5
3.5
4.5
5.5
6
7
8
9
9.5
10.5
11.5
12.5
13
14
15
16
16.5
17.5
7.5
```

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

## 08 Global Turn

`Auxiliary/Rulebook/08_Global_Turn.psd` (id `08-global-turn`)

```text
THE GLOBAL TURN
Each slot on the playing field takes it’s turn one at a time. When every slot has taken its turn once, that is called a GLOBAL TURN. When the global turn resets, each team is set to 7 Star points.
At the end of each GLOBAL TURN, each team gains 1 BRICK.
TURN ORDER
Turns are taken lane by lane. This means that both creatures in a lane must have taken their turn before the next lane can go.

Turn order of lanes is determined by first adding up the TOTAL SPEED of the fastest creature of each side in each lane. The lane with the most total speed goes first. (Only takes into account the PRIMARY FIELD)

If the total speed is tied, then the lane with the least HEALTH LOST (MAX HEALTH - CURRENT HEALTH) goes first.
If the TOTAL HEALTH LOST is tied, then the creature with the highest STAR LEVEL goes first.

If the HIGHEST STAR LEVEL is tied, then flip a dice, or roll a coin.

The slot within each lane that goes first is determined the same way - first by looking for the most speed, then for the least amount of health lost, and then by highest star level, then by RNG.
MOST SPEED > LEAST HEALTH LOST > HIGHEST STAR LEVEL > RNG
A
HP: 8/10
SPD: 5 [icon: ★]: 3
C
HP: 8/8
SPD: 6 [icon: ★]: 4
E
HP: 7/7
SPD: 3 [icon: ★]: 5
In the example to the left, both Lane 1 and Lane 2 have a total of 10 Speed, while Lane 3 has a total of 6 Speed. Lane 1 is missing 3 HP, while Lane 2 is missing 0 HP. Therefore, Lane 2 goes first, then Lane 1, then Lane 3. 

LANE 2: Creature C has more speed than Creature D, so the Global Turn begins with C’s turn, then D takes its turn. 

LANE 1: Both creatures have equal speed, but B has less health missing. Therefore, B goes before A. 

LANE 3: Both creatures have the same speed and no health missing, so the creature with the highest star level goes first (Creature E).
L1
L2
L3
B
HP: 8/9
SPD: 5 [icon: ★]: 5
D
HP: 6/6
SPD: 4 [icon: ★]: 4
F
HP: 8/8
SPD: 3 [icon: ★]: 3
C > D > B > A > E > F
TURN MANIPULATION
Turn order may be manipulated through speed changes by (AAT) abilities and facedown items when deciding which lane goes next.
When a slots turn is about to start, the creature may swap out for 2 points. This may change turn order within the lane.
8
```

Icons: the star glyph (★ = star level) was lost from the extracted example cards; it is restored above as [icon: ★].

Diagram: the six example creatures in three lanes. A lane is a column here: the top creature is one team, the bottom the other.

| lane | top creature | bottom creature |
|---|---|---|
| L1 | A: HP 8/10, SPD 5, 3★ | B: HP 8/9, SPD 5, 5★ |
| L2 | C: HP 8/8, SPD 6, 4★ | D: HP 6/6, SPD 4, 4★ |
| L3 | E: HP 7/7, SPD 3, 5★ | F: HP 8/8, SPD 3, 3★ |

The resulting order is "C > D > B > A > E > F".

Notes: Lane 1 "missing 3 HP" is A's 2 plus B's 1: health lost is summed over the lane. The lane's speed is the fastest creature of each side added up: 5+5 = 10, 6+4 = 10, 3+3 = 6.

## 09 TakingYourTurn

`Auxiliary/Rulebook/09_TakingYourTurn.psd` (id `09-takingyourturn`)

```text
TAKING YOUR TURN
It’s time to put your creatures to use!
Every creature gets one ACTION when their turn starts. The action can be used to ATTACK, MINE, or to put the creature into DEFENSE MODE.
ATTACK
A creature can choose any creature on the playing field to ATTACK. Attacking removes HP from the chosen creature equal to the ATTACK VALUE of the attacking creature, minus the DEFENSE VALUE of the creature being attacked. (See “ATTACKING” for more info)
[hidden duplicate layer omitted]
MINING
A creature can choose to MINE, generating BRICKS equal to its STAR LEVEL for the player (see FIELD CARDS for more)
A creature can use its action to go into DEFENSE MODE, doubling its defense while it is in this state. It goes out of defense mode at the beginning of its next turn. This state is singified by turning the creature 90 degrees.
[hidden duplicate layer omitted]
DEFENSE MODE
TURN PHASES
PHASE 1
After it has been decided which slot’s turn it is, (see: “The Global Turn”), the slots turn begins and (BOT) effects Activate.
PHASE 2
This is the main part of a turn.
- May use (OYT) abilities.
- May now use items from the hand.
- May now swap facedowns to and from the hand.
- May now buy from the shops, and shuffle the item shop.
- May use actions.
- May swap primary and secondary fields if no actions have been used. (See: “THE FIELD”)
PHASE 3
- (ENDT) effects activate.
- Can no longer do anything you could in phase 2.
PHASE 4
The turn is now over.
- Status Effects deal damage/heal, duration ticks down by 1.
9
```

Diagram: decoration only (a fox and Steve with a pickaxe). The section order on the page is ATTACK, MINING, DEFENSE MODE, then TURN PHASES. In the extracted text the DEFENSE MODE heading comes after its paragraph.

Notes: rules.json repeats the ATTACK paragraph and the DEFENSE MODE paragraph. The second copy of each is a HIDDEN layer in the PSD (groups "Group 2/Attack" and "Defense Mode/Defense Mode", checked with psd-tools) and is left out above.

## 10 Attacking

`Auxiliary/Rulebook/10_Attacking.psd` (id `10-attacking`)

```text
ATTACKING
ATTACKS
- Attacks occur when creatures use an action to perform an attack, when an attack item is used, or an ability deals damage (DMG) separate from the ATK stat of a creature.
- Attacks deal ATK - DEF in damage.
- Attacks can only hit the PRIMARY FIELD by defualt. (See: FIELDS)
COMBO DAMAGE
If an attacking creature has TWICE THE SPEED VALUE of the creature being attacked, then the final damage is multiplied by 1.5. If an attacking creature has TRIPLE THE SPEED VALUE, then the final damage is doubled.
A
HP: 10/10
DEF: 2
ATK: 7
SPD: 4
Creature A attacks Creature B. What happens?

Creature B takes 
7 minus 1 Damage, for a total of 6. Their HP is reduced to 10 minus 6: they have 4 HP left.
C
HP: 10/10
DEF: 2
ATK: 7
SPD: 6
Creature C attacks Creature D. What happens?

Creature D takes 
7 minus 1 Damage, for a total of 6. However, Creature C’s speed is DOUBLE that of Creature D’s! The final damage value is therefore multiplied by 1.5, resulting in 9 total damage, leaving them at 1 HP.
L1
L1
B
HP: 10/10
DEF: 1
ATK: 6
SPD: 4
D
HP: 10/10
DEF: 1
ATK: 6
SPD: 3
SNEAK ATTACKS
Creatures have the option to perform a "sneak attack" in place of a normal attack, which is an attack that deals piercing damage equal to 25% of their attack stat +1, and has the same on-hit effects as a normal attack. This is so creatures with high defense don’t become unkillable in certain circumstances.
10
```

Diagram: two worked examples set out as cards. A (HP 10/10, DEF 2, ATK 7, SPD 4) attacks B (HP 10/10, DEF 1, ATK 6, SPD 4): 7 - 1 = 6 damage, B at 4 HP. C (HP 10/10, DEF 2, ATK 7, SPD 6) attacks D (HP 10/10, DEF 1, ATK 6, SPD 3): 6 x 1.5 = 9, D at 1 HP. Both pairs are in lane L1. A Kirby-like ninja sprite decorates SNEAK ATTACKS.

Notes: these examples subtract the full DEF (B and D have DEF 1). Compare 21 Damage Types "NORMAL: DMG - (1/2*DEF)" (see Notes there).

## 10 Attacking2

`Auxiliary/Rulebook/10_Attacking2.psd` (id `10-attacking2`)

```text
ATTACKING 2
MULTI-HIT ATTACKS
- Multi-hit attacks count as seperate attacks for each hit, and each instance may be reacted to seperately.
- When a creature uses an ability or item to attack multiple enemy creatures at once, the attacks happen left-to-right from the attacker's perspective and counter damage or on death effects both happen after all instances of the attack are resolved.
REDIRECTIONS
- Redirections & Interceptions, are instances where an attack’s target may be forecfully changed.
- Redirections change the attack’s target to a creature of choice.
- Interceptions change the target of an attack to be the creature which is intercepting.
- When a creature moves to intercept an attack, it cancels (and stays out of) defense mode first.
INFLICTIONS
- If a creature attacks or is attacked, any debuffs or stat changes inflicted on offense or defense are inflicted before the attack strikes, allowing those effects to alter the result of the attack. 
- Inflictions that occur from a creatures’ attack cannot be separated from the attack itself, and cannot be independently Negated/Reflected/Redirected/Intercepted whereas separate inflictions (from items and non-on attack abilities) can be.
- Only attack actions can proc any “on attack” effects.
ROUNDING
In the case that damage isn’t a.5 multiple, damage rounds up, for example 3.25 DMG rounds to 3.5 DMG
10.2
```

Diagram: decoration only (sleepy Kirby by REDIRECTIONS, an infliction icon by INFLICTIONS).

## 11 Abilities

`Auxiliary/Rulebook/11_Abilities.psd` (id `11-abilities`)

```text
ABILITIES
INTRODUCTION
- A card’s ability is the main chunk of text at the bottom half of a creature that describes the unique function of the card past its basic actions. 
- Most abilities are unique, but how and when they can be activated are held together and defined by abbrviations. (See: “Abbrviations”)
- A creature’s ability is only active while on the field.
WHEN ABILITES CAN BE USED
1
2
Although (OYT) and (AAT) do not often appear on their own, they are the basis of when an ability can activate.
1. This ability can activate “on your turn” (OYT). Meaning you have to wait until the creature with this ability’s turn to use it.
2. This ability can be used “at any time” meaning at any point you may use this. These are able to be use reactivley (See: REACTIONS). This ability can be used to heal a creature that is about to die from an attack reactively.
[hidden layers omitted: a PASSIVES heading and paragraph, superseded by 11 Abilities7]
11
```

Diagram: the example ability box is an IMAGE; its text is not in rules.json. Transcribed from the PNG:

1. "(OYT) Spend 3 points to cure a tier 2 debuff."
2. "(AAT) Spend a point to heal this creature by 1 HP."

The "1" and "2" in the text are these callouts.

Notes: rules.json ends with a PASSIVES heading and paragraph that are HIDDEN layers in the PSD ("Group 2/PASSIVES", "Group 2/Note copy 2"). They are left out above. The visible version of that rule is on 11 Abilities7. "Tier 2 debuff": debuffs carry star levels (the debuff cards sit in Auxiliary/Status Effects/Debuffs/<N> Stars), and "tier" means that star level. The rulebook never says so.

## 11 Abilities2

`Auxiliary/Rulebook/11_Abilities2.psd` (id `11-abilities2`)

```text
ABILITIES 2
BASIC ABILITES
(1PT) Summon a Mini Bee.
(2PGT) Heal a creature by 2 HP.
(AAT)(3PG)  Spend 7 points to deal 10 piercing damage.
1
2
3
1. (XPT) “X Per Turn” is the most common  ability type. (XPT) implies (OYT) “On your turn”. X in this case is 1. So this ability may be performed once on each turn the creature takes.
2. (XPGT) “X Per Global Turn” implies (AAT) “At any time”. So this creature can heal twice at any point during the global turn. This ability can be used to heal a creature that is about to die from an attack reactively, two times per global turn.
3. (XPG) “X Per Game” means the creature can perform this ability X times, ever. In this case it’s 3 times per game. This kind of ability is always either (AAT) or (OYT). Here it is (AAT), so it can be done at any point during the game.
“(1PG)(AAT) Kill an enemy creature”
11.2
```

Diagram: the example ability box shows "(1PT) Summon a Mini Bee.", "(2PGT) Heal a creature by 2 HP." and "(AAT)(3PG) Spend 7 points to deal 10 piercing damage.", with callouts 1-3. A Dr. Eggman face quotes "(1PG)(AAT) Kill an enemy creature" as a joke.

## 11 Abilities3

`Auxiliary/Rulebook/11_Abilities3.psd` (id `11-abilities3`)

```text
ABILITIES 3
AUTOMATIC
(BOGT) +1 SPD
(BOT) +1 ATK
(ENDT) +5 DEF until (ENDGT)
1
2
3
These abilities are non-optional and assumed to happen automatically.
1. (BOGT) “Beginning of Global turn” abilities happen before anything else in the global turn. This creature gains SPD, which will likely effect turn order.
2. (BOT) “Beginning of Turn: abilities happen when a creature begins taking it’s turn, before it does anything else. Here, gains an ATK, so it’s strengthened right before every turn it takes.
3. (ENDT) “End of Turn” Effects happen when a creature finishes its turn, after everything else it can do. Here, this effect lasts until “End of Global Turn”, meaning once the global turn finishes, the gained DEF goes away. The SPD from the earlier ability means it’ll get this ability earlier and earlier into the global turn, meaning it’ll have it’s defense for longer and longer.
Don’t give this
 ability to sonic.
11.3
```

Diagram: the example box shows "(BOGT) +1 SPD", "(BOT) +1 ATK" and "(ENDT) +5 DEF until (ENDGT)". Dr. Eggman says "Don't give this ability to sonic.".

## 11 Abilities4

`Auxiliary/Rulebook/11_Abilities4.psd` (id `11-abilities4`)

```text
ABILITIES 4
MULTI-TURN
1
(E2T) Summon a Mega Bee.
(FTE2T) Summon Brombus.
(E2GT) Deal  3 piercing damage.
2
3
1. (EXT) “Every X Turns” abilities are refreshed on the creature’s Xth turn after either the start of the game or the last time the ability was used. Here, the creature can summon a Mega Bee on its  second turn, and every 2 turns after that. (Turns 4, 6, etc.) (It’s implied these abilities can only be used (OYT) “on your turn”.)
2. (FTEXT) Brombus may be summoned on the first turn, then every 2 turns after. (Turns 3, 5, etc.) This ability happens on turns where the above doesn’t.
3.(E2GT) “Every X Global Turns”. This works the same, but ability is “at any time” (AAT), and is refreshed when the global turn starts.
If you forget to use your time based ability, it’s saved so you can use it on the next turn, but its cooldown doesn’t reset until it’s used!
Abilties do not count time passing while in the hand.
11.4
```

Diagram: the example box shows "(E2T) Summon a Mega Bee.", "(FTE2T) Summon Brombus." and "(E2GT) Deal 3 piercing damage.". Two tips: Sonic's ring ("If you forget to use your time based ability ...") and an Alphys-style face ("Abilties do not count time passing while in the hand.").

## 11 Abilities5

`Auxiliary/Rulebook/11_Abilities5.psd` (id `11-abilities5`)

```text
ABILITIES 5
LIFE AND DEATH
(OFS) Summon a Koopa.
(BOGT)/(OS) Declare which slot this creature will be attacking and it will attack all targets in that lane.
(OD) Deal 5 damage to everything on the PF.
1
2
3
1. (OFS) “On First Summon” abilities activate when a creature first comes out of the hand onto the field. These abilities cannot be negated or true negated, and are forced to happen. They also inherently count as (1PG) abilities.
2. (OS) “On Summon” abilities happen whenever a creature is summoned onto the field, and can happen an unlimited number of times. Like in the example, it is typically used to supplement (BOGT) “Beginning Of Global Turn” abilities that put the creature in a certain state, to ensure it can enter that state even if it was not on the field at the beginning of the global turn.
3. (OD) “On Death” abilities activate when a creature dies. This is also forced.
Summon me any time!
11.5
```

Diagram: the example box shows "(OFS) Summon a Koopa.", "(BOGT)/(OS) Declare which slot this creature will be attacking and it will attack all targets in that lane." and "(OD) Deal 5 damage to everything on the PF.". Decoration: Mettaton (box form) and another character with the line "Summon me any time!".

## 11 Abilities6

`Auxiliary/Rulebook/11_Abilities6.psd` (id `11-abilities6`)

```text
[off-canvas layer omitted: ABILITIES 7]
ABILITIES 6
In rare cases some abilities can conflict. If you happen to run into these, here’s what to do.
SAME TIME
- When multiple effects happen simultaneously, such as (BOGT)/(ENDGT)/conditional effects, the creature with higher SPD takes precedence.

- After conflicting creature effects go, the team with the highest cumulative SPD’s active items go, field cards are determined in the same way.

- In case of a tie for any of these, use turn order tie breakers.

- If one team has conflicting active item or field card effects on its own side, owner chooses order. (active items still go first).

- If multiple abilities conflict on a singular creature(such as multiple on death abilities), the owner may choose the order/which abilities activate.
CONFLICTION
- If both teams can set the value of the same thing, (such as setting the value of RNG outcomes), it remains random.

- In the case of multiple revives procing at the same time that set  HP to a specific amount, the HP is combined. (Cannot go above Max)
Example: Two Life shrooms would revive a creature with 10 HP.

- If two simultaneous effects have directly conflicting effects, such as reviving in your base form or a distinct revive form at the same time, the player can choose.
Example: Azamoth with lifeshroom can revive as Azamoth The Possessed, or in it’s base form as The Forgotten king, but it it chooses it’s baseform the life shroom will be wasted.
11.6
```

Notes: the first line of rules.json, "ABILITIES 7", is a header layer placed off the canvas (y = -162), so it does not show. The page title is ABILITIES 6. Sleepy Kirby sits in the title, so the whole page is "technical rules".

## 11 Abilities7

`Auxiliary/Rulebook/11_Abilities7.psd` (id `11-abilities7`)

```text
ABILITIES 7
Here are some specfic common abilities in the game and some specifics about them.
Ability Copying/Stealing/Swapping
- Stolen/Swapped abilities refresh as brand new per creature, if the creature loses that ability then gains it back, it’s not reset but instead incremented by a turn.
- If a creature gains an on first summon ability (or if this is the first opportunity that an OFS ability can be procced), proc it now.
MISC
If a healing item is used to deal DMG, it is instead treated as an attack item, and vice versa.
- Abilities that count damage dealt to creatures do not count overkill damage.
PASSIVES
(WOF) Friendly creatures gain +1 DEF.
This creature attacks inflict Burning.
This creature is immune to Poison.
This creature has 2 innate actions.
Passive abilities are abilites that happen automatically, without the player actively choosing to use it.
Most of these are in constant effect (WOF) “while the creature is on the field”. “(WOF)” is often only included when the ability effects other creatures as to signfy that it goes away when the creature exits the field.
11.7
```

Diagram: the PASSIVES example box shows "(WOF) Friendly creatures gain +1 DEF.", "This creature attacks inflict Burning.", "This creature is immune to Poison." and "This creature has 2 innate actions.". The MISC healing-item sentence is a side note by an Alphys-style face. Sleepy Kirby sits in the title.

## 12 Reactions

`Auxiliary/Rulebook/12_Reactions.psd` (id `12-reactions`)

```text
REACTIONS
1
Stats before
Stats after
2
When a play occours, a player may choose to react to it with one of their items or abilities. When you react to something, the reaction happens before whatever you reacted to. Here, is a simple reaction that is played in this order:
1. The Mighty Pebble is thrown by the enemy (5 DMG).
2. Toy Bonnie uses it’s ability to give itself 6 stacks of hardened, giving it 3 defense. 
Since Toy Bonnie reacted with it’s ability, it gains it’s defense first, and then it takes the damage from Mighty Pebble, resulting in it doing only 2 damage instead. [overflow, not shown on page: of 5.]
12
```

Diagram: (1) the Mighty Pebble item ("Deal 5 damage") and (2) Toy Bonnie (2★, machine type). Toy Bonnie's stats go from "Stats before" HP 8, DEF 0, ATK 5, SPD 5 to "Stats after" HP 6, DEF 3, ATK 5, SPD 5. Six Hardened markers sit above it.

Notes: the trailing "of 5." in rules.json overflows its text box and is not drawn.

## 12 Reactions2

`Auxiliary/Rulebook/12_Reactions2.psd` (id `12-reactions2`)

```text
REACTIONS 2
Here, the order of events are listed by number that they’re played in, which is the reverse of the order they actually happen, and nothing happens until plays stop being made reactively. 
1. Mighty pebble is thrown (5 DMG)
2. Toy Bonnie reacts to the pebble with it’s ability to gain 6 Hardened (3 DEF) to make the damage from the Pebble 2 instead of 5.
3. The enemy reacts with Metronome, to try to negate (turn off) Toy Bonnie’s ability before it can give itself defense, so that the pebble would do 5 damage.
4. The player reacts to Metronome with Stopwatch, negating the Metronome before it can negate Toy Bonnie.
Once both players stop reactively using items, they then resolve and all the effects take place. So here, the order of events is 4>3>2>1. Resulting in the Pebble only doing 2 damage.
3
4
1
Stats before
Stats after
2
[hidden layer omitted: an older draft of the 12 Reactions3 text; see Notes]
12.2
```

Hidden layer text (not shown on the page), kept for reference:

```text
Although reactions happen in reverse order, anything that deals damage resolves last. Meaning all other effects take place before damage is dealt. Attacks also resolve forward, meaning the first damage item played resolves first.
1. The Mighty Pebble is thrown by the enemy (5 DMG).
2. Toy Bonnie uses it’s ability to give itself 5 stacks of hardened, giving it 3 defense. 
3. The enemy throws a bomb (7 DMG).
4. Toy Bonnie is equipped with metal sheild. (+1.5 DEF).
Now, the reaction chain is over. If their were no attacks, the order the plays resolve in would be 4>3>2>1, however since 2 of these do damage, the order of is actually 4>2>1>3.
So Toy Bonnie gains 4.5 defense from the sheild and hardened, then the mighty pebble does.5 damage, lastly the bomb deals 2.5 damage. So Toy Bonnie takes 3 damage total.
```

Diagram: (1) Mighty Pebble, (2) Toy Bonnie, (3) Metronome (a negation item) and (4) Stopwatch. Toy Bonnie goes from HP 8/DEF 0/ATK 5/SPD 5 to HP 6/DEF 3/ATK 5/SPD 5.

Notes: rules.json also contains a second block, a HIDDEN layer ("INTRODUCTION/Item shop copy"), shown above for reference. It is an older draft of 12 Reactions3 that adds "Attacks also resolve forward, meaning the first damage item played resolves first", and so gets the order 4>2>1>3 (pebble 0.5, then bomb 2.5). The visible 12 Reactions3 gets 4>2>3>1 instead: the later damage resolves first, i.e. damage also resolves in reverse. The visible page wins.

## 12 Reactions3

`Auxiliary/Rulebook/12_Reactions3.psd` (id `12-reactions3`)

```text
REACTIONS 3
3
1
Stats before
Stats after
4
2
Although reactions happen in reverse order, anything that deals damage resolves last. Meaning all other effects take place before damage is dealt. 
1. The Mighty Pebble is thrown by the enemy (5 DMG).
2. Toy Bonnie uses it’s ability to give itself 5 stacks of hardened, giving it 3 defense. 
3. The enemy throws a bomb (7 DMG).
4. Toy Bonnie is equipped with metal sheild. (+1.5 DEF).
Now, the reaction chain is over. If there were no attacks, the order the plays resolve in would be 4>3>2>1, however since 2 of these do damage, the order of is actually 4>2>3>1.
So Toy Bonnie gains 4.5 defense from the sheild and hardened, then the Bomb deals 2.5 damage, lastly the Mighty Pebble does 0.5 damage. So Toy Bonnie takes 3 damage total.
12.3
```

Diagram: (1) Mighty Pebble, (2) Toy Bonnie, (3) Bomb ("Deal 7 damage") and (4) Metal Shield, whose card reads "+1.5 DEF -.5 SPD". Toy Bonnie goes from HP 8, DEF 0, ATK 5, SPD 5 to HP 5, DEF 4 + 0.5 chip (4.5), ATK 5, SPD 4 + 0.5 chip (4.5). The SPD drop comes from Metal Shield's card; the text does not mention it.

Notes: the text says "5 stacks of hardened, giving it 3 defense". Hardened is +0.5 DEF per stack (Auxiliary buff card), so 3 DEF needs 6 stacks, as on 12 Reactions/12 Reactions2. The 4.5 total = 3 + 1.5 also needs 6 stacks. The page number is printed "123" (meaning 12.3).

## 12 Reactions4

`Auxiliary/Rulebook/12_Reactions4.psd` (id `12-reactions4`)

```text
REACTIONS 4
1
5
3
4
2
Whatever was originally being reacted to within a reaction chain will always be resolved at the very end.
1. Whispy Woods attacks 2 HP Toy Bonnie, with 4 ATK, attempting to kill it.
2. Toy Bonnie reacts by activating it’s ability, giving it 3 DEF, which would let it live at 1 HP.
3. The enemy reacts with a bomb (7 DMG). As it is an attack item, it will resolve after Toy Bonnie gains its defense, meaning it will only do 4 damage, killing Toy Bonnie with 2 damage overkill. This would happen before Whispy Wood’s attack goes off though as it was played later in the attack chain. (If the chain ended here, Whispy Wood’s action would be refunded as the bomb would kill Toy Bonnie before it could attack)
4. The player uses pep drink (Heal 4 HP) on Toy Bonnie. As this happens before the attacks, Toy Bonnie heals up to 6 HP, then after the Bomb is used, is lowered to 2 HP. If the chain were to end here, Whispy woods would now do 1 damage and Toy Bonnie would live at 1 HP.
5. Whispy Wood’s equips Amy Rose’s Hammer, (Attacks inflict 2 Softened). Although the equip now happens before anything else, the attack happens last, so the Softened is only applied on the attack from Whispy Woods, and does not effect Bomb damage. As on hit effects apply before attacks happen however, when Whispy Woods finally attacks, Toy Bonnie loses 1 defense, and the attack does 2 damage, which kills Toy Bonnie exactly.
Resolve Order: 5>4>2>3>1
12.4
```

Diagram: (1) Whispy Woods (counters HP 21, DEF 2, ATK 4, SPD 3; ability upside-down "(OPT) You can spend 2 points to heal all friendly creatures by 3 HP"), (2) Toy Bonnie (HP 2, DEF 0, ATK 5, SPD 5), (3) Bomb, (4) Pep Drink ("Heal a creature by 2.5 HP and give it 3 turns of 1 Peppy, or heal [icon] a by 4 HP") and (5) Amy Rose's Hammer. Softened (-0.5 DEF, counter 2) and Hardened (+0.5 DEF) markers are shown. Three "after" columns of counters show HP -2/DEF 3, then HP 2/DEF 3, then HP 0/DEF 2: the state if the chain ended after step 3, after step 4, and the final result. Toy Bonnie's card ability reads "(1PGT) Inflict up to 6 stacks of Softened for 1 turn distributed across any creatures, and this creature inflicts an equal number of stacks of Hardened on itself for 1 turn." (small print; the tag may read (OPGT)). The page number is printed "124" (12.4).

## 12 Reactions5

`Auxiliary/Rulebook/12_Reactions5.psd` (id `12-reactions5`)

```text
REACTIONS 5
Other things to note about reactions:
SPENDING
- Star points are used immediately upon an item being played rather than when they are being resolved.
- If a reaction prevents a creature from being targeted by an item, (i.e forcibly switched out/effect preventing targeting/killed beforehand), when the item is suppose to resolve, the item is not used, returns to it’s origin and its star points are regained.
- If a reaction leads to a creature dying before an attack is resolved, then the attacker gets back the resource, whether it’s an item/action/ability.
NO TAKE BACKS
- A decision to attack or use an ability/item cannot be retracted if your opponent reacts to it in some way.
DURING REACTIONS
- Items may be played from your hand reactively only if it’s currently your turn.
- While a reaction is occuring, nothing can be purchased from any shops.
- Items may be reactively unequipped and re-equipped to another creature if it is your turn.
CLARIFICATION
- You may react to time based [such as (BOGT)] effects with any (AAT) effects or facedown items.
12.5
```

Notes: sleepy Kirby sits in the title, so this is "technical rules". The page number is printed "125" (12.5).

## 13 Items

`Auxiliary/Rulebook/13_Items.psd` (id `13-items`)

```text
ITEMS
FUNCTION
Items are your main resource besides creatures. You start with only 10 of them, spend them well!
- Items are what your Star Points are for. In order to play an item, you must spend Star Points equal to the star level of an item. The stars are in the top right of an item. For example, A 3 star item costs 3 star points to play.
- At the start of every global turn, you are set to 7 Star Points.
-Once an item is played, the item is put into the discard pile after it resolves; unless it’s a multi-use item, then it goes back facedown or to the hand.
- You have 3 “facedown” slots. Items in these slots can be played at any time!
- You may only swap items between your hand and the facedown slots on your turn. Plan carefully!
SELLING ITEMS
During your turn you can sell items for points equal to their star level. You can also sell items that are facedown at any time.
You cannot sell an item that has previously been used. (Equips Included)
MISC
Some effects require you to draw a random item. In this case, draw from a copy of the full deck until an eligible item is found. After the eligible item is drawn, do not replace it in the copy deck, and set aside the copy deck for next time a random item is drawn from this effect. If the entire deck is drawn and no eligible items were found, make a new copy of the full deck, or if there cannot possibly be any eligible items, this effect does nothing.
“I have come here to steal the flames of disaster from you! And to take the miracle gems that are the key to its' secret, The Chaos Emerald!”
Refer to “ITEM TYPES” to understand how each type of item works.
13
```

Diagram: a row of item art at the top. Mr. Krabs (money) marks SELLING ITEMS. A Dr. Eggman face says the "Chaos Emerald" quote. Sleepy Kirby marks MISC.

## 14 Item Types

`Auxiliary/Rulebook/14_Item_Types.psd` (id `14-item-types`, rules.json types attack, machine, food)

```text
ITEM TYPES
Item types are the symbol are the top left of an item that dictate how the items work. These items types cannot be manipulated.
The item types displayed here are items that by default are discarded after being played.
HEALING
Also referred to as “food” these items heal creatures. May not be used on [icon: machine].
ATTACK
These items do damage, and count as an attack when used.  (See: “ATTACKING”)
NEGATION
These items “Negate” something, taking it out of play. (See: “ABILITIES”,“ITEMS”, “ATTACKING”)
SUPPORT
These items are general use items with various effects. [overflow, not shown on page: if you find this text dont delete it]
COUNTER
These items can only be activated under certain conditions. These can only be used reactively. These items tend to have multiple uses and are only discarded when all uses have been used.
INFLICTION
These items inflict a status effect and count as an infliction.
14
```

Icons: each heading carries its item type icon: HEALING (Food, a tomato), ATTACK (a bomb), NEGATION (a metronome), SUPPORT (a blue ring), COUNTER (spiky grass), INFLICTION (Mr. Krabs on fire). On the page the HEALING sentence ends with the machine icon (gears): "May not be used on [machine]." This matches 20 Creature Types (MACHINE: "Cannot have food items be used on them").

Notes: rules.json says "May not be used on [support]." That is a parser mis-injection: the machine icon is above the card midline, so it went to the page's types, and the gap got the next icon in reading order (support). The trailing "[infliction]" in rules.json is a leftover icon appended after the page number. The sentence "if you find this text dont delete it" overflows its text box and is not drawn.

## 14 Item Types2

`Auxiliary/Rulebook/14_Item_Types2.psd` (id `14-item-types2`, rules.json types equip)

```text
ITEM TYPES 2
These item types do not get discarded immediately after played.
EQUIP
- These items you may equip onto your own creatures and grant a passive effect to the creature they are on which is lost when unequipped.
- May be unequipped and returned to your hand during your turn.
- If an equip has a negative effect, cannot be unequipped until the start of the global turn after it was equipped.
- If equipped creature dies, return equip to your hand.
Equip items stay equipped onto creatures swapped into the hand, and may be unequipped from creatures in your hand on your turn.
- Equip items may be equipped to an enemy if they have no negative effects, but the enemy would then own the item.
- An equip can be “Forcibly” unequipped by an effect at any time regardless of if the item has negative effects or is on cooldown. (Unless specified that it cannot be unequipped).
FRAGILE EQUIP
These items are same as equips but cannot be unequipped by any means and are discarded when the creature they are on dies.
ACTIVE
These items stay on the field with a constant effect until they are destroyed. They have health and defense, although they are not creatures, cannot have items equipped to them, cannot have healing items used on them and cannot be inflicted with status effects. When HP reaches zero, they are discarded.
DEPLOY
These items stay on a chosen slot (doesn’t count as being used on the creature in that slot) with a constant effect. They cannot be attacked, but wear off through other means.
14.2
```

Icons: EQUIP (a sword with a red ribbon), FRAGILE EQUIP (the same sword with an alarm clock), ACTIVE (a radio tower) and DEPLOY (a Peashooter). rules.json appended the leftover heading icons as " [fragile equip] [active] [deploy]" after the page number; they are left out above.

## 15 Minions

`Auxiliary/Rulebook/15_Minions.psd` (id `15-minions`)

```text
MINIONS
INTRODUCTION
- Minions are small creatures that when summoned are attached onto a parent prime creature.
- Minions are always attached to their parent creature, and stick with them even if the parent creature is moved or swapped into the hand.
- For most purposes, Minions are considered creatures, and have their own action, but they do not yield points on death. They do go to the graveyard on death.
- Every minion and parent creature in the primary field may use their actions (See: “THE FIELD”) in whatever order the player choses.
- Minions cannot mine for BRICKS.
- If Parent creature dies, all minions in the slot die (excluding persistent minions)
TYPES:
[icon: offensive_minion]:Offensive minions occupy the same field as the summoner. (See: “FIELDS”)
[icon: defensive_minion]:Protective Minions always occupy whatever field the parent creature and offensive minions are not in. When summoned, these push the parent creature (and the offensive minions attached to it) back to the secondary field, and are placed in the primary field. As a creature can only act in the primary field, This typically cannot act on the same turn as a parent creature, as they can never both occupy the primary field. (See: “THE FIELD”)
- If a protective minion is summoned after a parent creature has already used an action that turn, it does not gain an action for that turn.
PERSISTENT
If a minion is “persistent”, when its parent creature dies, it will be attached to the next prime creature that gets placed into the slot. If there is no creature to fill the slot, it occupies the slot alone, and must be killed for the game to be won by the opponent.
15
```

Icons: under TYPES, the offensive minion icon (a brown bunny-eared head, Types/Offensive_Minion.png) starts the offensive line, and the protective minion icon (a green Koopa shell, Types/Defensive_Minion.png) starts the protective line. rules.json lost both gaps ("TYPES:Offensive", "):Protective") and appended "[defensive_minion]" after the page number. Restored above.

Notes: the rulebook says "protective minion" and the icon file is named Defensive_Minion. They are the same thing.

## 16 The Field

`Auxiliary/Rulebook/16_The_Field.psd` (id `16-the-field`)

```text
THE FIELD
THE FIELD
- The place that creatures are placed on is called the field, which is organized into 6 slots, 3 for each team.
- Every slot must be filled if possible. You must immediately summon a creature into an empty slot if possible.
- Each slot containing creature(s) takes one turn during each global turn.
- Each slot is split into two fields, the PRIMARY FIELD and the SECONDARY FIELD
- If an entire slot/lane of creatures is dead, then skip over that slot/lane, but count time as if that slot/lane had gone.
PRIMARY FIELD
- By default, creatures are placed onto the primary field. This is the only field that can be targeted by default, and the only field creatures can act in. Usually, this is the only field to exist in a slot.
- If something targets “everything in the Primary Field”, it is referring to all 3 slots.
SECONDARY FIELD
- Behind the primary field is the secondary field. This is created by something pushing a creature back into it, either a protective minion or a protective active item designated by this symbol: [icon: defensive_minion]
- By default this field cannot be targeted.
- Once you kill everything in the primary field, anything in this field is brought back to the primary field.
- A creature in this field, cannot use actions, but can use abilities.
- A parent creature can swap fields with its protective minions once during its turn, but only if no actions have been used in the slot yet.
1. The Parent creature, in the secondary field. May not use actions, may use abilities, cannot be targeted.
2./3. Protective minions; can use actions, can be targeted, can use abilities.
He pissed on my fucking wife
2
3
1
16
```

Icons: "designated by this symbol:" is followed by the protective minion icon (the green shell). In rules.json the icon is appended after the page number instead.

Diagram: a slot with (1) the parent creature (an "Agent" card with counters) at the front-bottom, and (2) and (3) two protective minions above it. The page calls (1) the parent creature in the secondary field. A Dr. Eggman face says the joke line.

## 16 The Field2

`Auxiliary/Rulebook/16_The_Field2.psd` (id `16-the-field2`)

```text
THE FIELD 2
DIFFERENT TYPES OF MINIONS AT ONCE
- Offensive minions are always in the same slot as the parent creature.
- All protectors in a field together, which is the field that the parent creature is not in regardless of summoner.
- Minions that are summoned by other minions are automatically attached to the parent prime creature of the slot. Protective minions are placed in the same field as other protective minions, and offensive minions are placed in the same field as the parent creature, regardless of summon origin.
5
6
1
4
3
2
1. Primary Field
2. Secondary Field
3. Parent creature in the secondary field. [Cannot use actions]
4. Offensive minion attached to the parent creature by an outside source in the secondary field. [Cannot use actions]
5. Protective minion summoned by the parent creature in the PRIMARY FIELD. [Can use actions]
6. Protective minion summoned by the protective minion on the left in the PRIMARY FIELD. [Can use actions]
*If there was a protective active item here, it would be in the primary field alongside the protective minions.
16.2
```

Diagram: one slot in two coloured boxes. The green box (1, PRIMARY FIELD) holds two protective minions with green shells, (5) Yarn and (6) Grey Prince Zote. The red box (2, SECONDARY FIELD) holds (4) an offensive minion, Gulka, and (3) the parent creature Holo Defense MK, with counters 0/2 +0.5/5/5. The digits 5 6 1 4 3 2 are these callouts.

## 17 Death

`Auxiliary/Rulebook/17_Death.psd` (id `17-death`)

```text
DEATH
ON DEATH
- When a friendly creature’s HP reaches 0 it dies, and goes to the graveyard on your side of the field.
- When this happens, your team gains points equal to that creatures star level.
- When a creature dies, it’s “On Death” effects proc. (Does not apply to creatures not on the field)
REVIVES
- Creatures that are revived from “On Death” effects only yield Points when they are fully killed.
- For creatures revived by other means, Points are kept from it dying, but killing the revived version awards 0 Points.
SPECIFICS
- Creatures composed of multiple targets only award points when all targets are killed.
- A creature being “destroyed” instantly by any effect, causes it to take enough damage to kill it. (regardless of healing or any possible factors)
“If you die outside your game, you DON’T regenerate!
17
```

Diagram: a large tombstone, a knocked-out character and a ghost by REVIVES. Sleepy Kirby marks SPECIFICS. Sonic's ring speaks the quote.

## 18 Field Cards

`Auxiliary/Rulebook/18_Field_Cards.psd` (id `18-field-cards`, rules.json types field)

```text
FIELD CARDS
 These are cards that when on the playing field have a constant effect.
OBTAINING
- Field cards are cards that can be obtained from the Field shop, purchased with BRICKS. (See: “SHOPS”)
- Prime creatures may spend an action to MINE, to generate an amount of BRICKS equal to their star level.
PLAYING
- When these are obtained, they may immediately be put into play on your teams field card slot, for no extra cost.
- You may take these out of play and put them in your hand. 
- If you want to put a field card from your hand into play, it’ll cost star points equal to the card’s star level, just like a regular item.
- You may have only 1 of these out at a time.
COUNTER-PLAY
- At the beginning of the global turn, or on your turn, you may spend 1x the star level of a field card in BRICKS to negate it until the end of the global turn (field card stays on the field and can be swapped out, but loses its effect temporarily).
- Field cards cannot be negated on the same turn that they are played.
- Field cards cannot be negated 2 turns in a row.
- At the beginning of the global turn, or on your turn, you may spend 3x the star level of a field card in BRICKS to destroy it.
“I kissed a tree in the jungle and I liked it
```

Icons: the field type icon (a Terraria tree) decorates the page; rules.json lists types: field. A Dryad sprite speaks the quote.

Notes: the COUNTER-PLAY bullets are repeated word for word under FIELD CARDS on 22 Negation.

## 19 Status Effects

`Auxiliary/Rulebook/19_Status_Effects.psd` (id `19-status-effects`)

```text
STATUS EFFECTS
STATUS EFFECTS
- A status effect, is either a buff or debuff that has a an of effect on a creature, represented by Mr. Krabs either suffering or thriving.
- If a status effect deals DMG or Heals, then by default it procs at (ENDT). Otherwise, by the status effect take effect immediately.
- The duration of any status effect counts down at (ENDT).
This Panda was just inflcited with both 2 turns of Softened (-0.5 DEF), and 2 turns of Smouldering (2 DMG). Instantly, the DEF reduction takes effect. When Panda ends its next turn, it takes 2 damage from smouldering, and the duration for both of the debuffs goes from 2 to 1. 
SPECIFICS
- DMG dealing status effects count as one instance of damage per unique stack. (5 stacks of Poison is one instance of 5 damage.)
- Per unique status effect at (ENDT), the owner of the creature chooses the order in which to resolve these effects. (May chose if Regeneration heals before Smouldering does damage first)
- If a status effect deals DMG it is piercing, and doesn’t count as an attack.
- If a creature started the global turn with a status effect and the global turn ends without the creature having ended its turn, then the effect ticks at the end of the end of the global turn, even if the creature is in hand.
19
```

Diagram: Panda (panda type, 2★) with the Softened card (-0.5 DEF, counter 2) and the Smouldering card (counter 2) above it. Its counters are HP 16, DEF 0 + 0.5 chip, ATK 7, SPD 6. The Mr. Krabs pictures are the status effect cards: thriving Krabs for buffs, suffering Krabs for debuffs.

## 19 Status Effects2

`Auxiliary/Rulebook/19_Status_Effects2.psd` (id `19-status-effects2`, rules.json types buffstack, debuffstack, buff, debuff)

```text
STATUS EFFECTS 2
SYNTAX
Statuses are formatted like "[X]T[Y] [Status]", where X is duration and Y is stacks. For example, "5T1 Poison" means 5 turns of 1 stack of Poison.
TYPES
Standard status effects can only have one instance of themselves applied to a single creature. [icon: buff] [icon: debuff]
Stacking status effects can have any number of themselves applied. [icon: buffstack] [icon: debuffstack]
This buff can be applied many times to gain as many stacks of it to gain alot of speed. 10 stacks = +10 SPD.
This debuff cannot be applied more than once.
DURATION
When a status effect is stacked/upgraded/reapplied, set the duration to the longest lasting of the applications.
UPGRADABLE [icon: debuffupgrade]
Certain status effects upgrade themselves when applied, Ex:
Smouldering + Smouldering -> Burning
Smouldering/Burning + Burning -> Blazing
Chilled + Chilled -> Frostbite
Chilled/Frostbite + Frostbite -> 1 turn of Frozen
Sickness + Sickness  -> Plague
A creature already inflicted by Frozen/Blazing/Plague can’t have the prior debuffs reapplied onto them, except to refresh the duration (need to upgrade to refresh duration).
->
->
->
->
->
19.2
```

Icons: the buff (single orange up arrow) and debuff (single blue down arrow) icons follow the Standard sentence. The buffstack (double orange arrows) and debuffstack (double blue arrows) icons follow the Stacking sentence. The debuffupgrade icon (a down arrow with a second arrow) sits by UPGRADABLE. rules.json put "[debuff]" after the page number; restored above.

Diagram: examples. Peppy (buffstack, "+1 SPD") beside "This buff can be applied many times ...". Withering (debuff, 4★, "(BOT) 2 DMG -0.5 DEF -1 ATK -1 SPD.") beside "This debuff cannot be applied more than once.". The upgrade chains drawn as cards:

- Smouldering (1★, "2 DMG, can't affect [fire]") -> Burning (1★, "3 DMG, can't affect [fire]") -> Blazing (2★, "4 DMG, can't affect [fire]").
- Chilled (2★, "-25% SPD can't affect [ice]") -> Frostbite (2★, "2 DMG. -25% SPD, can't affect [ice]") -> Frozen (3★, "Can't affect [ice]. Lose all actions, cannot switch out.").
- Sickness (1★, "-0.5 DEF -1 ATK -1 SPD can't affect [undead] [machine]") -> Plague (2★, "-1 DEF -1 ATK -2 SPD Can't affect [machine] [undead]").

(Card texts checked against cards.json.) The card file is spelled "Smoldering"; this page spells it "Smouldering".

## 20 Creature Types

`Auxiliary/Rulebook/20_Creature_Types.psd` (id `20-creature-types`, rules.json types ice, fire, panda, rock, machine, problems MISSPELT TYPE: tornado)

```text
CREATURE TYPES
Every type has a unique function, typically an immunity to a debuff.
Cannot have food items be used on them
Immune to Sickness, Plague, Poison
MACHINE
Can swap out/in for 1 less point
WIND
+1 Brick when mining for Bricks
ROCK
 Immune to Stuffed, Engorged
PANDA
Immune to Smoldering, Burning, Blazing
FIRE
Immune to Chilled, Frostbite, Frozen
ICE
Immune to Suffocation
WATER
Immune to Paralyzed
ELECTRIC
Immune to Sickness, Plague
UNDEAD
Immune to Afraid
DARK
Immune to Blindness
LIGHT
Immune to Corruption
DARK MATTER
Once per game on your turn may permanently change this type into other creature type.
UNKNOWN
Gain 1 additional point (OD)
[overflow, not shown on page: This may only be an innate type in special cases, but may still be added to creatures through effects.]
GIRL
20
```

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

## 21 Damage Types

`Auxiliary/Rulebook/21_Damage_Types.psd` (id `21-damage-types`)

```text
DAMAGE TYPES
NORMAL
Reduced by half of the defense of the target. 
DMG - (1/2*DEF)
PIERCING
Ignores DEF.
COUNTER
Can only be dealt as retaliation to an attack. Does not count as an attack, and can bypass abilities and protective minions as a result.
LIFE STEAL
DMG dealt after DEF is healed before the attack resolves, with a percent scalar where 100% is 1:1, 50% is 1:2, 200% is 2:1, etc.
SHOCKWAVE
Deals damage to all enemy targets on the primary field aside from the original target, and also the secondary field slot behind the main target (This does not count as an attack).
TRUE DAMAGE
Ignores DEF, and cannot be negated, redirected, or reduced by any means.
21
```

Notes: NORMAL "DMG - (1/2*DEF)" appears to be left over from the old rules. MDW/Old Rules/17_Damage_Types has the same sentence; there DEF was the shown bar value (the old DEF sheet uses whole numbers 1-10). Today's 06/07/10 pages halve the DEF bars into the DEF value and then subtract the full DEF ("Attacks deal ATK - DEF", the 10 Attacking examples). Both give the same damage if 21's "DEF" means the bars shown on the card. The rulebook never says this.

## 22 Negation

`Auxiliary/Rulebook/22_Negation.psd` (id `22-negation`)

```text
NEGATION
ATTACKS
If an attack is negated it does not happen. No damage is dealt, no on-hit effects occur, the creature does not get its action back, you do not pass go.
CREATURES
- When a creature is negated it is unable to use any of its own abilities (unless otherwise stated), but may still use the abilities of any equipped items or minions, and can still use actions.
- Multi turn abilities such as (E2T) abilities still count time passing even while they are negated.
- Minions/Items/Stats/Counters summoned/gained by an ability remain even if the ability that summoned them is negated or if the creature that summoned them dies. 
Immunity:
- 5+ star creatures are immune to negation.
- True Negation: Negation that can negate negation-immune creatures.
- True Negation Immunity: Complete immunity to any possible sources of negation, including true negation.
- (OFS) abilities have true negation immunity.
- If an item/ability does something upon being negated, the “upon being negated” effect itself has true negation immunity.
ITEMS
- If an item is negated, it means it cannot be played or is taken out of play, and it returns to where it was before being played (such as the hand or facedown slot).
- If an item is negated or destroyed during use, the star points expended to use it are not regained.
- Items that are currently negated cannot be sold, or used for any other purpose until the negation ends.
- Equip, Active, and Deploy items can be negated after their activation while they are still in effect; this sends them back to the hand, requiring star points to be used again to reactivate them.
[hidden layer omitted: an older draft of the ITEMS bullets; see Notes]
FIELD CARDS
- At the beginning of the global turn, or on your turn, you may spend 1x the star level of a field card in BRICKS to negate it until the end of the global turn (field card stays on the field and can be swapped out, but loses its effect temporarily).
- Field cards cannot be negated on the same turn that they are played.
- Field cards cannot be negated 2 turns in a row.
- At the beginning of the global turn, or on your turn, you may spend 3x the star level of a field card in BRICKS to destroy it.
22
```

Hidden layer text (not shown on the page), kept for reference:

```text
- If an item is negated, it means it cannot be played or is taken out of play, and it returns to the hand, (or back to being face down if it was denied play from a facedown slot.)
- If an item is negated or destroyed during use, the star points expended to use it are still expended.
- Items that are negated cannot be sold, or used for any other purpose until the negation ends.
- Equip and active items can be negated after their activation while they are still in effect, this sends them back to the hand, requiring star points to be used to reactivate them.
```

Diagram: decoration only (a rock, sleepy Kirby by CREATURES, a bullet character by ITEMS).

Notes: rules.json contains a second ITEMS block from a HIDDEN layer ("Items/Item shop copy"), shown above for reference. It differs from the visible text: a negated item "returns to the hand, (or back to being face down ...)" instead of "where it was before being played"; star points "are still expended" (the visible text says "are not regained", the same outcome); and it lists only "Equip and active items", without Deploy. The visible text wins. The FIELD CARDS section repeats 18 Field Cards COUNTER-PLAY word for word.

## Summary: what the rulebook covers

By chapter (book numbering):
1. Getting Started: the goal (defeat all of the opponent's creatures) and the "Don't be stupid" rule.
2. Abbreviations: stat names and all 16 timing tags. 🕐 marks "time based" (refreshed) tags, ⏳ marks "multi turn" tags.
3. Setting Up: the starting hand (7 creatures, 10 items), 2 points and 7 star points, face-down creatures, 3 facedown items.
4. Table (5 pages): every Tabletop Simulator object. Decks by star level, developer and simple decks, counters, lanes and slots, graveyard, discard pile, auxiliary decks, markers and chips, holders, bags.
5. Shops: item shop (4x stars in points, reshuffle for 1 point once per global turn), field shop (2x stars in BRICKS, no restock), N.M.E. shop.
6. Reading Cards: creature and item card anatomy. The bars show MAX stats (HP bars x2, DEF bars x0.5); the counters show current stats.
7. Stats (5 pages): bar values, MAX HP increases and decreases, markers, the rounding rule (nearest 0.5, halves up), additive before multiplicative, multiplicative effects added together, and the lookup table.
8. The Global Turn: lane-by-lane turn order (most speed > least health lost > highest star level > RNG), 7 star points each global turn, +1 BRICK at its end, swapping out for 2 points.
9. Taking Your Turn: one action (attack, mine, defense mode) and the four turn phases.
10. Attacking (2 pages): ATK - DEF, combo damage (x1.5 at 2x speed, x2 at 3x), sneak attack, multi-hit, redirection and interception, inflictions on attack, damage rounding.
11. Abilities (7 pages): when each tag can be used, refresh and cooldown behaviour, conflict and simultaneity order, copied/stolen abilities, passives.
12. Reactions (5 pages): reaction chains resolve in reverse, damage resolves last, the original play resolves at the very end; refunds, no take backs, limits during reactions.
13. Items: star point cost, the 3 facedown slots, selling, random-item draws.
14. Item Types (2 pages): Healing (food), Attack, Negation, Support, Counter, Infliction (discarded after use); Equip, Fragile Equip, Active, Deploy (stay in play).
15. Minions: parent prime creature, offensive vs protective minions, persistent minions.
16. The Field (2 pages): 6 slots, primary vs secondary field, protective minions pushing the parent back.
17. Death: points on death, revives, multi-target creatures, "destroyed".
18. Field Cards: obtained with BRICKS, played free on purchase or for star points from hand, one in play, negate for 1x stars in BRICKS or destroy for 3x.
19. Status Effects (2 pages): buffs and debuffs, ticking at (ENDT), the `[X]T[Y]` syntax, standard vs stacking, duration refresh, upgrade chains.
20. Creature Types: 14 types and their immunities or perks.
21. Damage Types: normal, piercing, counter, life steal, shockwave, true damage.
22. Negation: negating attacks, creatures, items and field cards; 5+ star immunity; true negation and true negation immunity.

## Summary: what the rulebook leaves implicit, inconsistent or undefined

Undefined terms and tags:
- Card texts use timing tags the rulebook never defines (counts from cards.json, 2026-10-20). (XCD) appears 27 times ("(2CD) Attacks hit 2 separate creatures."), (XGCD) 24 times ("(1GCD) Increase a friendly creature's ATK by 1.") and (XU) 8 times ("(1CD)(2U) Draw one ingredient:"). By context these mean an X-turn cooldown, an X-global-turn cooldown and X uses; that reading is an inference. Old Rules defined "(CD) = Counter damage", which today is CDMG.
- "Prime creature" (15, 18) is never defined. By context it is a non-minion creature, the one minions attach to.
- "Tier" of a debuff (11) is never defined. Debuff cards have star levels (1-4), which is what tier means; buffs have no stars.
- "Proc", "instance", "on-hit", "reactively", "cleanse" and "innate actions" are used, not defined.
- N.M.E. is never expanded. Its shop price is "written on each card"; the currency is never named.

Status effects:
- The rulebook names about 20 status effects (Chilled, Paralyzed, Enraged, Hardened, Softened, Smouldering/Smoldering, Burning, Blazing, Frostbite, Frozen, Sickness, Plague, Poison, Regeneration, Peppy, Withering, Stuffed, Engorged, Suffocation, Afraid, Blindness, Corruption). It defines their effects only through examples. The real definitions are the 64 status effect cards in `Auxiliary/Status Effects` (52 debuffs, 12 buffs).
- Spelling drifts between the pages and the cards: "Smouldering" on 19 vs "Smoldering" on 20 and the card; "Regeneration" on 19 vs the "Regen" card.

Contradictions between pages:
- 21 Damage Types NORMAL "DMG - (1/2*DEF)" vs 07/10 "ATK - DEF". It is an old-rules leftover; it agrees with 07/10 only if its DEF means the DEF bars shown.
- 12 Reactions3 says "5 stacks of hardened, giving it 3 defense". Hardened is +0.5 per stack, so 3 needs 6 stacks.
- 07 Stats3 text "-1 DEF BAR --> 1" vs its picture, which shows 1.5.
- 19 says damage and heal status effects proc at (ENDT). 09 puts "Status Effects deal damage/heal, duration ticks down by 1" in Phase 4, after Phase 3's (ENDT) effects.
- Hidden PSD layers carry older drafts that disagree with the visible pages: 12 Reactions2 has resolve order 4>2>1>3 vs 4>2>3>1, and 22 Negation has a different ITEMS block. That text is still in rules.json, stats.pkl and /query_rulebook.

Rules the book implies but never states:
- How many creatures can be in hand, when a creature may be summoned from hand besides filling an empty slot, and what happens when the hand runs out. 16 says "every slot must be filled if possible".
- The win condition: "Defeat all of your opponent's creatures". Whether creatures still in hand count is implicit. 15 says a persistent minion alone in a slot "must be killed for the game to be won".
- Lane tie-breaks: "the creature with the highest STAR LEVEL" is ambiguous at lane level. Within a lane, health lost is summed (08 example: Lane 1 "missing 3 HP" = 2 + 1).
- Whether combo damage multiplies sneak attacks or ability damage. How "25% of their attack stat +1" groups and rounds.
- Unspent star points are lost: they are "set to 7" each global turn and never carried over. Points and BRICKS carry over.
- Multiple actions: "one ACTION" (09) vs "2 innate actions" (11 Abilities7) and the Inspired buff (+1 Action). Who acts in a slot: 15 says every minion and parent in the primary field acts, in the player's chosen order.
- "5+ star creatures are immune to negation" implies creatures above 5 stars. The decks top out at 5★, but icon stars for 6/7/8 exist (`Types/Stars/6star.png` ...).
- "COUNTER" names both an item type (14) and a damage type (21, CDMG). They are related (counter items react) but are not the same rule.
- Swapping creatures: 2 points at the start of a slot's turn (08), 1 less for WIND (20). Swapping at other times, and status effect ticking for creatures in hand, appear only in 19's last bullet.
