# Impossibility Simulator rulebook glossary

Source: the website chapters `TTSCardMaker/site/rulebook/NN-slug.md` (24 files, read 2026-11-01 at `c6b1e144`). They are the only rule text and they win over this file. The Photoshop pages are history in `MDW/Old Rules/`. Citations are chapter numbers, with the section where it helps ("11 Multi-turn"). Text in quotes is verbatim, spelling included; the rest summarises a longer rule. The last section lists terms the chapters use but never define.

## 1. Abbreviations (ch 02)

| Abbreviation | Meaning (verbatim) | Group in the chapter | Also explained in |
|---|---|---|---|
| X | "“X” is a variable that can represent any number!" | header | 23 |
| SP | Star Points | GENERAL | 04, 13 |
| DMG | Damage | GENERAL | 10 |
| HP | Health stat | GENERAL | 06, 07 |
| ATK | Attack stat | GENERAL | 07 |
| DEF | Defense stat | GENERAL | 07 |
| SPD | Speed stat | GENERAL | 07 |
| PF | Primary field | GENERAL | 16 |
| CDMG | Counter damage | GENERAL | 21 Counter |
| (OYT) | On your turn | TIME FRAME | 11 When abilities can be used |
| (AAT) | At any time | TIME FRAME | 11 When abilities can be used |
| (XCD) | "Cooldown: needs X turn(s) of charge." 🕐, implies (OYT) | 🕐ON YOUR TURN | 11 Basic abilities: "the most common ability type"; 23 |
| (Y/XCD) | "Starts with Y turn(s) of charge." 🕐, implies (OYT) | 🕐ON YOUR TURN | 11 Multi-turn; 23 |
| (XGCD) | "Global cooldown: needs X turn(s) of charge." 🕐, implies (AAT) | 🕐AT ANY TIME | 11 Basic abilities; 23 |
| (Y/XGCD) | "Starts with Y turn(s) of charge." 🕐, implies (AAT) | 🕐AT ANY TIME | 11 Multi-turn; 23 |
| (BOT) | Beginning of turn | 🕐FORCED ("must happen automatically") | 11 Automatic, 09 Phase 1 |
| (ENDT) | End of turn | 🕐FORCED | 11 Automatic, 09 Phase 3 |
| (BOGT) | Beginning of global turn | 🕐FORCED | 11 Automatic: "happen before anything else in the global turn" |
| (ENDGT) | End of global turn | 🕐FORCED | 11 Automatic |
| (OFS) | On first summon | FORCED | 11 Life and death: cannot be negated or true negated, counts as (1PG); 22: true negation immunity |
| (OS) | On summon | FORCED | 11 Life and death: "can happen an unlimited number of times" |
| (OD) | On death | FORCED | 11 Life and death ("also forced"), 17 |
| (WOF) | While on field | FORCED | 11 Passives: often written only when the ability affects other creatures, to show it ends when the creature exits the field |
| (XPG) | "Ability can only be activated X time(s) per game." | PER GAME | 11 Basic abilities: "always either (AAT) or (OYT)" |
| (XU) | "X use(s) per full charge." | USES | 11 Basic abilities; 23 |
| (Action) | "Costs the creature's action" | colour note (grey) | 23 Other tags |
| ⏳ | "(XCD), (Y/XCD), (XGCD) and (Y/XGCD) abilities with X of 2 or more are considered "multi turn" abilities." | footer | 11 Multi-turn |
| 🕐 | "Abilites marked with 🕐 are considered "time based"." | footer | 11 Multi-turn tip |
| usable again | "Time based abilities with a cooldown become usable again once they are fully charged." | footer | 11 Basic abilities |

X is replaced by the number on cards: (1CD), (2GCD), (1/2CD), (2U), (3PG). In a pair "the first number is the charge the ability starts with: it can also be 0 or −1" (23). Ch 23's tables are also the tooltips: (1/2CD) "Cooldown: starts with 1 of 2 turns of charge", (-1/1GCD) "Global cooldown: starts with -1 of 1 turn of charge". Badge colours (02, 23): blue on your turn, purple at any time, red forced, orange per game and triggers, grey (Action).

Old tags appear only in ch 23's OLD SYSTEM table: (1PT) now (1CD), (E2T) now (2CD), (EXT) now (XCD), (2PT) now (1CD)(2U), (XPT) now (1CD)(XU), (1PGT) now (1GCD), (E2GT) now (2GCD), (EXGT) now (XGCD), (2PGT) now (1GCD)(2U), (XPGT) now (1GCD)(XU), (FTE2T) now (1/2CD), FGTE2GT now (1/2GCD), 1PT except first turn now (-1/1CD).

## 2. Defined terms, by chapter

| Term | Definition | Chapter |
|---|---|---|
| sleepy Kirby | "If you see sleepy Kirby, that means the textbox or card is explaining more technical rules." | 00 |
| to win | "Defeat all of your opponent's creatures, while making sure they don't defeat all of yours!" | 01 |
| creatures | "This game lives and breathes by them." Players use their creatures and items "to attack and protect" | 01 |
| starting hand | one 5★, two 4★, three 3★, one 2★ creature; two 4★, three 3★, three 2★, two 1★ items | 03 |
| starting resources | "Each team starts with 2 points, and 7 star points." | 03; 04 |
| facedown items | 3 items, "placed one behind each slot"; "Only your side can see your "Facedown Items" while they're in the holders"; "can be used at any time" | 03; 04; 13 |
| creature decks | "5 star Creatures are in the leftmost deck, 4 Stars are next, then 3 Stars, then 2 Stars." | 04 |
| item decks | 4★ leftmost, then 3★, 2★, 1★ | 04 |
| field card decks | 4★ leftmost, then 3★, 2★, 1★, for the field card shop | 04 |
| N.M.E. items | "N.M.E. Items for the N.M.E. Shop"; "N.M.E items are not in the standard deck" | 04; 05 |
| developer deck | "a reduced version of the full deck, meant to only included newer or recently changed cards that require more playtesting" | 04 |
| baseline creatures | "“Baseline” creatures that we mix into the Developer version. These cards are considered balanced" | 04 |
| global turn counter | "Starts at 1, and ticks up by 1 once every filled slot on the field has taken a turn." | 04 |
| star point counter / star points (SP) | set to 7 at the beginning of each global turn; "spent in order to play items. Items cost their star value in star points" | 04; 08; 13 |
| BRICKS | "used to buy Field cards"; obtained by mining (BRICKS = star level) and 1 per team at the end of every global turn | 04; 08; 09; 18 |
| points | "Each team starts the game with 2 points. You gain points equal to your creature's star level when it is defeated. Points are spent on abilities, and buying items from the item shop" | 04; 17 |
| lane | "A LANE on the PLAYING FIELD. There are three lanes. Your creatures occupy one side of the lanes, while your opponent's creatures occupy the other." A place on the table; turns are pair turns (08) | 04 |
| slot | "When occupied by a creature, this takes a turn." 6 slots, 3 per team | 04; 16 |
| status effect markers | markers for status effects on a slot | 04 |
| graveyard | "Your defeated creatures go in the graveyard on your side of the playing field." | 04; 17 |
| discard pile | "Used items go here." | 04; 13 |
| field card slots | where field cards are played | 04; 18 |
| auxiliary items deck | "Items summoned by specific abilities. These items are not in the main item pool" | 04 |
| auxiliary minions deck | "Where all the minions summoned by specific abilities are drawn from." | 04 |
| minion card holder | holds a summoned minion "to keep track of summoned minion stats" | 04 |
| Redirection cube | "Use this to determine RNG." (a die) | 04 |
| Order Determinator | "Use this to determine RNG if the odds are 1 in 2." (a coin) | 04 |
| 0.5 chip | "incase a counter uses half integers" | 04 |
| counters | "used to represent the duration of status effects, or for ability counters" (also the HP/DEF/ATK/SPD counters beside a slot, 06) | 04; 06 |
| health/defense/attack/speed markers | "indicate when a creature has a stat up or down. (Flip them over for stat downs)"; one marker = one unit of real value (1 DEF marker = 0.5 DEF, 1 HP marker = 1 HP) | 04; 07 |
| active item holder | holds an active item "to keep track of its stats" | 04 |
| simple version | decks for new players with "More complex creatures an items" removed; no field cards or N.M.E. items | 04 |
| copy of the deck | for accidental deletion, discussion, "or for certain abilities that require you to draw from a copy of the deck" | 04; 13 |
| creature type tokens | a bag of tokens "to signify if a creature type changes" | 04 |
| item shop | 2 items per star level; "Items in the shop cost 4x their star level in points"; reshuffle once per global turn for 1 point; restocked with the same star level | 05 |
| field shop | 2 field cards per star level, 1 per level revealed at start, the other 4 at the end of global turn 1; "Field Cards cost 2x their star level in BRICKS"; "does not restock" | 05 |
| N.M.E shop | 4 random N.M.E. items, restocked at random; "purchase cost is written on each card" | 05 |
| types (creature) | "Special markers that give creatures certian effects." | 06 |
| star level | "A higher star level means the creature is more powerful."; for items, "how much the item costs to play in star points" | 06 |
| stat bars | "Max Health, Defense, Attack, and Speed bars." "The health value is double of what is shown." "The defense value is half of what is shown." | 06; 07 |
| abilities | "Text explaining what a creature is capable of doing while they are on the field."; "the main chunk of text at the bottom half of a creature" | 06; 11 |
| HEALTH | "A creature perishes when its health reaches zero." Each bar = 2 HP | 06; 07 |
| DEFENSE | "Subtracted from any attack damage this creature takes." Each bar = 0.5 DEF | 06; 07 |
| ATTACK | "The amount of damage this creature deals to opponents." Each bar = 1 ATK | 06; 07 |
| SPEED | "Determines the order that creatures take their turn in"; "determines turn order and combo damage." Each bar = 1 SPD | 06; 07 |
| effects (item) | "Text explaining the effect the item has when it is played." | 06 |
| MAX HP | "the HP a creature starts with. They cannot be healed above this amount naturally." | 07 Health |
| MAX HP increase | "increases their current HP by the same amount (This counts as healing)" | 07 Health |
| instance of healing | two healing effects from one item or ability "is counted as one instance of healing" | 07 Health |
| MAX HP decrease | "does not decrease their current HP, unless if their current HP would become greater than their Max HP (in which case it is set to Max HP). If its current HP was already greater than its Max HP, the decrease lowers its current HP by the same amount."; "does not count as damage" | 07 Health reduction |
| base stats | "A creature's base stats are the stats printed on its card, before any changes. Stat changes affect its current stats, not its base stats, unless an effect says it changes a base stat." | 07 Rounding + stat changes |
| rounding (stats) | "Stats are rounded to the nearest half in the positive direction, after reductions or gains are applied." | 07 Rounding + stat changes |
| additive / multiplicative | "Additives are applied first and multiplicatives second."; "Multiplicative effects are applied additively" | 07 Rounding + stat changes |
| stat floors | "SPD can't go below 1. ATK and DEF can't go below 0. MAX HP can't be reduced below 1 bar (2 HP)." | 07 Rounding + stat changes |
| GLOBAL TURN | "When every slot has taken its turn once, that is called a GLOBAL TURN." | 08 |
| pair turn | "Turns are taken in pair turns: a slot from one team, then a slot from the opposing team." Replaced lane turns | 08 Turn order |
| turn order | fastest creature, then the opposing team's fastest, then the next fastest and its opposite, then the fastest remaining and its opposite. "Only the PRIMARY FIELD counts. A slot's speed is its fastest creature's." "MOST SPEED > LEAST HEALTH LOST > HIGHEST STAR LEVEL > RNG" | 08 Turn order |
| HEALTH LOST | "(MAX HEALTH - CURRENT HEALTH)" | 08 Turn order |
| turn manipulation | speed changes by (AAT) abilities and facedown items; "the creature may swap out for 2 points. This cannot change turn order." | 08 |
| no stalling | "During a full global turn, if none of a team's creatures attack, those creatures take 1 TRUE DAMAGE at the end of the global turn." | 08 |
| ACTION | "Every creature gets one ACTION when their turn starts." Used to ATTACK, MINE or go into DEFENSE MODE. "Actions can only be used (OYT)." | 09 |
| MINE / MINING | "generating BRICKS equal to its STAR LEVEL" (prime creatures only; "Minions cannot mine for BRICKS.") | 09; 15; 18 |
| DEFENSE MODE | "doubling its defense while it is in this state. It goes out of defense mode at the beginning of its next turn." Shown by turning the creature 90 degrees | 09 |
| turn phases | Phase 1 (BOT); Phase 2 main (OYT abilities, items, facedown swaps, shops, actions, field swap); Phase 3 (ENDT); Phase 4 status effects deal damage/heal and tick | 09 |
| attacks | "occur when creatures use an action to perform an attack, when an attack item is used, or an ability deals damage (DMG) separate from the ATK stat"; "Attacks deal ATK - DEF in damage."; hit the PRIMARY FIELD by default; "Creatures with 0 ATK cannot declare an attack (of any type)." | 10 |
| retaliation limits | "Damage from abilities and item effects cannot trigger counter damage as retaliation. Attacks from abilities cannot trigger counter items." | 10; 21 |
| COMBO DAMAGE | twice the target's speed: x1.5; triple: x2 | 10 |
| sneak attack | "an attack that deals piercing damage equal to 25% of their attack stat, plus 1, and has the same on-hit effects as a normal attack"; "Minion sneak attacks are only 25% of their attack stat, without the +1." | 10 |
| multi-hit attacks | "Multi-hit attacks do not count as separate attacks. Each instance of damage cannot be reacted to on its own, only the entire attack can." | 10 Multi-hit attacks |
| attacking several creatures at once | by ability or item: "the attacks happen left-to-right from the attacker's perspective and on death effects happen after all instances of the attack are resolved." | 10 Multi-hit attacks |
| Redirections | "change the attack's target to a creature of choice" | 10 Redirections |
| Interceptions | "change the target of an attack to be the creature which is intercepting"; cancels defense mode | 10 Redirections |
| inflictions | debuffs or stat changes inflicted on offense or defense apply before the attack strikes; attack inflictions cannot be separately Negated/Reflected/Redirected/Intercepted | 10 Inflictions |
| “on attack” effects | "Only attack actions can proc any “on attack” effects." | 10 Inflictions |
| rounding (damage) | "damage rounds up, for example 3.25 DMG rounds to 3.5 DMG" | 10 Rounding |
| basic abilities | the (XCD)/(XGCD)/(XPG) family | 11 Basic abilities |
| charge | (XCD): "X is how many turns of charge the ability needs to be used (it can't go over X). Abilities start with no charge. 1 turn of charge is gained at any point during phases 1-3 of the creature's turn, if the ability hasn't gained charge yet that turn and hasn't been fully charged at any point during that turn." (XGCD): "gains charge at any point during the global turn, under the same conditions." (Y/XCD): "Y is the charge it starts with." Negated cooldown abilities "still gain charge" (22) | 11 Basic abilities, Multi-turn |
| uses (XU) | "how many uses the ability has when charged. If even one use is spent the ability can gain charge again; charged again without spending every use, it only goes back up to X uses. No (XU): 1 use per full charge." | 11 Basic abilities |
| automatic abilities | "non-optional and assumed to happen automatically" ((BOGT), (BOT), (ENDT)) | 11 Automatic |
| multi-turn | (2CD) "needs 2 turns of charge"; a forgotten ability: "it's saved so you can use it on the next turn, but its cooldown doesn't reset until it's used!"; "Abilties do not count time passing while in the hand." | 11 Multi-turn |
| life and death | (OFS)/(OS)/(OD) | 11 Life and death |
| SAME TIME | simultaneous effects: higher SPD creature first, then active items of the team with the highest cumulative SPD, then field cards; ties use the turn order tie-breakers; the owner orders own-side conflicts | 11 Same time |
| CONFLICTION | both teams setting the same value: "it remains random"; simultaneous revives that set HP add together (up to Max); directly conflicting effects: player chooses | 11 Confliction |
| ability copying/stealing/swapping | "Stolen/Swapped abilities start as brand new per creature, if the creature loses that ability then gains it back, it's not reset but instead gains 1 turn of charge."; "The first time a creature gains an on first summon ability (or if this is the first opportunity that an OFS ability can be procced), proc it now." | 11 |
| healing items dealing DMG | "If a healing item is used to deal DMG, it is instead treated as an attack item, and vice versa."; "If a healing item deals DMG instead of healing (such as with Chef Saltbaker), effects that increase healing (such as Toy Chica) increase that damage." | 11 Misc |
| passive abilities | "abilites that happen automatically, without the player actively choosing to use it"; most are (WOF) | 11 Passives |
| reaction | "When a play occours, a player may choose to react to it with one of their items or abilities. When you react to something, the reaction happens before whatever you reacted to." | 12 |
| reaction chain / resolve order | plays resolve in reverse; "anything that deals damage resolves last"; "Whatever was originally being reacted to within a reaction chain will always be resolved at the very end." | 12 |
| SPENDING | star points are spent when played; refunds when the target becomes invalid, or when the target dies before an attack resolves | 12 Spending |
| NO TAKE BACKS | "A decision to attack or use an ability/item cannot be retracted if your opponent reacts to it in some way." | 12 |
| targeted for negation | "If something is targeted for negation (or destruction, in the case of field cards), its owner cannot reactively activate any (AAT) effects on that card." | 12 Clarification; 22 |
| items | "your main resource besides creatures"; cost star points equal to star level; discarded after resolving unless multi-use; "You cannot use items on creatures in the hand or graveyard. You can still unequip items from creatures in the hand." | 13 |
| multi-use item | "goes back facedown or to the hand" after being played | 13 |
| selling items | "During your turn you can sell items for points equal to their star level. You can also sell items that are facedown at any time."; not if previously used | 13 |
| random item draw | draw from a copy of the full deck until an eligible item is found | 13 |
| item types | "the symbol are the top left of an item that dictate how the items work. These items types cannot be manipulated." | 14 |
| HEALING ("food") | "these items heal creatures. May not be used on [machine]." | 14 |
| ATTACK (item) | "These items do damage, and count as an attack when used." | 14 |
| NEGATION (item) | "These items “Negate” something, taking it out of play." | 14 |
| SUPPORT | "general use items with various effects" | 14 |
| COUNTER (item) | "can only be activated under certain conditions. These can only be used reactively."; discarded when all uses are spent | 14 |
| INFLICTION (item) | "These items inflict a status effect and count as an infliction." | 14 |
| EQUIP | equipped to your own creatures for a passive effect; "Creatures do not inherit the abilities of equip items, they are only affected by them." A creature's innate abilities and its equip's are separate; negating a creature does not negate its equipped items (22); unequip on your turn; negative-effect equips are locked until the next global turn; returned to hand on death | 14 |
| forcibly unequipped | "An equip can be “Forcibly” unequipped by an effect at any time" | 14 |
| FRAGILE EQUIP | "same as equips but cannot be unequipped by any means and are discarded when the creature they are on dies" | 14 |
| ACTIVE | "placed in a slot and stay there on the field with a constant effect until they are destroyed, whether or not they affect that slot. They are considered on the primary field. They have health and defense, although they are not creatures" | 14; 16 |
| DEPLOY | "stay on a chosen slot (doesn't count as being used on the creature in that slot) with a constant effect. They cannot be attacked" | 14 |
| minions | "small creatures that when summoned are attached onto a parent prime creature"; "considered creatures" for most purposes; "they do not yield points when defeated. They do go to the graveyard when defeated."; "5 star minions cannot be targeted by minion effects." | 15 |
| summon (bare word) | "If something just says “summon”, it means onto the creature that the ability is from." | 15 |
| parent creature | the prime creature minions attach to | 15 |
| parent creature | the creature a minion is on, not always its summoner; minions are tied to it; defeated, its minions die unless persistent | 15; 22 |
| offensive minions | "occupy the same field as their parent creature" | 15; 16 |
| protective minions | occupy the field the parent is not in; push the parent "back to the secondary field" | 15; 16 |
| persistent (minion) | survives the parent's defeat and attaches to the next prime creature in the slot; alone in an empty slot it "must be killed for the game to be won by the opponent" | 15 |
| the field | "The place that creatures are placed on is called the field, which is organized into 6 slots, 3 for each team." | 16 |
| prime creature | "Prime creatures are the creatures that fill the slots: every creature that isn't a minion. A card with several forms or made of multiple creatures (like Mantis Lords) is one prime creature. Prime creatures yield points when defeated." | 16 |
| PRIMARY FIELD (PF) | "the only field opponents can target by default, and the only field creatures can act in"; "everything in the Primary Field" = all 3 slots; "Active items are on the primary field, so something that hits everything on the PF hits them too. Something that hits all enemy creatures on the PF does not." | 16 |
| SECONDARY FIELD | "Behind the primary field"; created when a protective minion or protective active item pushes a creature back; "By default this field cannot be targeted by opponents."; abilities only, no actions | 16 |
| protectors | "All protectors in a field together, which is the field that the parent creature is not in" | 16 |
| death (dies) | "When a friendly creature's HP reaches 0 it dies: its “On Death” effects proc (does not apply to creatures not on the field), it loses all status effects and all of its items are unequipped."; "If something revives it on the spot (such as a Life Shroom, or an (OD) ability that revives it or switches it to its next form), it is not defeated." | 17 On death |
| defeat (defeated) | "A creature that dies and isn't revived on the spot is defeated, and goes to the graveyard on your side of the field."; "When a prime creature is defeated: your team gains points equal to that creature's star level, all non-persistent minions in that slot die, and the slot is empty so a new prime creature can be placed in it."; "Outside of “On Death” effects, anything that says a prime creature “dies” means it is defeated." | 17 Defeat |
| revives | "Creatures that are revived from “On Death” effects only yield Points when they are defeated."; revived by other means: "Points are kept from it being defeated, but defeating the revived version awards 0 Points."; "its HP and all of its stats, abilities, charge and counters are reset to their base, unless otherwise specified." | 17 Revives |
| destroyed | "A creature being “destroyed” instantly by any effect takes true damage equal to its current HP." | 17 Specifics |
| field card | "cards that when on the playing field have a constant effect"; bought with BRICKS; "You may have only 1 of these out at a time."; "Outside of using star points to play them from the hand, field cards do not behave like items in any way. Anything meant for items cannot be used for field cards (such as Stopwatch or facedown slots)." Written "field item" in the ban list and the TTS save until 2026-11-01 | 18 |
| counter-play (field cards) | negate for 1x star level in BRICKS until the end of the global turn; destroy for 3x | 18; 22 |
| status effect | "either a buff or debuff that has an effect on a creature, represented by Mr. Krabs either suffering or thriving" | 19 |
| buff / debuff | the two kinds of status effect | 19 |
| status timing | DMG/heal effects proc in phase 4, after (ENDT) effects, others immediately; durations count down in phase 4; DMG from status effects is piercing and not an attack | 19 |
| unique stack instance | "DMG dealing status effects count as one instance of damage per unique stack." | 19 |
| conflicting debuffs | "If multiple debuffs have conflicting effects, the effect of the debuff with the highest star level takes precedence." | 19 |
| status syntax | ""[X]T[Y] [Status]", where X is duration and Y is stacks"; "5T1 Poison" = 5 turns of 1 stack | 19 Syntax |
| standard status effects | "can only have one instance of themselves applied to a single creature" | 19 Types |
| stacking status effects | "can have any number of themselves applied" | 19 Types |
| duration (stack/upgrade/reapply) | "set the duration to the longest lasting of the applications" | 19 Duration |
| upgradable | "Certain status effects upgrade themselves when applied": Smouldering→Burning→Blazing, Chilled→Frostbite→1 turn of Frozen, Sickness→Plague | 19 Upgradable |
| creature types | "Every type has a unique function, typically an immunity to a debuff." (14 types, see §2a) | 20 |
| damage types together | "Multiple damage types can be applied to an instance of damage, combining their effects." | 21 |
| NORMAL (damage) | "Reduced by the defense of the target." DMG - DEF | 21 |
| PIERCING | "Ignores DEF." | 21 |
| COUNTER (damage) | "Can only be dealt as retaliation to an attack, or by something that deals damage when it dies. Does not count as an attack, and can bypass abilities and protective minions as a result." The "when it dies" part is the owner's ruling of 2026-11-01 | 21 |
| LIFE STEAL | "DMG dealt after DEF is healed before the attack resolves, with a percent scalar where 100% is 1:1, 50% is 1:2, 200% is 2:1" | 21 |
| SHOCKWAVE | "Deals damage to all enemy targets on the primary field aside from the original target, and also the secondary field slot behind the main target (This does not count as an attack)." | 21 |
| TRUE DAMAGE | "Ignores DEF, and cannot be negated, redirected, or reduced by any means." | 21 |
| OVERKILL | "Only applies when a card says so. Damage beyond what kills the target carries over to the next target the card names, and is reduced by that target's DEF. This does not count as an attack, and other effects of the attack (such as status effects) do not proc on overkill."; "Abilities that count damage dealt to creatures do not count overkill damage." (11 Misc) | 21; 11 |
| negated attack | "it does not happen. No damage is dealt, no on-hit effects occur, the creature does not get its action back" | 22 |
| negated creature | "unable to use any of its own abilities (unless otherwise stated)", but "may still use the abilities of any equipped items or minions, and can still use actions" | 22 |
| negation immunity | "5+ star creatures are immune to negation" | 22 |
| True Negation | "Negation that can negate negation-immune creatures." | 22 |
| True Negation Immunity | "Complete immunity to any possible sources of negation, including true negation." (OFS abilities and "upon being negated" effects have it) | 22 |
| negated item | "cannot be played or is taken out of play, and it returns to where it was before being played"; star points are not regained; cannot be sold while negated; equip/active/deploy items go back to hand | 22 |
| timing tags | every tag's meaning, the website's tooltips, the old tags' conversions | 23 |

### 2a. Creature types (ch 20)

| Type (icon file) | Effect |
|---|---|
| MACHINE (machine) | Cannot have food items be used on them. Immune to Sickness, Plague, Poison |
| WIND (Wind; PSD layer "tornado") | Can swap out/in for 1 less point |
| ROCK (rock) | +1 Brick when mining for Bricks |
| PANDA (panda) | Immune to Stuffed, Engorged |
| FIRE (fire) | Immune to Smoldering, Burning, Blazing |
| ICE (ice) | Immune to Chilled, Frostbite, Frozen |
| WATER (water) | Immune to Suffocation |
| ELECTRIC (electric) | Immune to Paralyzed |
| UNDEAD (undead) | Immune to Sickness, Plague |
| DARK (dark) | Immune to Afraid |
| LIGHT (light) | Immune to Blindness |
| DARK MATTER (darkmatter) | Immune to Corruption |
| UNKNOWN (unknown) | Once per game on your turn may permanently change this type into other creature type. |
| GIRL (Girl) | Gain 1 additional point (OD). "This may only be an innate type in special cases, but may still be added to creatures through effects." |

### 2b. Item type icons (ch 14, 15, 18, 19)

HEALING = Food (tomato), ATTACK = Attack (bomb), NEGATION = Negation (metronome), SUPPORT = support (blue ring), COUNTER = Counter (spiky grass), INFLICTION = Infliction (burning Mr. Krabs), EQUIP = Equip (sword), FRAGILE EQUIP = Fragile Equip (sword + clock), ACTIVE = active (radio tower), DEPLOY = Deploy (Peashooter). Minion icons: offensive = Offensive_Minion, protective = Defensive_Minion (green shell). Status icons: buff, debuff, buffstack, debuffstack, debuffupgrade. Field cards: field (tree).

### 2c. Status effects named in the chapters

The chapters name these but do not define them, except as noted. The effect text is from the status effect cards (`Auxiliary/Status Effects`, db/stats.json, checked 2026-11-01). It is supporting data, not rulebook text.

| Status effect | Kind, stars (card) | Card text | Named in |
|---|---|---|---|
| Chilled | debuff (upgradable), 2★ | -25% SPD can't affect [ice] | 07 ("-25% SPD"), 19, 20 |
| Paralyzed | debuff, 2★ | -50% SPD can't affect [electric] | 07 ("-50% SPD"), 20 |
| Enraged | buff (stacking) | +25% ATK | 07 |
| Hardened | buff (stacking) | +0.5 DEF | 12 ("6 Hardened (3 DEF)") |
| Softened | debuff (stacking), 1★ | -0.5 DEF | 12, 19 ("Softened (-0.5 DEF)") |
| Smouldering / Smoldering | debuff (upgradable), 1★ | 2 DMG, can't affect [fire] | 19 ("Smouldering (2 DMG)"), 20 |
| Burning | debuff (upgradable), 1★ | 3 DMG, can't affect [fire] | 11, 19, 20 |
| Blazing | debuff, 2★ | 4 DMG, can't affect [fire] | 19, 20 |
| Frostbite | debuff (upgradable), 2★ | 2 DMG. -25% SPD, can't affect [ice] | 19, 20 |
| Frozen | debuff, 3★ | Can't affect [ice]. Lose all actions, cannot switch out. | 19, 20 |
| Sickness | debuff (upgradable), 1★ | -0.5 DEF -1 ATK -1 SPD can't affect [undead] [machine] | 19, 20 |
| Plague | debuff, 2★ | -1 DEF -1 ATK -2 SPD Can't affect [machine] [undead] | 19, 20 |
| Poison | debuff (stacking), 1★ | 1 DMG, can't affect [machine] | 11, 19 ("5 stacks of Poison"), 20 |
| Regeneration (card "Regen") | buff (stacking) | Heal 1 HP | 19 |
| Peppy | buff (stacking) | +1 SPD | 19 (card picture) |
| Withering | debuff, 4★ | (BOT) 2 DMG -0.5 DEF -1 ATK -1 SPD. | 19 (card picture) |
| Stuffed | debuff (stacking), 1★ | -1 SPD, can't affect [panda] | 20 |
| Engorged | debuff (stacking), 2★ | -25% SPD, can't affect [panda] | 20 |
| Suffocation | debuff (stacking), 3★ | 1 DMG Can't Effect [water] | 20 |
| Afraid | debuff, 2★ | -2 ATK -2 SPD Can't affect [dark] | 20 |
| Blindness | debuff, 1★ | Attacks hit a random enemy creature can't affect [light] | 20 |
| Corruption | debuff, 3★ | Can't affect [darkmatter]. If anything occurs in a way that would resolve differently than if this creature's ability was negated, take 4 DMG. Can only activate (1GCD). | 20 |

## 3. Used but never defined

| Term | Where used | Meaning from context (inference) |
|---|---|---|
| lane turn, a lane going (on cards) | Delicate Flower "after 3 lane turns have ended"; The Last Cabbage Merchant "each time a lane turn ends"; Time Riddle "until 2 lanes have gone"; in the same sense Asteroid Field "Every time a lane ends its turn", Forget Me Now "after the current lane ends", NME Salesman "At least 3 lanes' turns must end", Lost Kin "go first in its lane" | older card wording for pair turns; the chapters no longer have lane turns. "Lane" alone as a place (same lane, adjacent lanes) is still ch 04's |
| recharge, usages (on cards) | Michael "(2GCD) Recharge 1 turn of charge to another creature's ability."; Energy Ritual "fully recharges and restores all usages"; The Bank | the cards' words for gaining charge (11) and remaining uses |
| refresh (on cards) | Cave Of The Past "All 🕐time-based effects and abilities do not refresh."; Tainted Bethany "Refresh the first ability."; Hot Spring, The Kestrel, The Maverick, Warcraftian Druid | older word for becoming usable again. Ch 02 now says "fully charged"; of refresh only the status effect sense (ch 19) remains in the chapters |
| time based (on cards) | Time Riddle (cooldown tags as examples), Energy Ritual "time-based and (XPG) abilities", Cave Of The Past | the 🕐 tags of ch 02 |
| cooldown (of an equip) | 14 "regardless of if the item has negative effects or is on cooldown"; Stalwart Shell "Unequipping this item resets its cooldown" | ch 11 gives abilities a cooldown; what an equip's is goes unsaid |
| tier (of a debuff) | 11 ("cure a tier 2 debuff") | the debuff card's star level |
| proc | 10, 11, 17, 19, 21 | to trigger or activate |
| instance | 07, 10, 19, 21 | one application of damage or healing |
| on-hit effects | 10, 12, 22 | effects applied by an attack (see "on attack") |
| innate actions / innate type | 11, 20 | the actions or types a card has on its own |
| cleanse, refresh (status) | 19 ("need to upgrade to refresh duration"), card texts | remove / reset duration |
| N.M.E. | 04, 05 | never expanded |
| “Don't be stupid” rule | 01 | named, never stated |
| swap out / swap in | 08, 14, 15, 20 (WIND) | return a creature to hand and summon another into its slot; cost 2 points (08) |
