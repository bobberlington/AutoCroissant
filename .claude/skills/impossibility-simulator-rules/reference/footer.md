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
- Card texts use timing tags the rulebook never defines. Since the cooldown update of 2026-10-21 the cards print (nCD) cooldowns (159 uses on 138 cards, e.g. "(2CD) Attacks inflict 2T Afraid."), (nGCD) global cooldowns (106 on 93), the pairs (1/2CD) (15 creatures) and (-1/1GCD) (once), and (nU) uses (24 on 22, e.g. "(1CD)(2U) Draw one ingredient:"), in place of (XPT), (XPGT), (EXT), (FTEXT) and (EXGT); counts from the parse of 2026-10-22. The pages above still define only the old tags. Card text suggests an n-turn cooldown, an n-global-turn cooldown and n uses per cooldown; that reading is an inference (SKILL.md §3 lists the card evidence). Old Rules defined "(CD) = Counter damage", which today is CDMG.
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
