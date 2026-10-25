# Impossibility Simulator: data facts (extracted 2026-10-20, re-extracted 2026-10-22)

Sources, all read-only. Card repo: TTSCardMaker clone at `73904dbd` (2026-10-21 23:57 -0700, then equal to origin/main),
after the cooldown update `df02b3b6`..`7cb4c867`. Fresh parse: a sandboxed forced sweep of that HEAD with the bot's own
parser (AutoCroissant origin/main `39da9ed`, parser v2) → `stats_now.pkl`, and from it `cards_now.json` with
build_site.py's own card loop (the shape of `site/dist/data/cards.json`). Before: AutoCroissant `stats.pkl` at PICKLE
`56b43b9` (2026-10-21 05:30, parser v1) → `cards_before.json` the same way. Both live in `/tmp/rules_refresh` (scratch,
gone on reboot; recipes in SKILL.md Provenance). Also `site/dist/data/rules.json` (rebuilt 2026-10-22 00:43, same 48 page texts);
`MDW/Fonts/stoogedings-src/stoogedings.json` and `build_stoogedings.py`; `site/assets/deck.js`, `text.js`, `app.js`,
`search.js`; `site/Constructed Ban List.txt`; `TTS/Impossibility Simulator.json` and `TTS/decks.json`, audited with a
copy of `TTS/audit.py` (removed 2026-10-25; `git show 843a4a99:TTS/audit.py`) that writes to /tmp. Extraction scripts: `.claude/skills/impossibility-simulator-rules/scripts/`
(rows.py, counts.py, agg.py, tags.py, lines.py, mentions.py; each takes cards.json or the card data's db/stats.json;
a loadpkl.py dump of stats.pkl until 2026-10-25).
Since then the clone's HEAD moved to `2291b5de` (a site-only commit) and origin/main to `902e01e3` (3 commits that do
not include it); no PSD changed after `73904dbd` and the 48 page texts are identical, so the data below still holds.
AutoCroissant PICKLE `3888392` (2026-10-22 00:23) pushed this parse with the new tags and printed titles: its `stats.pkl`
matches `stats_now.pkl` exactly (819 entries, no ability, display_name or stars difference), and the site's
`cards.json` rebuilt at 00:43 gives the same `tags.py` total (625 uses of 27 tags in 352 cards, no (1PT), (1PGT) or
(E2T)). Sections 1-3 and 5-6 re-checked on the fresh parse: unchanged unless noted.

Terms follow the rulebook: creatures, minions, items (item types), field cards, N.M.E. items, status effects
(buffs, debuffs), stars, types, abilities, timing tags, global turn, starting hand, shops, points, star points (SP), BRICKS.

## 0. Totals and how the three sources agree

| Source | Entries | Notes |
|---|---|---|
| PSDs in HEAD (outside MDW/ and Auxiliary/Markers/) | 813 | the sweep parsed all 813: 0 errors, 0 unknown types, 0 moved; no card added or removed by the update |
| stats.pkl (fresh sweep) | 819 | 813 PSDs + 6 copy rows from `site/Card Edits.json` duplicates (A Snack2, A Snack3, All Purpose Medical Kit2, NME Stock2, Regeneration Potion2, Watermelon2) |
| stats.pkl by card_type | creature 297, item 188, minion 91, debuff 52, field 50, aux item 48, rulebook 48, nme 26, buff 12, creatures 7 | item 188 = 182 PSDs + the 6 copy rows |
| cards.json `cards` | 773 rows | 767 distinct cards + 6 copy rows (`dup`); 771 distinct names. Rulebook pages are not in cards.json; they are in rules.json (48) |
| rules.json | 48 pages | text identical to the 48 `rulebook` entries in stats.pkl, before and after the update (checked page by page) |

PNG-only cards (no PSD, so not parsed, `parsed: false` on the site): `Creatures/Other/2 Stars/Stubborn_Queen.png`,
`Items/Support/2 Stars/Gust_of_Trees.png`, plus the six copy PNGs (`A_Snack2.png`, `A_Snack3.png`,
`All_Purpose_Medical_Kit2.png`, `NME_Stock2.png`, `Regeneration_Potion2.png`, `Watermelon2.png`).
Copies: A Snack has copies named Breakfast and Supper; Regeneration Potion has Grape Juice; Watermelon has
Inconspicuous Watermelon; All Purpose Medical Kit and NME Stock each have one copy with the original's printed title.
The 28 doubled `<id>-2` rows of the 2026-10-20 local build are gone; the 2026-10-21 builds have none.

Names. Parser v2 stores the printed title as `display_name`; the website shows it as `name` and keeps the file-derived
name as `key`. In the fresh parse 131 rows changed name (130 cards and 1 copy row), mostly case or punctuation
("FAWFUL", "Mr. Krabs", "Michael’s Blessing" with a curly apostrophe, "Doopliss", "Militron (no Armor)"), a few more:
Blockade → "The Wall By Ech", Leeroy Jenkins → "LEEEEEEEEEROY", Rasta Beast → "Pasta Beast", Sir Grodus → "Grodus",
The Chicken → "Chicken", Zenith Martlet → "Martlet Zenith of Monsterkind", Computer Virus → "Computer Virus appears!",
The Bagel → "The Bagel Or... is it". Counted against file names, the printed title differs for 159 cards (165 rows with
the copy rows): the 130 above plus 29 already renamed in `site/Card Edits.json` (the Clots, the Ingredients and others).
PICKLE `3888392` put them in the bot and the site. This file names cards by file name.

## 1. Card kinds, folders, counts

Counts are distinct PSDs (stats.pkl) unless noted. Stars for Creatures, Items, Field and Debuffs come from the leaf
folder name; for N.M.E, Auxiliary/Items and Auxiliary/Minions they are counted from the PSD's star layers.

### 1.1 Creatures (`Creatures/<Series folder>/<N Stars>/`, card_type `creature`): 297 PSDs (+1 PNG-only, Stubborn Queen)

By stars (stats.pkl, 297): 2★ 45, 3★ 103, 4★ 93, 5★ 56. There are no 1★ creature folders. 24 series folders.

| Series folder | Total | 2★ | 3★ | 4★ | 5★ |
|---|---|---|---|---|---|
| Kirby | 58 | 2 | 28 | 14 | 14 |
| Other | 48 | 8 | 13 | 11 | 16 |
| Mario | 28 | 10 | 10 | 7 | 1 |
| Hollow Knight | 24 | 2 | 7 | 11 | 4 |
| Omori | 20 | 1 | 9 | 6 | 4 |
| Pandas | 20 | 7 | 9 | 3 | 1 |
| World Of Warcraft | 13 | - | 1 | 11 | 1 |
| Fnaf | 12 | - | 5 | 4 | 3 |
| Isaac | 10 | 2 | 4 | 4 | - |
| Undertale | 10 | - | 3 | 4 | 3 |
| DST | 9 | 1 | 6 | 1 | 1 |
| Monkey D Folder | 9 | 3 | 3 | 3 | - |
| Everhood | 6 | 1 | 1 | 2 | 2 |
| FTL | 5 | 2 | - | 2 | 1 |
| In Sound Mind | 5 | 1 | 2 | 2 | - |
| Lisa | 4 | 1 | 1 | 2 | - |
| Souls | 4 | - | 1 | 2 | 1 |
| Crypt | 3 | 2 | - | 1 | - |
| BTD | 2 | - | - | 1 | 1 |
| Bug Fables | 2 | 1 | - | - | 1 |
| ROTMG | 2 | - | - | 1 | 1 |
| Cuphead | 1 | - | - | - | 1 |
| R.E.P.O | 1 | 1 | - | - | - |
| Hollow Knight Silksong | 1 | - | - | 1 | - |

The PNG-only Stubborn Queen (Other 2★) is not in stats.pkl; on the TTS table it sits in the Minions decks.
The `series` field (from card data, not the folder) has 59 distinct values on creatures (87 over all cards); top: Kirby
56, "Impossibility Simulator" 29, Mario 28, Hollow Knight 24, OMORI 20, World of Warcraft 13, Five Nights at Freddy's 12.
The Mario games became a `subseries` on 2026-10-20 (`15a16be2`; Paper Mario: The Thousand-Year Door 20), so the
2026-10-20 count of 66 is older than that split.

Creature types on the 297 creatures (distinct creatures per type; a card can have several): machine 50, light 40,
fire 27, dark 25, wind 25, panda 25, undead 23, rock 22, water 18, electric 15, unknown 10, darkmatter 10, ice 9, girl 1;
75 creatures have none. Pandora ['panda','panda'], Kyle ['unknown','unknown'] and Z!!!!! ['girl','girl'] list their type
twice in stats.pkl, so counting icon occurrences gives panda 26, unknown 11, girl 2 (cards.json: the same 25/10/1).

### 1.2 Items (`Items/<Item type folder>/<N Stars>/`, card_type `item`, `subtype` = folder lowercased): 182 PSDs

By stars (182): 1★ 24, 2★ 59, 3★ 69, 4★ 30. No 5★ items.

| Folder (subtype) | Total | 1★ | 2★ | 3★ | 4★ |
|---|---|---|---|---|---|
| Equip | 65 | 9 | 17 | 23 | 16 |
| Heal | 35 | 6 | 17 | 11 | 1 |
| Support | 30 | 3 | 11 | 12 | 4 |
| Attack | 16 | 2 | 5 | 7 | 2 |
| Infliction | 10 | 1 | 2 | 4 | 3 |
| Active | 9 | - | 2 | 5 | 2 |
| Counter | 9 | 2 | 1 | 5 | 1 |
| Negation | 5 | - | 2 | 2 | 1 |
| Deploy | 3 | 1 | 2 | - | - |

Plus in stats.pkl 6 copy rows (Heal 2★: A Snack2, A Snack3, All Purpose Medical Kit2, Regeneration Potion2,
Watermelon2; Support 2★: NME Stock2) = 188 `item` rows, and the PNG-only Gust of Trees (Support 2★).
Nine item folders; the rulebook's "14 Item Types" names ten item types: Healing ("also referred to as food"), Attack,
Negation, Support, Counter, Infliction (discarded after being played) and Equip, Fragile Equip, Active, Deploy (not
discarded immediately). Fragile Equip has no folder: it is a type icon on cards in Equip (and Auxiliary/Items).
Type icons on the 182 items: equip 55, food 36, support 30, attack 17, infliction 12, fragile equip 11, active 9,
counter 9, negation 5, deploy 3.

### 1.3 Field cards (`Field/<N Stars>/`, card_type `field`): 50
1★ 9, 2★ 18, 3★ 12, 4★ 11. Every one has the `field` type icon.

### 1.4 N.M.E. items (`N.M.E/`, card_type `nme`): 26
Stars counted from the PSD: 0★ 1 (Bubble, "NO STARS FOUND"), 1★ 8, 2★ 11, 3★ 4, 5★ 1 (Michael's Blessing), 7★ 1 (Sigil of Divinity).
1★: Double Pain, How To Jump, Magnet-shroom, Metronome Lite, Star Jar, Star Scraper, The Bagel, The Evil, Fricked Up Cursed Idol.
2★: Crock Pot, Electric Dart, Jera, Minion Dematerializer, Paintbrush, Remote Detonator, Rope, Staff of Purity, The Bank, Whacka Bump, chuckya.
3★: Black Hole, Duel Flag, Termination Shock, The Claw.
Type icons: nme 26 plus an item type on some (support 11, active 5, equip 4, food 3, deploy 2, attack 1, negation 1).
Rulebook 05 Shops: the N.M.E. shop shows 4 random N.M.E. items; their purchase cost is written on each card; N.M.E. items are not in the standard deck.

### 1.5 Auxiliary (`Auxiliary/...`)

| Folder | card_type | Count | Stars | Notes |
|---|---|---|---|---|
| `Auxiliary/Minions/` | `minion` | 91 | from PSD: 0★ 1 (God Tamer), 1★ 15, 2★ 31, 3★ 27, 4★ 11, 5★ 6 | types: offensive_minion 68, defensive_minion 22, wind 20, machine 12, undead 11, dark 9, fire 7, light 3, water 3, darkmatter 2, electric 2, ice 1; 1 with none |
| `Auxiliary/Status Effects/Buffs/` | `buff` | 12 | none | see section 3 |
| `Auxiliary/Status Effects/Debuffs/<N Stars>/` | `debuff` | 52 | 1★ 12, 2★ 16, 3★ 14, 4★ 10 | see section 3 |
| `Auxiliary/Items/` | `aux item` | 48 | from PSD: 1★ 4, 2★ 21, 3★ 12, 4★ 11 | items summoned by abilities ("04 Table4": the auxiliary items deck); types: equip 10, food 9, active 8, fragile equip 8, infliction 5, attack 4, field 4, deploy 2, support 1 |
| `Auxiliary/Creatures/` | `creatures` (plural, folder fall-through) | 7 | none | the large cards: Azamoth the Possessed Vessel, Igor Roballtowski, Mettaton EX, Mettaton NEO, Militron (No Armor) (printed "Militron (no Armor)"), Morpho Knight, Shadow Queen. `type==creature` misses them; `type~creature` finds both |
| `Auxiliary/Rulebook/` | `rulebook` | 48 PSD+PNG | - | pages 00..22 (section 6) |
| `Auxiliary/Markers/` | excluded | 1 PSD (`Green and red.psd`), 10 PNG | - | Green.png/Red.png (the .5 tile faces), health/defense/attack/speed_marker.png and their minus_ versions (stat markers) |

Retired cards: `MDW/The Morgue/` (12 PNGs incl. cardback: Cho'Gath, Developper, Ech, Fire Guardian's Stronghold,
Grandfather Clock, Pandeity Old, Poison Incantation, Satou, Satou Matsustaka, The Missile, ...); excluded from stats.
On the TTS table 7 of them are the "Deck of Rejects" in the Specific Item Decks bag.

## 2. Type vocabulary (Types/, Types/Stars/, stoogedings.json)

`stoogedings.json` has one key, `icons`: 91 entries = 35 `type` (Types/*.png, PUA U+E000..U+E022) and 56 `star`
(Types/Stars/*.png, U+E023..U+E05A). U+2605 (★) is the gold star. Token grammar (build_stoogedings.py docstring):
`[fire]` icon in its colors, `[fire:tint]` one-color in the text color, `[fire:sticker]` icon on an outline, `[fire:no]`
icon crossed out with a red prohibition sign (`[fire:glow]` / `[fire:darkglow]` on a white / black halo, branch
work/glow-tokens); case-insensitive, space/_/- interchangeable.

The 35 type icons, grouped by what the rulebook uses them for:

| Group | Tokens (aliases) |
|---|---|
| Creature types (rulebook "20 Creature Types", 14) | machine, wind, rock, panda, fire, ice, water, electric, undead, dark, light, darkmatter ("dark matter"), unknown, girl |
| Other creature-type icons | lightshadow ("light shadow"; the TTS "Light" token uses this PNG), null (TTS token "No Type") |
| Minion types | offensive_minion, defensive_minion |
| Item types (rulebook "14 Item Types", 10) | attack, counter, deploy, equip, food (= Healing items), fragile equip, infliction, negation, support, active |
| Status effect kinds | buff, buffstack ("buff stack"), debuff, debuffstack ("debuff stack"), debuffupgrade ("debuff upgrade") |
| Card kinds | field, nme ("n.m.e") |

Tint modes: `auto` 61 entries, `silhouette` 26 (10 types, 16 stars), `{"mode": "detail", "cut": "dark"}` 4 (undead, unknown, nebula_star, bfables).
Silhouette types: girl, buff, dark, darkmatter, debuff, fire, light, rock, support, water.
Star icons with aliases: 6star/7star/8star ("6 star"...), isaac_no_circle, sapphire_starold ("sapphire star"),
the_poorly_drawn_barnstar ("poorly drawn barnstar", "barnstar"), thunderfury_blessed_blade_of_the_starseeker ("thunderfury"),
warpstar ("warp star"), yinyangultra ("yin yang ultra", "yin yang"). The other star icons are series-themed stars
(crystal/diamond/emerald/garnet/gold/ruby/sapphire star and their _pmttydns_icon versions, dst_star, elden_ring,
elden_stars, faith, fallen_star, feather, greed, guild_seal, hk_star, lol_star, mario_star, monkey_d_star, nebula_star,
nether_star, ninja_star, null_star, omori_star, patrick_star, repo_orb, scrap, silksong_star, solar_sun, star_of_beth,
starstorm, stuffed_star, triforce, bfables, bgold, bronze, isaac, nuke, silver, star, undertale).

Type icons in use on cards (all kinds, website): equip 69, offensive_minion 68, machine 64, field 54, food 53, wind 45,
support 43, light 43, dark 35, undead 35, fire 35, debuff 27, panda 26, nme 26, defensive_minion 22, active 22, attack 22,
rock 22, water 21, debuffstack 20, fragile equip 19, infliction 17, electric 17, darkmatter 13, unknown 12, buffstack 11,
ice 10, counter 9, deploy 7, negation 6, debuffupgrade 5, girl 2, buff 1. Never on a card: lightshadow, null.

Creature type effects (rulebook "20 Creature Types"; the page's text puts each description BEFORE its type name):
MACHINE: cannot have food items used on them; immune to Sickness, Plague, Poison. WIND: can swap out/in for 1 less point.
ROCK: +1 Brick when mining for Bricks. PANDA: immune to Stuffed, Engorged. FIRE: immune to Smoldering, Burning, Blazing.
ICE: immune to Chilled, Frostbite, Frozen. WATER: immune to Suffocation. ELECTRIC: immune to Paralyzed.
UNDEAD: immune to Sickness, Plague. DARK: immune to Afraid. LIGHT: immune to Blindness. DARK MATTER: immune to Corruption.
UNKNOWN: once per game on your turn may permanently change this type into another creature type.
GIRL: gain 1 additional point (OD); may only be an innate type in special cases, but may be added through effects.
These match the status effects' own "can't affect [type]" text (checked for every immunity listed).

### Timing tags (badge_tags)
`stoogedings.json` has NO badge_tags key. The tag list lives in `build_stoogedings.py` `badge_tags()` and `BADGE_GROUPS`
(cooldown and uses badges since `8ec1b925`, 2026-10-21), mirrored by `site/assets/text.js` `tagInfo()` (`c3f9d685`,
`73904dbd`, MichaelJSr, written during the update: a reading of the cards, not a designer source). n = 1-9 or X; in a pair a/b, a = -1, 0-9 or X and b = 1-9 or X; a "-" is drawn as a minus sign. Groups and
colors (the rulebook "02 Abbreviations" time frames; the old tags stay for the rulebook pages):

| Group (color) | Tags | Website tooltip (`tagInfo`) |
|---|---|---|
| On your turn (`turn`, blue #2563eb) | (OYT), (nCD), (a/bCD); old (XPT), (EXT), (FTEXT) | "On your turn"; "Cooldown: 1 turn" / "2 turns"; "(1/2CD) Cooldown: 1 turn, then 2"; old ones "Old system: usable once per turn (on your turn)", "Old system: every 2 turns, starting on turn 2 (multi-turn)", "... starting on your 1st turn (multi-turn)" |
| At any time (`any`, purple #7c3aed) | (AAT), (nGCD), (a/bGCD); old (XPGT), (EXGT) | "At any time"; "Global cooldown: 1 global turn"; "(-1/1GCD) Global cooldown: -1 global turn, then 1"; "Old system: usable once per global turn (at any time)", "Old system: every 2 global turns (at any time, multi-turn)" |
| Forced (`forced`, red #dc2626) | (BOT), (ENDT), (BOGT), (ENDGT) | "Beginning of turn (forced)" and so on |
| Per game and triggers (`trigger`, orange #b45309) | (OFS), (OS), (OD), (WOF), (XPG), (nU) | "On first summon", "On summon", "On death", "While on field", "Can only be activated once per game", "(2U) 2 uses" |
| Other tags (`neutral`, slate grey #475569; not a time frame) | (Action), since TTSCardMaker `f6ce500e` (2026-10-22); any letter case draws it since the font's CASE lookup (TTSCardMaker work/badges-any-case, 2026-10-22) | "Action" |

(nU) in the per-game group could suggest uses per game; Time Riddle's "(1CD) -> (1CD)(2U)" suggests uses per cooldown.

Rulebook also defines on that page: SP = Star Points, DMG = Damage, HP/ATK/DEF/SPD = the stats, PF = Primary field,
CDMG = Counter damage; "Abilities marked with ⏳ are considered "multi turn" abilities."; "Abilites marked with 🕐 are
considered "time based"."; 'Time based abilities are considered "refreshed" and become usable again in the specified
timeframe.' The website's Rulebook tab glossary (`site/assets/app.js`) shows 22 badges: OYT, 1CD, 1/2CD, AAT, 1GCD,
1/2GCD, BOT, ENDT, BOGT, ENDGT, OFS, OS, OD, WOF, 1PG, 2U, Action (under "Other tags"), then the old 1PT, E2T, FTE2T, 1PGT, E2GT, with "X can be any
number; the first number of a pair such as (1/2CD) can also be 0 or −1."

## 3. Status effects (64 cards: 12 buffs, 52 debuffs)

Rulebook "19 Status Effects" / "19 Status Effects2" (verbatim points): a status effect is a buff or debuff. If it deals
DMG or heals, by default it procs in phase 4, after (ENDT) effects; otherwise it takes effect immediately. Duration counts down in phase 4.
DMG-dealing status effects are one instance of damage per unique stack ("5 stacks of Poison is one instance of 5 damage"),
piercing, and not an attack. The creature's owner chooses the order of resolution per unique status effect in phase 4.
If a creature started the global turn with a status effect and the global turn ends before it ended its turn, the effect
ticks at the end of the global turn, even in hand. Syntax: "[X]T[Y] [Status]", X = duration in turns, Y = stacks
("5T1 Poison" = 5 turns of 1 stack of Poison). Standard status effects: only one instance per creature. Stacking: any
number. On stack/upgrade/reapply, the duration becomes the longest of the applications. Upgrades: Smouldering +
Smouldering -> Burning; Smouldering/Burning + Burning -> Blazing; Chilled + Chilled -> Frostbite; Chilled/Frostbite +
Frostbite -> 1 turn of Frozen; Sickness + Sickness -> Plague. A creature with Frozen/Blazing/Plague can't have the prior
debuffs reapplied except to refresh the duration (need to upgrade to refresh). (The rulebook spells "Smouldering"; the
card is "Smoldering".)

Kind icons (the card's `types`): buffs: `buff` 1 (Diligent Watch), `buffstack` 11. Debuffs: `debuff` 27, `debuffstack`
20, `debuffupgrade` 5 (exactly the five upgradable ones: Smoldering, Burning, Chilled, Frostbite, Sickness). Mapping
`debuff`/`buff` = standard and `*stack` = stacking follows the rulebook 19.2 examples (inference from the page's example
icons; the page text itself does not name the icons). Buffs have no stars; debuffs have 1-4 stars from their folder.
Ability text below is identical in stats.pkl and cards.json, and the cooldown update changed no status effect card
(re-checked 2026-10-22: texts and the mention counts below are unchanged). "Cards naming it" = other cards (not status effects) whose
ability contains the name as a whole word, case-insensitive (approximate: "Frozen" also counts "Frozen In Time" mentions;
the website flags 0-mention ones as "Unused").

| Kind | Stars | Name | Type icon (kind) | Ability text (verbatim, `\n` = line break) | Cards naming it |
|---|---|---|---|---|---|
| buff | - | Diligent Watch | `buff` (standard) | When the creature this is on dies, the creature that inflicted this buff gains +3 ATK. | 1 |
| buff | - | Empowered | `buffstack` (stacking) | +1 ATK | 10 |
| buff | - | Enraged | `buffstack` (stacking) | +25% ATK | 2 |
| buff | - | Hardened | `buffstack` (stacking) | +0.5 DEF | 8 |
| buff | - | Hyper | `buffstack` (stacking) | +25% SPD | 4 |
| buff | - | Inspired | `buffstack` (stacking) | +1 Action | 4 |
| buff | - | Marked For Life | `buffstack` (stacking) | -25% DMG Taken | 2 |
| buff | - | Peppy | `buffstack` (stacking) | +1 SPD | 10 |
| buff | - | Regen | `buffstack` (stacking) | Heal 1 HP | 11 |
| buff | - | Reinforced | `buffstack` (stacking) | +50% DEF | 0 |
| buff | - | Thickened | `buffstack` (stacking) | +1 MAX HP | 2 |
| buff | - | Vigor | `buffstack` (stacking) | +25% MAX HP | 1 |
| debuff | 1 | Blindness | `debuff` (standard) | Attacks hit a random\nenemy creature\ncan't affect [light] | 8 |
| debuff | 1 | Burdened | `debuffstack` (stacking) | -1 SPD | 7 |
| debuff | 1 | Burning | `debuffupgrade` (upgradable) | 3 DMG, can't affect [fire] | 12 |
| debuff | 1 | Disgusted | `debuff` (standard) | Cannot consume healing items | 5 |
| debuff | 1 | Enfeebled | `debuffstack` (stacking) | -1 ATK | 20 |
| debuff | 1 | Poison | `debuffstack` (stacking) | 1 DMG, can't affect [machine] | 13 |
| debuff | 1 | Sickness | `debuffupgrade` (upgradable) | -0.5 DEF -1 ATK -1 SPD\ncan't affect [undead] [machine] | 9 |
| debuff | 1 | Smeckeldorfed | `debuffstack` (stacking) | 1 DMG | 5 |
| debuff | 1 | Smoldering | `debuffupgrade` (upgradable) | 2 DMG, can't affect [fire] | 20 |
| debuff | 1 | Softened | `debuffstack` (stacking) | -0.5 DEF | 14 |
| debuff | 1 | Stuffed | `debuffstack` (stacking) | -1 SPD, can't affect [panda] | 12 |
| debuff | 1 | Thinned | `debuffstack` (stacking) | -1 MAX HP | 1 |
| debuff | 2 | Afraid | `debuff` (standard) | -2 ATK -2 SPD\nCan't affect [dark] | 10 |
| debuff | 2 | Blazing | `debuff` (standard) | 4 DMG, can't affect [fire] | 11 |
| debuff | 2 | Chained | `debuffstack` (stacking) | -25% SPD | 0 |
| debuff | 2 | Chilled | `debuffupgrade` (upgradable) | -25% SPD can't affect [ice] | 13 |
| debuff | 2 | Confusion | `debuff` (standard) | Attacks hit random creatures.\nActions must be used for attacking. | 4 |
| debuff | 2 | Edible | `debuffstack` (stacking) | Creatures that attack this deal +25% lifesteal damage.\n(MAX 4 stacks) | 2 |
| debuff | 2 | Empty | `debuffstack` (stacking) | -25% MAX HP | 0 |
| debuff | 2 | Engorged | `debuffstack` (stacking) | -25% SPD, can't affect [panda] | 1 |
| debuff | 2 | Exposed | `debuffstack` (stacking) | -25% DEF | 0 |
| debuff | 2 | Frostbite | `debuffupgrade` (upgradable) | 2 DMG.\n-25% SPD, can't affect [ice] | 10 |
| debuff | 2 | Glued | `debuff` (standard) | Can't swap out. | 1 |
| debuff | 2 | Loony | `debuffstack` (stacking) | -25% ATK | 0 |
| debuff | 2 | Paralyzed | `debuff` (standard) | -50% SPD can't affect [electric] | 7 |
| debuff | 2 | Plague | `debuff` (standard) | -1 DEF -1 ATK -2 SPD\nCan't affect [machine] [undead] | 3 |
| debuff | 2 | Restrained | `debuffstack` (stacking) | (BOT) -1 Action | 3 |
| debuff | 2 | Spirit Flame | `debuff` (standard) | 2 DMG.\n-1 DEF | 2 |
| debuff | 3 | Become Back Money | `debuffstack` (stacking) | Attackers gain 1 point | 3 |
| debuff | 3 | Cheap | `debuff` (standard) | Afflicted creature gives 1 point on death. | 0 |
| debuff | 3 | Corrupted Blood | `debuff` (standard) | When the creature this is inflicted on attacks or is attacked, move this debuff to the attacker/attacked.\n(BOT) 4 DMG. | 0 |
| debuff | 3 | Corruption | `debuff` (standard) | Can't affect [darkmatter].\nIf anything occurs in a way that would resolve differently than if this creature's ability was negated, take 4 DMG. Can only activate (1GCD). | 6 |
| debuff | 3 | Cursed Inferno | `debuff` (standard) | 3 DMG. Lasts indefinitely | 2 |
| debuff | 3 | Disarmed | `debuff` (standard) | Equips have no effect. | 1 |
| debuff | 3 | Frozen | `debuff` (standard) | Can't affect [ice].\nLose all actions, cannot switch out. | 9 |
| debuff | 3 | Hexed | `debuff` (standard) | If 3 creatures on the field are inflicted with Hexed, Hexed is cleansed and they take 2X DMG until end of their next turn.\n2 DMG if afflicted creature is in the hand. | 2 |
| debuff | 3 | Imprisoned | `debuff` (standard) | Lose all actions, cannot switch out. | 5 |
| debuff | 3 | Insane | `debuff` (standard) | Cannot switch out.\nForced to use all actions to attack a random friendly creature. | 1 |
| debuff | 3 | Pessimism | `debuffstack` (stacking) | -1 HP from healing. | 1 |
| debuff | 3 | Resentful Glare | `debuff` (standard) | When dealing damage to an enemy\ncreature, take 3 DMG. | 2 |
| debuff | 3 | Shell Shock | `debuff` (standard) | -1 DEF, +1 ATK, 50% chance when attacking to hit target one clockwise slot over instead \n(can hit friendly creatures). | 1 |
| debuff | 3 | Suffocation | `debuffstack` (stacking) | 1 DMG\nCan't Effect [water] | 4 |
| debuff | 4 | Condemned | `debuff` (standard) | After this expires, this creature is destroyed. | 2 |
| debuff | 4 | Contracted | `debuff` (standard) | No effect. Lasts until the\ncreature exits the field. | 0 |
| debuff | 4 | Curse | `debuffstack` (stacking) | (BOGT) Lose 1 SP | 0 |
| debuff | 4 | Despair | `debuff` (standard) | -3 ATK -0.5 DEF -3 SPD | 2 |
| debuff | 4 | Frozen In Time | `debuff` (standard) | Lose all actions, can't switch out, use items, equip items, use ability, or be healed | 2 |
| debuff | 4 | Hemorrhaging | `debuffstack` (stacking) | 0.5 True DMG | 2 |
| debuff | 4 | Infected | `debuff` (standard) | +2 ATK,\nThis creature is negated. | 2 |
| debuff | 4 | Marked For Death | `debuffstack` (stacking) | +25% DMG Taken | 1 |
| debuff | 4 | Recursion | `debuff` (standard) | (BOT) Double all stackable buffs and debuffs on this creature | 0 |
| debuff | 4 | Withering | `debuff` (standard) | (BOT) 2 DMG -0.5 DEF -1 ATK -1 SPD. | 2 |

## 4. Timing tags actually used in ability texts

`tags.py` over the ability text of the 767 distinct cards in the cards.json shape (copy rows skipped; rulebook pages
excluded). Its regex also matches pairs such as (1/2CD) and (-1/1GCD), which the 2026-10-20 regex missed (hence 604/351
then, 606/352 for the same data now). Now = fresh parse of `73904dbd`; before = PICKLE `56b43b9`. Group = badge group
(section 2). Meanings of the new tags are the website's readings, not rule text (SKILL.md §3).

| Tag | Now uses | Now cards | Before uses / cards | Group | Where now (card kind: uses) |
|---|---|---|---|---|---|
| (1CD) | 101 | 88 | 13 / 13 | turn | creature 90, item 6, minion 3, creatures 1, nme 1 |
| (1GCD) | 94 | 82 | 22 / 20 | any | creature 41, minion 16, field 15, item 12, nme 6, aux item 3, debuff 1 |
| (2CD) | 55 | 55 | 14 / 14 | turn | creature 45, minion 6, item 3, creatures 1 |
| (OYT) | 44 | 42 | 40 / 38 | turn | creature 18, item 13, field 5, aux item 3, nme 3, creatures 1, minion 1 |
| (BOGT) | 41 | 39 | 41 / 39 | forced | creature 17, field 9, item 8, nme 3, aux item 2, creatures 1, debuff 1 |
| (ENDGT) | 39 | 37 | 39 / 37 | forced | creature 16, item 11, nme 6, field 3, minion 2, aux item 1 |
| (OFS) | 35 | 34 | 35 / 34 | trigger | creature 31, field 2, creatures 1, minion 1 |
| (1PG) | 32 | 30 | 32 / 30 | trigger | creature 26, item 3, field 2, aux item 1 |
| (ENDT) | 28 | 26 | 28 / 26 | forced | creature 20, aux item 3, item 2, creatures 1, minion 1, nme 1 |
| (BOT) | 25 | 24 | 25 / 24 | forced | creature 15, debuff 4, item 3, aux item 1, minion 1, nme 1 |
| (OD) | 23 | 23 | 23 / 23 | trigger | creature 21, minion 2 |
| (WOF) | 22 | 20 | 22 / 20 | trigger | creature 21, aux item 1 |
| (2U) | 18 | 17 | 6 / 6 | trigger | creature 11, item 3, minion 3, field 1 |
| (1/2CD) | 15 | 15 | 1 / 1 | turn | creature 15 |
| (AAT) | 14 | 13 | 14 / 13 | any | creature 9, field 2, aux item 1, item 1, minion 1 |
| (2GCD) | 11 | 11 | 2 / 2 | any | creature 9, aux item 1, field 1 |
| (OS) | 8 | 8 | 8 / 8 | trigger | creature 8 |
| (2PG) | 4 | 4 | 4 / 4 | trigger | creature 3, item 1 |
| (3PG) | 3 | 3 | 3 / 3 | trigger | creature 3 |
| (XPG) | 2 | 2 | 2 / 2 | trigger | item 2 |
| (3U) | 2 | 2 | 1 / 1 | trigger | creatures 1, field 1 |
| (4U) | 2 | 2 | 1 / 1 | trigger | creature 1, item 1 |
| (3CD) | 2 | 2 | 0 / 0 | turn | creature 2 (Empress Bulblax, Future Niiue) |
| (XU) | 2 | 1 | 0 / 0 | trigger | nme 2 (Michael's Blessing) |
| (-1/1GCD) | 1 | 1 | 0 / 0 | any | aux item 1 (The Temple of Ahn'Qiraj) |
| (XCD) | 1 | 1 | 0 / 0 | turn | nme 1 (Michael's Blessing) |
| (XGCD) | 1 | 1 | 0 / 0 | any | nme 1 (Michael's Blessing) |

Total now: 625 occurrences of 27 tags in 352 cards. Before: 606 occurrences of 37 tags in 352 cards, of which these are
now gone from the cards: (1PT) 83 / 72, (1PGT) 61 / 55, (E2T) 42 / 42, (FTE2T) 14 / 14, (E2GT) 8 / 8, (2PGT) 6 / 6,
(2PT) 6 / 6, (E3T) 2 / 2, (XPT) 2 / 2, (-1/1CD) 1 / 1, (3PGT) 1 / 1, (4PGT) 1 / 1, (EXGT) 1 / 1, (EXT) 1 / 1, (XPGT) 1 /
1. The tags the rulebook defines that remain on cards: 320 uses in 220 cards. Family totals now (uses / cards): (nCD)
159 / 138, (a/bCD) 15 / 15, (nGCD) 106 / 93, (a/bGCD) 1 / 1, (nU) 24 / 22; 218 cards carry at least one.

### 4.1 What changed, per line (`changes.py` over the 178 changed PSDs)

Line pairs before → now (pairs by line position): (1PT) → (1CD) 74, (1PGT) → (1GCD) 49, (E2T) → (2CD) 40, (FTE2T) →
(1/2CD) 11, (E2GT) → (2GCD) 8, (2PGT) → (1GCD)(2U) 5, (2PT) → (1CD)(2U) 4, (1PGT)(OYT) → (1GCD)(OYT) 3, (E3T) → (3CD)
2, and once each (1PT) → (1GCD)(OYT), (1PT)(Action) → (1CD)(Action), (4PGT)(OYT) → (1GCD)(4U)(OYT), (FTE2T)(BOT) →
(1/2CD)(BOT), (E2T)(BOT) → (2CD)(BOT), (2CD) → (2GCD)(OYT), (-1/1CD) → (-1/1GCD)(OYT), (3PGT) → (1GCD)(3U), (1CD) →
(1GCD)(OYT), (FTE2T)(Action) → (1/2CD)(Action), (E2T)(Action) → (2CD)(Action). Michael's Blessing reordered its rows, so
its positional pairs "(XPT) -> (XCD)" and "(EXT) -> (1CD)(XU)" read by meaning as (XPT) → (1CD)(XU) and (EXT) → (XCD).
Old tags after other text on a line are not in these pairs: 20 uses on 12 cards (the right-hand sides of the two
conversion tables, the label and sentence-end tags listed in 4.2, and The Bank's deleted line). By commit: `df02b3b6`
50 PSDs, `09131c67` 49, `1b4f36a0` 50, `7cb4c867` 29. By kind: creature 141, item 17, field 14, nme 4, aux item 2.

The six cards whose wording changed (before → now; everything else in the 178 changed tags only):
- Energy Ritual: "refreshes all time-based and (XPG) abilities that have been used up" → "fully recharges and restores
  all usages of all time-based and (XPG) abilities".
- Michael: "(E2GT) Refresh another creature's ability usage as though another turn has passed" → "(2GCD) Recharge 1 turn
  of charge to another creature's ability"; "Permanently double the ability usage of an ability" → "... double the usage
  amount of an ability"; the (WOF) line's "3/4 shop price (1PT)" → "(1CD)".
- The Bank (N.M.E.): "If a creature does not use abilities usages and/or actions by (ENDGT)" → "... does not use actions
  and/or ability usages by (ENDGT)"; "Will store individual uses of a multi-use (XPT) abilities." deleted; "Stored
  actions/ability's cooldowns refresh as if they were used." → "Cooldowns of stored actions/abilities recharge as if they
  were used."; "stored charges are lost" → "stored actions and ability usages are lost". It keeps (ENDGT).
- Landia: "(OD) Equip Dragon Mounts to all other friendly creatures currently on the field." → "(OD) Equip a Dragon Mount
  to all other friendly creatures currently on the field." (reads as one each).
- Tainted Bethany ("destoryed" → "destroyed") and Tainted Lazarus ("Flip to to" → "Flip to"): typo fixes, plus their tags.

Conversion tables on cards, verbatim (now; before in brackets):
- Time Riddle (Items/Equip 4★): "Has no effect until 2 lanes have gone. / Doubles all time-based and (XPG) ability usage,
  for example: / (2CD) -> (1CD) / (1CD) -> (1CD)(2U) / (1PG) -> (2PG) / (1GCD) -> (1GCD)(2U)" [(E2T) -> (1PT) / (1PT) ->
  (2PT) / (1PG) -> (2PG) / (1PGT) -> (2PGT)].
- Michael's Blessing (N.M.E. 5★): "Purchase Cost: 5 Points / (OYT) -> (1GCD) / (BOT)/(ENDT) -> (1GCD) / (BOGT)/(ENDGT) ->
  (1GCD) / (XCD) -> (XGCD) / (1CD)(XU) -> (1GCD)(XU)" [(OYT), (BOT)/(ENDT), (BOGT)/(ENDGT) -> (1PGT) / (XPT) -> (XPGT) /
  (EXT) -> (EXGT)].

### 4.2 Line facts (`lines.py`, fresh parse)

| Family | Lines | With (OYT) | With (AAT) |
|---|---|---|---|
| (nCD) | 157 | 0 | 0 |
| (a/bCD) | 15 | 0 | 0 |
| (nGCD) | 105 | 9 | 0 |
| (a/bGCD) | 1 | 1 | 0 |
| (nU) | 23 | 1 | 0 |

- Global cooldown + (OYT) lines (10): Clock, Crock Pot, Error room, The Claw, Water Totem "(1GCD)(OYT)"; The Gates of
  Ahn'Qiraj "(2GCD)(OYT)"; Star Cannon "(1GCD)(4U)(OYT)"; The Temple of Ahn'Qiraj "(-1/1GCD)(OYT)"; Jetpack "(1GCD) May
  either negate the damage of an attack (not towards itself) or (OYT) negate an ability until (ENDGT)."; Michael's
  Blessing "(OYT) -> (1GCD)". Before the update Clock, Error room and The Claw read (1PGT)(OYT) and Star Cannon
  (4PGT)(OYT); the other four gained (OYT) when they moved to a global cooldown.
- The tag right before each (nU): (1GCD) 13, (1CD) 11; (nU) is on 22 cards.
- Line starts (numbers as n): (nCD) 109, (nGCD) 70, (nCD)(n points) 24, (n/nCD) 10, (nGCD)(nU) 9, (nGCD)(OYT) 6, (nCD)(nU)
  5, (nCD)(nU)(n points) 3, (nCD)(n point) 3, (nGCD)(n points) 3, (nCD)(Action) 2, and single ones such as
  (-n/nGCD)(OYT), (n counters)(nGCD), (nGCD)(BOGT), (n/nCD)(BOT), (nCD)(BOT), (n points)(nCD).
- A cooldown tag after other text: 20 lines, 8 of them the two conversion tables. The other 12: Corruption "... take 4
  DMG. Can only activate (1GCD).", Michael "(WOF) Discarded items may be bought for 3/4 shop price (1CD).", Niko "If you
  have no attack items, (1CD)(2 points) may draw a Mighty Pebble.", OMORI "2+ death counters: +1 SPD, (1/2CD) Attacks
  inflict 2T Afraid.", Pix "Blue: (1GCD) ...", Spice "Can only be used on friendly creatures (1GCD).", Supreme Jeh'Oul
  "When this creature falls to 50% HP or below, this ability becomes and stays (1CD)(2U).", Waddle Dee "When an attack
  or infliction is declared towards your team, (1GCD)(2.5 points) may redirect ...", Warcraftian Druid "Bear: +1 DEF,
  (1GCD) Intercept ...", "Moonkin: +2 ATK, (1GCD) Gain 1 SP.", "Tree: +0.5 DEF, (1CD) Heal ...", Yin Yarn "Magic Form:
  (1CD) ...". Po's three "- (1CD) ..." / "- (1GCD) ..." list items count as line starts.
- (1/2CD) cards (15, all creatures): Auto Croissant, Chris & Jordi, Dark Daroach, Daroach, Doctor Zomboss, Drawcia,
  Flurrie, Isma, Koops, Kracko, omori, The Abyss Watchers, Vivian, Yoshi Kid, Zero. With a (2CD) ability too: Chris &
  Jordi, Daroach, Dark Daroach, Doctor Zomboss, Koops, The Abyss Watchers, Vivian, Yoshi Kid (each was an (FTE2T)/(E2T)
  pair before, except Chris & Jordi, which already read (2CD)/(1/2CD)); Isma pairs it with (1CD).
- Uses written in words on items, not as (nU): "3 uses:" Crystal Spikes, Flame Trap, Iron Spikes, Spiky Ball Trap, Stone
  Spikes, Wooden Spikes ("3 Uses:"); "5 uses:" Poison Trap; "Uses: 3" Return To Sender; "2 uses" Star Scraper (N.M.E.);
  "Uses: ∞" Pile Of Swords, Spice, Wand of Obesity.
- Words for the cooldown idea on cards: "recharge" (Energy Ritual, Michael, The Bank), "usage(s)" (Energy Ritual,
  Michael, The Bank, Time Riddle, E. Gadd's Garage, Utica Stock Exchange "(1GCD)(3U) -0.5 points to ability cost for that
  usage."), "refresh" (Cave Of The Past, Hot Spring, Tainted Bethany "(1CD)(4 points) Refresh the first ability.", The
  Kestrel, The Maverick, Warcraftian Druid), "cooldown" (The Bank; Stalwart Shell "Unequipping this item resets its
  cooldown"), "time-based" (Cave Of The Past, Energy Ritual, Time Riddle).
- "(OPT)": on no visible card text. It is in the hidden `old` text layer of Bobberlington.psd and Michael.psd (an earlier
  Fledgling Creation ability: "(5 counters) (OPT) summon a non-protective 1/2/3/4 Fledgling Minion ...") and in the
  12_Reactions4 picture of Whispy Woods (the card reads "(1CD)(2 points) Heal all friendly creatures by 3 HP.").
- (3CD) on Empress Bulblax is in two layers: a hidden `Ability` layer and the visible `text 4`; the parse counts it once.

Other parenthesised words in ability text (not timing tags): (Action) 36 (+1 lowercase "(action)"; 36 uses in 28 cards; a grey
badge since 2026-10-22, the lowercase one stays text; the Stats page's "Timing tags" chart leaves it out),
point costs (2 points) 34, (4 points) 15, (3 points) 14, (1 point) 13, (1.5 points) 3, (5 points) 3, (0.5 points) 1,
counters such as (x counters) 3, (3 counters) 2, (4 counters) 2.

## 5. Constructed starting hand

### 5.1 Rulebook "03 Setting Up" (verbatim content)
Each team starts with 2 points and 7 star points. Starting hand: one 5-star creature, two 4-star creatures, three 3-star
creatures, one 2-star creature; two 4-star items, three 3-star items, three 2-star items, two 1-star items (the extracted
text shows "⭐⭐ ⭐⭐" under "Two 1-star items", a typo on the page). "Refer to SHOPS for setting up the shops!" Each team
fills each of the 3 slots with a creature face down; 3 "Facedown items" are placed one behind each slot. Flip the
creatures when everyone is ready; items are revealed only when played. Only your side can see your Facedown Items while
they are in the holders; pull them out to reveal and play them.
Related: "04 Table2": the Star Point counter is set to 7 at the beginning of each global turn; items cost their star
value in star points. BRICKS buy field cards (2x star level); mining gives BRICKS equal to the creature's star level;
1 BRICK to each team at the end of every global turn. Points: start with 2; gain points equal to your creature's star
level when it dies; spent on abilities and on the item shop (4x star level).
"05 Shops": item shop = 2 items per star level, revealed, cost 4x star level in points, reshuffle once per global turn
for 1 point, restock same star level. Field shop = 2 field cards per star level, only 1 per star level revealed before
the game, the remaining 4 revealed at the end of global turn 1, cost 2x star level in BRICKS, no restock. N.M.E. shop =
4 random N.M.E. items, all revealed, restocked randomly, cost printed on the card.

### 5.2 Website deck builder (`site/assets/deck.js`)
```js
export const HAND = [
  { kind: "creature", label: "Creatures", tiers: [[5, 1], [4, 2], [3, 3], [2, 1]] },
  { kind: "item", label: "Items", tiers: [[4, 2], [3, 3], [2, 3], [1, 2]] },
];
export const FIELD_MAX_STARS = 3;
```
17 slots (7 creatures + 10 items) plus an optional field card of 3★ or less "for the shop". Only `type == "creature"`
and `type == "item"` fill slots (minions, aux items, N.M.E. items, status effects, the plural "creatures" cannot).
One of each card (`already in this hand`); a banned card makes the hand illegal (it can still be added, with a warning);
the field card is also checked against the ban list. `randomFill` picks unbanned, parsed cards. Export: text list,
share link `d/#/deck/<base64>`, a TTS save with a creature pile (back `MDW/cardback.png`), an item pile and the field
card (back `MDW/cardback_item.png`), and a print sheet. Builder text (unchanged; only its comment now names `site/Constructed Ban List.txt`): "7 creatures and 10 items, one of each, none
banned, plus an optional field card (≤3★) for the shop."

### 5.3 Constructed ban list (`site/Constructed Ban List.txt`, moved from `MDW/` on 2026-10-20 in `13ac6802` with no change; also cards.json `banlist`, 53 entries)
Notes: "Constructed format is where you can choose whatever cards you want out of the entire deck for your hand." Ban
list "to avoid unfun playstyles and overly broken combos." "You may also only have one of each item in your hand."
"Each player can also choose one 3 star or lower field item to be active in the shop at the start of the game (besides
banned ones, obviously)."

| Group | Banned (reason) |
|---|---|
| 5 star creatures (8) | The Radiance (creature/ability copying), Marx (antifun), Baba (too easy to build combos, too powerful of a counter), The Puppet (reliance on RNG), Gandhi (reliance on RNG), Chef Saltbaker (too powerful of a counter), Michael (creature/ability copying), Leshy (creature/ability copying) |
| 4 star creatures (8) | Old Foxy, Acrophobia, Tainted Keeper (too difficult to counter and/or requires being countered), Fawful (antifun), DooPliss (creature/ability copying), Stranger (overcentralization, antifun), Buzzo, Green Mage (too easy to build combos) |
| 3 star creatures (9) | Leeroy Jenkins (too powerful of a counter), Shield Master Pi, Tainted Bethany (too easy to build combos), Something, Master Hand (antifun), Dark Matter Swordsman, Taranza, Holo defense API (creature/ability copying), Stay (antifun, too powerful of a counter) |
| 2 star creatures (4) | Peter Panda, Koops (too easy to build combos), Barry The Barrel (reliance on RNG), Auto Croissant (extremely dangerous) |
| 4 star items (8) | Amazing Mirror (creature/ability copying), The Wall (By Ech) (too easy to build combos), Golden Metronome (antifun, too powerful of a counter), Mimic, Flip (too easy to build combos), Magic Shield (antifun), Time Riddle (too easy to build combos, creature/ability copying), Standstill (antifun, too powerful of a counter) |
| 3 star items (8) | Stalwart Shell, Reflection, Blank Card, Credit Card, Wand of Obesity (too easy to build combos), Scapular, Life Link, Forget Me Now (too easy to build combos, too powerful of a counter) |
| 2 star items (3) | Tomorrow I'll (too easy to build combos, too powerful of a counter), Gust of Wind (too powerful of a counter), Berserk (too easy to build combos) |
| 1 star items (1) | NME stock (simply too strong) |
| Field Items (4) | The Bank, E. Gadd's Garage, Error room (too easy to build combos), Cave of the Past (too powerful of a counter) |

All 53 resolve to a card on the website, with the 2026-10-21 names and with the fresh printed titles (build_site.py's
letters-and-digits match on name, file name and aliases). Mismatches between the list and the card data: "The Wall (By
Ech)" is the card Blockade (Items/Active/4 Stars; printed title "The Wall By Ech", aliases "the wall", "Blockade");
"Leeroy Jenkins" prints as "LEEEEEEEEEROY"; "Berserk" is "Berserk!"; Tainted Bethany is a 4★ creature
(listed under 3 star); NME Stock™ is a 2★ Support item (listed under 1 star items); The Bank is an N.M.E. item (2★,
listed under Field Items); Cave Of The Past and Error Room are 4★ field cards (so already outside the ≤3★ field rule).

## 6. Rulebook pages (48; `Auxiliary/Rulebook/*.psd` + PNG)

00 Table of Contents, 0000 Cover, 0000 Cover Book, 01 Getting Started, 02 Abbreviations, 03 Setting Up, 04 Table,
04 Table2..5, 05 Shops, 06 Reading Cards, 06 Reading Cards2, 07 Stats1..5, 08 Global Turn, 09 TakingYourTurn,
10 Attacking, 10 Attacking2, 11 Abilities, 11 Abilities2..7, 12 Reactions, 12 Reactions2..5, 13 Items, 14 Item Types,
14 Item Types2, 15 Minions, 16 The Field, 16 The Field2, 17 Death, 18 Field Cards, 19 Status Effects, 19 Status Effects2,
20 Creature Types, 21 Damage Types, 22 Negation. The table of contents lists 22 chapters: Getting Started,
Abbreviations, Setting Up, Table, Shops, Reading Cards, Stats, The Global Turn, Taking Your Turn, Attacking, Abilities,
Reactions, Items, Item Types, Minions, The Field, Death, Field Cards, Status Effects, Creature Types, Damage Types,
Negation. Sleepy Kirby on a page marks more technical rules. Win condition ("01 Getting Started"): defeat all of your
opponent's creatures while they don't defeat all of yours. Old versions: `MDW/Old Rules/` (44 files, 22 PSD+PNG pairs,
e.g. 01_Start_of_the_Game, 02_Setting_Up, 03_The_Field, 04_Creatures, 05_Stats): history only, no text extraction.

## 7. What sits on the TTS table (`TTS/decks.json`, `TTS/Impossibility Simulator.json`; `TTS/audit.py` until 2026-10-25)

Rulebook "04 Table".."04 Table5" names these: main deck = CREATURE DECKS (5★ leftmost, then 4, 3, 2), ITEM DECKS (4★
leftmost, then 3, 2, 1), FIELD CARDS for the field card shop (4★..1★), N.M.E. items for the N.M.E. shop; DEVELOPER DECK
(reduced creature decks of newer/recently changed cards) and "Baseline" creatures mixed into it; Global Turn counter
(starts at 1, ticks up once every filled slot on the field has taken a turn); Star Point counter; BRICKS counter;
Points counter; item / field card / N.M.E. shops; three LANES on the PLAYING FIELD; FACE-DOWN ITEMS; slots; STATUS
EFFECT MARKERS; GRAVEYARD; DISCARD PILE; FIELD CARD SLOTS; Auxiliary items deck; Auxiliary Minions deck; Minion card
holder; Redirection cube (RNG); Order Determinator (RNG for 1 in 2); 0.5 chip "in case a counter uses half integers";
counters (status effect durations, ability counters); Health/Defense/Attack/Speed markers ("Flip them over for stat
downs. One of these represents the actual value of a stat, not the bar. (1 DEF marker is 0.5 DEF, 1 HP marker is 1
HP)"); Active item holder; the simple version (decks without complex cards, no field cards or N.M.E. items); a copy of
the deck; a bag of specific sorted item decks; a bag of tokens for each creature type.

Save totals (a copy of `TTS/audit.py` writing to /tmp, run 2026-10-22 against HEAD `73904dbd`; the removed
`TTS/AUDIT.md` (at `843a4a99`) is the 2026-10-20 run against `9fbe683`, before `dceef8eb` added 23 cards): 228 top-level objects; 57
decks (49 on the table + 8 in the bag "Specific Item Decks", 7 of them empty); 2417 card objects (CardCustom 2411 + 6
"Card"); by place: marker 8, deck 2362, rulebook 46, loose 1. 876 distinct repo images. Repo cards missing from the
complete decks: none. Nicknames match the website names (773 pairs, 0 mismatches). After the updater run of 2026-10-22
(TTSCardMaker, on b6330730): 2513 card objects (2458 in decks), no empty deck, 131 Nicknames (398 cards) set to the
printed titles of PICKLE 3888392. Since 2026-10-21 the links are pinned
to commits and re-pinned when a file changes (after the cooldown commits, four runs re-pinned 50, 50, 50 and 29 links).

### 7.1 Decks (57) by group and copies

| Group (tile) | Decks | Cards per deck | Copies |
|---|---|---|---|
| Creatures (complete) | 5★ 56, 4★ 93, 3★ 103, 2★ 45 (Stubborn Queen sits in Minions instead) | 1 copy each card | twin set at "Creatures Copy" |
| Creatures Developer Version | 5★ 14, 4★ 14, 3★ 21, 2★ 18 (hand-picked, unnamed decks; new cards join, `add_new_since` 318001ea: Z!!!!!, Tainted Lazarus, John Ward) | 1 | none |
| Baseline Creatures | 4★ 25, 3★ 22 (hand-picked) | 1 | none |
| Creatures Simple Version | 5★ 26, 4★ 42, 3★ 47, 2★ 17 (hand-picked) | 1 | none |
| Items (complete) | 4★ 30, 3★ 89 (Piko Piko Hammer since 2026-10-22), 2★ 96, 1★ 39 | usual count 1 (1★: 2); 3★: 18 cards at 2-3 copies; 2★: 21 cards at 2-5 copies (Metronome, Stopwatch ×5; Bomb, Basic Shield, Pep Drink ×3); 1★: 9 cards at 1 | twin set at "Items Copy" |
| Items Simple Version | 4★ 17, 3★ 68, 2★ 75, 1★ 30 (hand-picked) | same pattern (Metronome, Stopwatch ×4) | none |
| Field Items | 4★ 10 (White Palace to The Morgue, b6330730), 3★ 12, 2★ 18, 1★ 9 | 1 | twin set at "Field Items Copy" |
| N.M.E Items | 26 | 1 | "N.M.E. Items Copy" |
| All Status Effects | 64 (12 buffs + 52 debuffs) | 1 | 3 decks (2 on "All Status Effects", 1 "Status Effects Copy"); back `MDW/cardback_statuseffect.png` |
| Minions | 99 = Auxiliary/Minions 91 + Auxiliary/Creatures 7 + Stubborn Queen | 1 | 3 decks (2 + "Minions Copy") |
| Auxiliary Items | 60 = Auxiliary/Items 48 + 12 main-pool items summoned by abilities (Bomb, Poison Circle, Holy Card, Piko Piko Hammer, the 5 Ingredients, Pepperoni Pizza, Gasoline, N.M.E Rope) | 1 | 3 decks (2 + "Auxiliary Items Copy") |
| Bag "Specific Item Decks" | Deck of Rejects 7 (The Morgue, hand-picked, names kept), 1/2/3/4 Star Equips 9/17/23/16, 2/4 Star Attack Items 5/2, 2 Star Healing Items 22 (complete since 2026-10-22, back `MDW/cardback_item.png`) | 1 | none |

Backs: creatures and minions `MDW/cardback.png`; items, field cards, N.M.E. and auxiliary items `MDW/cardback_item.png`.
Repo cards missing from the complete decks: none. The 2026-10-20 gaps were filled by `dceef8eb` that day: The Gates of
Ahn'Qiraj (field 4★), Tainted Lazarus and John Ward (creatures 3★), Militron (creature 4★; a loose copy still sits on
the Penalty Box Creatures tile), Z!!!!! (5★, replacing the renamed ZOEY!!!!!), The Temple of Ahn'Qiraj (auxiliary
items), and Militron (No Armor), Anubisath Guardian, Qiraji Soldier, Silithid (minions).

### 7.2 Bags, counters, markers, tiles
- Rulebooks: Custom_Model_Bag with 46 page cards ("Page N: Title"); the two covers are not in it (0000_Cover_Book.png
  is the bag's own texture).
- Creature Types: Bag of 19 Custom_Token: Offensive Minion, Defensive Minion, Machine, Wind, Rock, Panda, Fire, Ice, Water,
  Electric, Undead, Dark, Light (lightshadow.png), Dark Matter, Unknown, Girl, No Type (null.png), Star (star.png),
  No Star (Null Star.png).
- Specific Item Decks: Bag of 8 decks (above).
- Infinite bags (12, two sets, one per side at x≈100.8): Counters ×2 (Counter), .5 ×2 (Custom_Tile, `Auxiliary/Markers/
  Green.png` face, `Red.png` back), Max HP ×2 (CardCustom "HP", `health_marker.png` / back `minus_health_marker.png`),
  DEF ×2 (`defense_marker` / `minus_defense_marker`), ATK ×2 (`attack_marker` / `minus_attack_marker`), SPD ×2
  (`speed_marker` / `minus_speed_marker`). These 8 marker cards are the stat markers: face = stat up, flip = stat down.
- Counters on the table (43): HP 10, DEF 10, ATK 8, SPD 8, Points 2, Star Points 2, Bricks 2, Global Turn 1.
- 36 unnamed Green/Red .5 tiles: 4 beside each of the 6 Creature Holders (24) and 12 near the bags.
- Holders and tiles: Creature Holder ×6 (3 per side, one per lane slot), Item Holder ×6 (facedown items, "Holds 1 card"),
  Minion Holder ×2, Active Holder ×2 (tokens), Field Item Holder ×2, Graveyard ×2, Discard Pile, Item Shop, Field Shop,
  N.M.E. Shop, deck tiles (Creatures, Creatures Copy, Creatures Developer Version, Baseline Creatures, Creatures Simple
  Version, Items, Items Copy, Items Simple Version, Field Items, Field Items Copy, N.M.E Items, N.M.E. Items Copy, All
  Status Effects ×2, Status Effects Copy, Minions ×2, Minions Copy, Auxiliary Items ×2, Auxiliary Items Copy).
- Penalty boxes: "Penalty Box Creatures" (Graveyard image) and "Penalty Box Items" (Discard image), both described
  "Needs to be changed"; the loose Militron card sits on Penalty Box Creatures. The rulebook does not mention a penalty box.
- Turn markers: 6 BlockSquare "Has not taken its turn". RNG: Die_6_Rounded "Redirection Cube", Quarter "Order
  Determinator". Also 2 go bowls "Pebbles", 16 HandTriggers, a Custom_Board (`MDW/Raid_Boss_Table_Top.png`), "Flex Table
  Control".
- Large cards: Lua `resizeTargets` keyed by Nickname (case-sensitive): Azamoth the Possessed Vessel, Igor Roballtowski,
  Mettaton EX, Mettaton NEO, Militron (No Armor), Morpho Knight, Shadow Queen: exactly the 7 Auxiliary/Creatures cards,
  resized to 3,1,3 out of a container; the updater keeps the table in step with the website Nicknames.

## 8. How the bot and the website expose the rules

Bot (AutoCroissant, `main.py` + `commands/query_card.py`):
- `/query_rulebook search_text [limit]`: plain case-insensitive substring (regex-escaped) over the 48 rulebook entries
  (`rulebook_dff`, rows whose path contains "Rulebook"); no DSL; results sorted by page name; each page's text sent with
  `render_type_tokens` (icons as emojis). The frame's ability column is lowercased, so the replies are lowercase text.
  Empty search returns every page.
- `/export_rulebook`: a text file of every rulebook page (non-ASCII stripped).
- `/query name`: card image by fuzzy file-name match (works for status effect cards, e.g. "poison", and for Types/ icons).
- `/query_ability query [limit] [howmany]`: the DSL over every non-rulebook card with ability text. For status effects
  and types: `type==debuff`, `type==buff`, `type==debuff & stars==3`, `types~debuffstack`, `types~debuffupgrade`,
  `types~undead`, `type~creature` (creature + the 7 large "creatures"), `subtype==equip`, text like `(1gcd)`, `"2t1 poison"`,
  `[fire]`, `can't affect [ice]`. The query is lowercased; `|` splits before `&`; `!` negates text only. It searches the
  bot's `stats.pkl` (new tags since PICKLE `3888392`): `(1gcd)` hits 82 cards and `(1pt)` none (only the rulebook page
  11 Abilities2 still holds it).
- `/website`: link to https://michaeljsr.github.io/TTSCardMaker.

Website (`site/`):
- Rulebook tab (`#/rules`): every page image + text, a search box ("Search the rulebook…"), a "Timing tags" glossary (21
  badges with their tooltips, old ones last; section 2) and the "Constructed ban list" section; each page has an Edit button.
- Cards tab: kinds Creatures, Items, Field cards, Minions, N.M.E. items, Auxiliary items, Status effects (buffs +
  debuffs), Auxiliary creatures; flags incl. "Unused status effects", "Duplicates", "Edited on the website"; banned cards
  get a "Banned" flag in the deck builder. Search uses the bot's syntax plus `total`, `name~`, `subseries~`, `!` on any
  term, `is:unparsed`, `has:history`, `has:problems`, `has:alias`. Timing tags in ability text render as
  colored badges with a tooltip when `tagInfo` knows them (anything else, such as "(OPT)" or "(2 points)", stays plain
  text); `[type]` tokens render as icons.
- Deck builder (`#/deck`, shared hands at `d/#/deck/<code>`): section 5.2.
- Stat colors used by the site: HP #2fbf3a (green), DEF #3d63ff (blue), ATK #ff3b3b (red), SPD #ffd21f (yellow).
