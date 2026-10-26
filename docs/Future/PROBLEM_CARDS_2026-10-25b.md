# Problem cards, 2026-10-25 (after fixes)

Cards are not required to have types or an ability: a card without them is listed only where an edit to it can't bake.

- **54 of 764 cards** have at least one website edit that can't bake (a real refusal).
- **4 cards** have an open data problem (Computer Virus has no stats; Twin Emperors' second creature has a different DEF on the website than on the card; 2 cards stuck under "Not parsed yet"); 5 more are ruled acceptable on purpose.
- **413 cards** bake some edit only with a note an editor must look at: 219 only on the census's title made 3 letters longer (it reaches the stars, type icons or picture); 150 on a stars edit (a folder move whose printed stars aren't redrawn, or stars laid out again); 38 on a same-length title fix or a one-word ability change; 15 on a stat or type edit.

Source: dry bakes of 13 kinds of edit over all 764 cards (6,813 bakes, AutoCroissant bef072e (99 cards re-run at work/censusfix), psd-tools fork cards-20261025b, TTSCardMaker a4d137bf (the 5 multi-picture PSDs at 572680d5, where their variant tags landed), nothing written), titles and abilities edited box by box as the website does since 2026-10-25, plus a check of the card data (data d547daef) and live website (2026-10-25 20:43 UTC). A card is listed under each reason it hits, so some cards appear more than once. Spot-check: I re-ran the dry bake for 4 refused edits (Jack-In-The-Box Zombie adding a star, Abyss Watcher adding an ability, Grimora HP, Lorian title) and all 4 refused again for the same reason.

> **What changed since this morning's list:** the website edits titles and abilities one box per layer (split, doubled and oddly named titles, several ability layers: one box each); a stars edit on a card starred by its "N Stars" folder moves the card; a card with no type icon takes its first at its template's spot (flagged); the bake follows card makers' own stat bars and star layouts. Still to come in code: adding an ability to a card that has no ability layer (+ability). Evening fixes (99 cards re-run): status effects' stars edits move them too; a type icon in a Stars group is no star (Toy Freddy); a split title keeps its parts after an edit; a first type icon touching the title bakes, flagged; a star added on a multi-picture PSD bakes.

## 1. Website edits that can't bake

Grouped by the part of the card. Each card shows its folder and the edits the reason blocks. "Who" is who would have to act for the edit to bake.

### Whole card

#### Sonic's SPD 20 is printed on a one-row bar (1 card)

- **Fix:** The bar shows SPD above 10 without the template's second row, so the read-back after any edit disagrees and every edit is refused. Rebuild SPD as the template's two-row bar.
- **Who:** Card maker in Photoshop (standardisation)

  - Sonic (Creatures/Other/4 Stars): ability, all four stats, stars (add and remove), types (add and remove)

### Title

#### No title layer the bot reads, so the website has no title box (9 cards)

- **Fix:** Make the title a visible text layer (named Name is best); the website then shows its box.
- **Who:** Card maker in Photoshop

  - ABBI (Creatures/Omori/5 Stars): title
  - Baba (Creatures/Other/5 Stars): title
  - Grey Prince Zote (Creatures/Hollow Knight/4 Stars): title
  - Holy Card (Items/Equip/2 Stars): title
  - Minecraft Panda (Creatures/Pandas/2 Stars): title
  - Pandanoko (Creatures/Pandas/2 Stars): title
  - The Last Cabbage Merchant (Creatures/Other/3 Stars): title
  - Tyler (Creatures/Other/5 Stars): title
  - Wolfgang (Creatures/DST/4 Stars): title

#### The title layer has another name, so the website has no title box (5 cards)

- **Fix:** Rename that one layer to Name. (The owner chose on 2026-10-22 to leave layer names as they are; these are the few the bot still can't match.)
- **Who:** Card maker in Photoshop

  - Alex YIIK (Creatures/Other/5 Stars): title; layer 'this card fucking sucks'
  - Chris & Jordi (Creatures/Everhood/5 Stars): title; layer 'Knight'
  - Lorian (Auxiliary/Minions): title; layer 'Twin Princes'
  - Lothric (Creatures/Souls/5 Stars): title; layer 'Twin Princes'
  - Thunder Thighs (Items/Heal/2 Stars): title; layer 'Snack'

#### The title layer is hidden, so the website has no title box (4 cards)

- **Fix:** Delete the hidden leftover Name layer; the printed title looks like part of the picture, so retype it as a visible text layer named Name to make it editable.
- **Who:** Card maker in Photoshop

  - E. Gadd's Garage (Field/2 Stars): title
  - Man Serpent Shield (Items/Equip/2 Stars): title
  - Sonic (Creatures/Other/4 Stars): title
  - Wigfrid (Creatures/DST/3 Stars): title

#### Two layers in the title area look like the title (2 cards)

- **Fix:** Merge them into one text layer named Name.
- **Who:** Card maker in Photoshop

  - Dark Matter Swordsman (Creatures/Kirby/3 Stars): title
  - Knight Lost A Lot (Creatures/Everhood/2 Stars): title

#### The bot's drawing of the new title is more than 25% off the card picture (4 cards)

- **Fix:** The bot's renderer draws this title far from how Photoshop drew it, so it won't risk the bake. Improve the renderer for these title styles, or edit the title in Photoshop.
- **Who:** Bot code (renderer); until then edit in Photoshop

  - Dark Tentacle (Auxiliary/Minions): title
  - Dull Knife (Auxiliary/Items): title
  - Shiny Knife (Items/Equip/3 Stars): title
  - Snow Angel (Creatures/Omori/3 Stars): title (typo fix only)

### Ability

#### No ability layer: adding an ability on the website can't bake yet (3 cards)

- **Fix:** The website offers "+ Add an ability" (a new layer from the card kind's template), but the bake for it is not built yet (planned after a Photoshop proof); the edit waits. Cards need no ability, so nothing is wrong until someone adds one.
- **Who:** Bot code (+ability, planned); until then add it in Photoshop

  - Abyss Watcher (Auxiliary/Minions): adding an ability
  - Shadow Duelist (Auxiliary/Minions): adding an ability
  - Warrior Dee (Auxiliary/Minions): adding an ability

### Stats (HP/DEF/ATK/SPD)

#### The stat bar's own layers can't show the new value (4 cards)

- **Fix:** The bake now follows the card maker's own bar layers; on these cards no layer can show the new value (bones painted into one layer, no dark picture for a cell, no layer switching at the value). Replace the bar row with the template's row, or edit in Photoshop.
- **Who:** Card maker in Photoshop

  - Dirty Bed (Items/Active/2 Stars): DEF
  - Grimora (Creatures/Other/3 Stars): all four stats
  - Skeleton Mate (Auxiliary/Minions): all four stats
  - Slugcat (Creatures/Other/2 Stars): SPD

#### Several bars for one stat (4 cards)

- **Fix:** The card prints more than one bar for a stat (Pix and Tainted Lazarus hold several creatures); edit these stats in Photoshop. No code change planned.
- **Who:** Nothing (edit in Photoshop)

  - False Knight (Creatures/Hollow Knight/2 Stars): all four stats
  - Pix (Creatures/Kirby/3 Stars): all four stats
  - Tainted Lazarus (Creatures/Isaac/3 Stars): all four stats
  - Ze'mer (Creatures/Hollow Knight/5 Stars): all four stats

#### The stats come from the card's creatures list on the website (1 card)

- **Fix:** Edit the creatures list on the website instead; a baked stat would be overwritten by it. The card also has several bars per stat.
- **Who:** Nothing (website creatures list)

  - Twin Emperors (Creatures/World Of Warcraft/4 Stars): all four stats

### Stars

#### A star layer has smart filters, so it can't be copied or moved (3 cards)

- **Fix:** Rasterise the star layer (or drop its smart filter).
- **Who:** Card maker in Photoshop

  - Jack-In-The-Box Zombie (Auxiliary/Minions): adding a star
  - Magnet-shroom (N.M.E): adding a star
  - Polevaulter Zombie (Auxiliary/Minions): adding a star

#### A new star would touch the title or icons (1 card)

- **Fix:** No room for another star; edit in Photoshop.
- **Who:** Nothing (edit in Photoshop)

  - PoD04Zebra Bloon (Auxiliary/Minions): adding a star

#### The stars are drawn twice, stacked, so the bot counts more stars than show (1 card)

- **Fix:** Delete the stacked copies.
- **Who:** Card maker in Photoshop

  - Living Bread (Auxiliary/Minions): stars (add and remove)

### Types

#### A new type icon would touch the title, stars or art (7 cards)

- **Fix:** No room for another icon; edit in Photoshop.
- **Who:** Nothing (edit in Photoshop)

  - Auto Croissant (Creatures/Other/2 Stars): adding a type; touches 'autocroissant'
  - Cloud Soiree (Field/1 Stars): adding a type; touches 'Color Fill 1 copy'
  - Crowned Magolor (Creatures/Kirby/5 Stars): adding a type; touches 'Dimensional_rift_C_Bsn_8VwAIzPgw.jpg_large_transparent copy 2'
  - Kyle (Creatures/Other/5 Stars): adding a type; touches 'Color Fill 1'
  - Masked Dedede (Creatures/Kirby/5 Stars): adding a type; touches 'Warpstar'
  - Poorly Drawn Cabbage (Auxiliary/Items): adding a type; touches 'Name'
  - Return To Sender (Items/Counter/4 Stars): adding a type; touches 'Aberson Linchdolt 101 Cardinal Dr. Macon IL 62544'

#### A type icon would end up off the card (5 cards)

- **Fix:** The icon row is already at the card's edge; edit in Photoshop.
- **Who:** Nothing (edit in Photoshop)

  - Crystal Guardian (Creatures/Hollow Knight/2 Stars): adding a type
  - Defcon Panda (Creatures/Pandas/3 Stars): adding a type
  - Masher (Auxiliary/Minions): types (add and remove)
  - Sword (Auxiliary/Minions): removing a type
  - X Fists (Auxiliary/Minions): adding a type

#### The type to remove is drawn as a frame round the card, not an icon (1 card)

- **Fix:** By design: a field frame isn't an icon the bot takes off; edit in Photoshop.
- **Who:** Nothing (edit in Photoshop)

  - Magnum Opus (Auxiliary/Items): removing a type

### Not counted: by design

- **Stars on 580 cards** come from the card's "N Stars" folder: a stars edit moves the card (PSD and PNGs) into the matching folder, creating it when missing (flagged). 579 of 580 moved in the census; a move that can't happen is listed above.
- **A first type icon** goes at the card template's spot (flagged "check it"): 77 of 77 cards with no type icon took one in the census; the rest are listed above.
- **The 12 buff status effects** show no stars at all, so there is nothing to add or remove.

## 2. Data problems

#### Card has no stats (1 card, open)

- **Fix:** Rebuild the card's stat bars from the template so HP/DEF/ATK/SPD can be read; separately, the website's "Parse problems" filter doesn't show this kind of problem (it is only reported in the card sync's notes).
- **Who:** Card maker in Photoshop for the bars; bot code for the website filter.

  - Computer Virus (Creatures/Kirby/3 Stars): no template stat bars, so the data and the website show no HP, DEF, ATK or SPD.

#### Website stats differ from the printed card (1 card, open)

- **Fix:** An editor checks which is right: the website's creatures list gives Vek'lor DEF 10, the card prints DEF 4 (edited on the website 2026-10-22).
- **Who:** Website editor (or card maker if the card is wrong).

  - Twin Emperors (Creatures/World Of Warcraft/4 Stars): second creature (Vek'lor) DEF 10 on the website, 4 on the card.

#### Picture without its own PSD, stuck under "Not parsed yet" (2 cards, open)

- **Fix:** The card data treats each as a copy of another card, but the website doesn't count such a copy as parsed, so no card sync will clear the "Not parsed yet" note. Teach the website these copies, or give each card its own PSD.
- **Who:** Bot code (website), or card maker in Photoshop.

  - Stubborn Queen (Creatures/Other/2 Stars): only a PNG; uses Stubborn King's text and stats. Its series also shows Other instead of Sort the Court!.
  - Gust of Trees (Items/Support/2 Stars): only a PNG; uses Gust of Wind's text and stats.

#### Ruled acceptable on purpose (5 cards, nothing to do)

- **Who:** Nothing; hidden from the website's problem filter on purpose.

  - Pix (Creatures/Kirby/3 Stars): several creatures on one card; an editor checked each one's stats.
  - Tainted Lazarus (Creatures/Isaac/3 Stars): several creatures on one card; an editor checked each one's stats.
  - Twin Emperors (Creatures/World Of Warcraft/4 Stars): several creatures on one card; an editor checked each one's stats.
  - Royal Eradicator Main Cannon (Creatures/FTL/5 Stars): ATK 12, above 10 on purpose.
  - Sonic (Creatures/Other/4 Stars): SPD 20, above 10 on purpose.

## 3. Bakes with a flag

These edits do bake; the bake carries a note for an editor. Most title notes come from the census's rename, which makes the title 3 letters longer (" II"); a shorter title change may bake clean. Cards marked * got the note even on a same-length typo fix. The stars notes come from adding or removing one star. By-design notes (a first type icon at the template's spot, a folder created by a stars move, copies and watermarks of the title that follow it) are not listed.

| Note | Edits | Cards | Editor must look? |
|---|---|---|---|
| title reaches the stars or type icons | 242 | 238 | yes |
| title prints past its old text onto the picture below | 174 | 171 | yes |
| stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (no group named Stars) | 159 | 87 | yes |
| title would print over a layer just below it | 123 | 121 | yes |
| stars: laid out again as rows of two (the card's own layout doesn't go on) | 41 | 41 | yes |
| stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (the bot reads another count from its Stars group's layers than the folder's, though the card prints the folder's: overlapping stars, one layer drawing several, or extra layers in the group) | 32 | 16 | yes |
| a layer above the title would cover part of the new title | 21 | 20 | yes |
| types: the first type icon, at the card template's spot, touches the title | 14 | 14 | yes |
| title runs off the card | 12 | 11 | yes |
| title runs past an effect layer clipped to it | 10 | 10 | yes |
| ability runs off the card | 10 | 10 | yes |
| ability runs over a stat bar | 9 | 9 | yes |
| ability prints over the picture or an effect layer | 6 | 6 | yes |
| title needs more lines than its box has | 5 | 4 | yes |
| a character of the new text has no glyph in the layer's font (drawn as a box) | 4 | 4 | yes |
| stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (a star layer has smart filters) | 3 | 3 | yes |
| stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (no room for another star) | 2 | 2 | yes |
| a styled word in the ability loses its style | 2 | 2 | yes |
| stat bar: a missing digit was made from the one below it | 1 | 1 | yes |
| a font the text uses isn't in MDW/Fonts: drawn in a stand-in | 1 | 1 | yes |
| stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (layout) | 1 | 1 | yes |
| overflow: ability: set at 94% of its size to stay on its panel | 1 | 1 | yes |
| ability redrawn whole, small pixel difference | 15 | 15 | no (pixel or font note) |
| title redrawn whole, small pixel difference | 10 | 7 | no (pixel or font note) |
| title's saved pixels don't match its named font | 7 | 4 | no (pixel or font note) |
| type icons baked with a small pixel difference | 4 | 4 | no (pixel or font note) |
| stars baked with a small pixel difference | 1 | 1 | no (pixel or font note) |

**Title reaches the stars or type icons (238 cards):** Abyss Watcher, Ancient Fuelweaver, Anubisath Guardian, Arachnophobia, Assault Panda, Asteroid Field, Auto Croissant, Autopsy Report, Azamoth the Possessed Vessel, Barry The Barrel, Battle Spear, Become Back Money, BergenTRUCK, Binge Eater, Blood Bomb, Bobberlington, Bobbery, Boggly Woods, Bolts and Screws, Brad Armstrong, Bucket Of Lard, Butterfly, Cabbage, Captain Spaceboy, Cave Of The Past, Chained, Charm Of The Vampire, Claw, Condemned, Corrupted Blood, Cortez, Crimson Flag, Croissant, Crystal Guardian, Crystal King, Crystal Spikes, Cursed Inferno, Dark Bramble, Dark Mind, Dark Nebula, Dark Ritual, Dark Sword, Daroach, Deepnest, Delicate Flower, Delta, Diligent Watch, Doc, Dogma, DooPliss, Door Shield, Drawcia, Drawcia Soul*, Dryya, Dumplings, Dyna Blade, Dynamite, Empress Bulblax, Excess Express, False Knight, Feast, Fire Staff, Flower Of Decay, Flurrie, Fool Eater, Fragile Heart, Frozen In Time, Funky D Luffy, Galacta Warrior, Glaive Main Cannon, Godskin Apostle, Godskin Noble, Gold Fuzzy, Golden Freddy, grand doomer, Green Mage, Greenpath, Grimmchild, Groda, Heavy Lobster, Heavy Mole, Helpful Mannequin, Holo Defense API, Holy Blade, Hooktail Castle, Hornet, HR-D3, hr-h, Humphrey, Ice Dragon, Ice Staff, Infected Crossroads, Ingredient Icy Sugar Cubes, Ipsilon, Irithyll, Jetpack, Jevil, Joel Miller, John Ward, King Dedede, Koro-Sensei, Landia, Leshy, Little Guy, Living Bread, Lord Crump, Macho Grubba, Magic Shield, Mantis Lords, Mari, Marilyn, Masher, Massive Moss Charger, Master Hand, Matterling, Maxim Tomato, Meta Knight, Metal Shield, Metronome Lite, Mettaton, Mettaton EX, Mettaton NEO, Miasma, Michael*, Militron, Mini Auto Croissant, Mini Bee, Mini Drill, Monkey D Goofy, Monkey D Letion, Monkey D Pression, Monkey D Vour, Moose Goose, Morshu, Moss Charger, Mr Krabs, Mr Softener, Mushroom, Necrodeus, Nefarious Chip, Nightmare, Niiue, NME Salesman, Ominous Red Circle, Oryx the Exalted God, Overseer, P.A.N.D.A. Mk A, P.A.N.D.A. Mk T, Paint Bomb, Paint Warrior, Pan D Luffy, Pandabal, Pandamonium, Papa Chip, Pike Beam, PoD04Zebra Bloon, PoD05Lead Bloon, PoD06Rainbow Bloon, Project P.A.N.D.A, Pure Vessel, Puzzled Cat, Qiraji Soldier, Quote, Rapier, Rasta Beast, Remote Detonator, Resentful Glare, Retro Kirby, Royal Eradicator Main Cannon, Sabre, Sacred Heart, Second Wind, Shield of the Guilty, Silithid, Skullord, Smeckeldorfed, Smorg, Snake, Snow Angel, SOMETHING, Soul Master, Soul Sanctum, Spaghetti, Spawn of the Forbidden One, Spicy Spray, SpiralHorn Shield, Spirit Flame, Springtrap, Spur, Staff of Purity, Stalwart Shell, Stranger, Sumo Panda, Super Shroom, Supreme Jeh'Oul, Sword, Sword Knight, Sword of Aeons, Tainted Bethany, Tainted Keeper, Tainted Magdalene, Taranza, Termination Shock*, The Bank, The Big Yellow Cat, The Bull, The Collector, The Core*, The Farlands, The Flash, The Freezer, The Gates of Ahn'Qiraj, The Krusty Krab, The Master, The Plaguelands, The Puppet, The Queen, The Ringed City, The Shade, The Structure, The Temple of Ahn'Qiraj, Tomorrow I'll, Toriel, Tutankoopa, Unbread Twins, Uumuu, Warrior Dee, Waxwell, Whippy D Cream, White Space, Wickerbottom, Wilson, Wokey D pronouns, Z!!!!!, Zero, Zero Override, Zero Two, Zompanda

**Title prints past its old text onto the picture below (171 cards):** Anubisath Guardian, Autopsy Report, Bag of Good, Binge Eater, Blindness, Blood Bomb, Bobberlington, Brad Armstrong, Bubble, Bucket Of Lard, Burdened, Butterfly, Cabbage, Cake Mix, Cheap, Claw, Croissant, Dark Esau, Dark Mind, Dark Nebula, Delicate Flower, Disgusted, Door Shield, Dryya, Dung Defender, Dyna Blade*, Dynamite, Earth Totem, Enraged, Exposed, False Knight, Feast, Fire Staff, Firebomb, Fog Canyon, Freeze Spell, Frog, Galacta Warrior, Gandhi, Godskin Apostle, Godskin Noble, Gold Fuzzy, Golden Freddy, Green Mage, Greenpath, Grimmchild, Groda, Heavy Lobster, Helpful Mannequin, Holo Defense API, Hooktail, Hooktail Castle, Hornet, Ice Dragon, Ice Staff, Igor Roballtowski*, Igor Robotowski, Imp, Infected Crossroads, Inkhole, Ipsilon, Jetpack, Jevil, John Ward, Katarina, Koops, Koro-Sensei*, Lit Bob-omb, Living Bread, Lor Starcutter, Lord Crump, Magic Shield, Marx, Masher, Master Hand, Matterling, Maxim Tomato, Meta Knight, Metal General, Metal Shield, Mettaton, Mettaton EX, Mettaton NEO, Mini Bee, Miracle Matter, Monkey D Letion, Moose Goose, Morshu, Moss Charger, Mr Jawsum, Nefarious Chip, Nightmare, Niko, Nosk, Ominous Red Circle, Oryx the Exalted God, Overseer, Paint Bomb, Paint Warrior, Pandamonium, Party!, Peppy, Pike Beam, PoD01Red Bloon, PoD05Lead Bloon, PoD06Rainbow Bloon, PoD08MOAB, PoD11ZOMG, Pure Vessel, Puzzled Cat, Qiraji Soldier, Quote, Rapier, Rasta Beast, Red Knife, Reflection, Regen, Regret, Reinforced, Remote Detonator, Retro Kirby, Rope, Royal Eradicator Main Cannon, Rusty Knife, Sacred Heart, Sandwich of Rejects, Second Wind, Shield of the Guilty, Silithid, Snow Angel, SOMETHING, Soul Master, Spaghetti, Spice, Springtrap, Spur, Stalwart Shell, Star Dream, Supreme Jeh'Oul, Tai Lung, Tainted Bethany, Tainted Keeper, Tainted Magdalene, Tanjerin, Termination Shock, Thalassophobia, The Bank, The Farlands, The Flash, The Gates of Ahn'Qiraj, The Master, The Puppet, The Queen, The Ringed City, The Structure, Toriel, Trick Room, Uumuu, Warcraftian Mage, Warrior Dee, Water Totem, Waxwell, Whacka Bump, Whelp, Whispy Woods, Wickerbottom, Z!!!!!, Zero, Zero Override, Zoltan, Zompanda

**Stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (no group named Stars) (87 cards):** Afraid, Amazing Mirror, Blazing, Blindness, Bobbery, Bone's Loans, Burdened, Burning, Chained, Chilled, Condemned, Confusion, Contracted, Corrupted Blood, Corruption, Crystal Peak, Curse, Cursed Inferno, Dark Sword, Despair, Disgusted, E. Gadd's Garage, Empty, Enfeebled, Engorged, Exposed, Forsaken City, Frostbite, Frozen, Frozen In Time, Glitz Pit, Glued, Goombella, Grey Prince Zote, Hemorrhaging, Hexed, Hooktail, HR-D3, Imprisoned, Infected, Insane, Joel Miller, Kel, Koops, Landia, Leshy, Loony, Marked For Death, Marx, Masked Dedede, Meta Knight, Mr Jawsum, Mr Softener, Ms Mowz, Niko, Paralyzed, Pillow, Pizza, Plague, Poison, POW Block, Recursion, Resentful Glare, Restrained, Reverse Mushroom, Rock Bottom, Sandwich of Rejects, Shell Shock, Sickness, Smeckeldorfed, Smoldering, Softened, Spice, Spicy Spray, Spirit Flame, Stuffed, Suffocation, Terry Hintz, The Core, The Freezer, The Krusty Krab, Thinned, Tutankoopa, Vivian, Watermelon, Withering, Yin Yarn

**Title would print over a layer just below it (121 cards):** Abyss Watcher, Ancient Fuelweaver, Arachnophobia, Assault Panda, Asteroid Field, Auto Croissant, Azamoth the Possessed Vessel, Bag of Evil, Barry The Barrel, Battle Spear, Become Back Money, BergenTRUCK, Bobbery, Boggly Woods, Bolts and Screws, Captain Spaceboy, Cave Of The Past, Charm Of The Vampire, Corrupted Blood, Cortez, Crimson Flag, Crowned Magolor, Crystal Guardian, Crystal King, Crystal Spikes, Dark Bramble, Dark Ritual, Dark Sword, Daroach, Deepnest, Delta, Doc, DooPliss, Drawcia, Drawcia Soul*, Dumplings, Empress Bulblax, Energy Ritual, Excess Express, Flame Trap, Flower Of Decay, Flurrie, Fool Eater, Fragile Heart, Funky D Luffy, Glaive Main Cannon, grand doomer, Heavy Mole, Holy Blade, HR-D3, hr-h, Humphrey, Ingredient Icy Sugar Cubes, Irithyll, Joel Miller, King Dedede, Landia, Leshy, Little Guy, Macho Grubba, Mantis Lords, Mari, Marilyn, Massive Moss Charger, Metronome Lite, Miasma, Michael, Militron, Mini Auto Croissant, Mini Drill, Monkey D Goofy, Monkey D Pression, Monkey D Vour, Mr Krabs, Mr Softener, Mushroom, Necrodeus, Niiue, NME Salesman, P.A.N.D.A. Mk A, P.A.N.D.A. Mk T, Pan D Luffy, Pandabal, Papa Chip, PoD04Zebra Bloon, Project P.A.N.D.A, Sabre, Sans, Skullord, Smorg, Snake, Soul Sanctum, Spawn of the Forbidden One, Spicy Spray, SpiralHorn Shield, Staff of Purity, Stranger, Sumo Panda, Super Shroom, Sword, Sword Knight, Sword of Aeons, Taranza, Termination Shock*, The Big Yellow Cat, The Bull, The Collector, The Core*, The Krusty Krab, The Plaguelands, The Shade, The Temple of Ahn'Qiraj, Thunderfury Blessed Blade of the Windseeker, Tomorrow I'll, Tutankoopa, Unbread Twins, Whippy D Cream, White Space, Wilson, Wokey D pronouns, Zero Two

**Stars: laid out again as rows of two (the card's own layout doesn't go on) (41 cards):** Alex YIIK, Ancient Fuelweaver, Baba, Bobberlington, Bugzzy, Butterfly, Captain Spaceboy, Chris & Jordi, Croissant, Dark Mind Eye, Doc, Fawful, Flame Trap, Frog, Galacta Knight, God Tamer, hr-h, Humphrey, Koro-Sensei, Michael, Michael's Blessing, Morpho Knight, Necrodeus, Overseer, Paintbrush, PoD09DDT, Pure Vessel, Royal Eradicator Main Cannon, Sans, Spiky Ball Trap, Springtrap, Star Dream, The Forbidden One, The Master, The Puppet, The Queen, The Radiance, The Thin Man, Undyne, Z!!!!!, Zenith Martlet

**Stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (the bot reads another count from its Stars group's layers than the folder's, though the card prints the folder's: overlapping stars, one layer drawing several, or extra layers in the group) (16 cards):** Auto Croissant, Dimension D, Grimora, Kyle, Nightmare, Nosk, Papyrus, Tainted Bethany, Tainted Eden, Tainted Lazarus, The Flash, The Lich King, Toy Chica, Zero Override, Zero Two, Zoltan

**A layer above the title would cover part of the new title (20 cards):** Chained, Condemned, Cursed Inferno, Diligent Watch, Doctor Zomboss*, Frozen In Time, Leshy*, Massive Moss Charger, Mosling, Moss Charger, Mr Jawsum, Resentful Glare, Smeckeldorfed, Spirit Flame, The Big Yellow Cat, The Farlands, The Freezer, The Structure, Wickerbottom, WX-78

**Types: the first type icon, at the card template's spot, touches the title (14 cards):** Helpful Mannequin, John Ward, King Dedede, Knight Lost A Lot, Leeroy Jenkins, Miles Edgeworth, Militron (No Armor), Stay, The Forbidden One, Warcraftian Druid, Warcraftian Hunter, Warcraftian Rogue, Warcraftian Warrior, Wokey D pronouns

**Title runs off the card (11 cards):** Crowned Magolor, Fawful, Helpful Mannequin, Indoctrination, King Dedede, Kyle, Moose Goose, Mosling*, Stay, Wickerbottom, Wokey D pronouns

**Title runs past an effect layer clipped to it (10 cards):** Air Totem, Cake Mix, Claw, Dogma, Earth Totem, Fire Totem, Miasma, Paint Bomb, Water Totem, Yin Yarn

**Ability runs off the card (10 cards):** Bugzzy, Dumplings, Holy Blade, Ice Storm, Krabby Patty, Love, Nebulous Precipice, Pile Of Swords, Silithid, Whelp

**Ability runs over a stat bar (9 cards):** Anubisath Guardian, Electro Probe, Galacta Warrior, Mini Bee, Paint Warrior, Qiraji Soldier, Silithid, Tentacle, Whelp

**Ability prints over the picture or an effect layer (6 cards):** Claw, Iron Spikes, Miasma, Paint Warrior, Remote Detonator, Zompanda

**Title needs more lines than its box has (4 cards):** Basic Shield, Computer Virus, Leeroy Jenkins, Pot Full of Spiders!*

**A character of the new text has no glyph in the layer's font (drawn as a box) (4 cards):** Mini Bee, Paint Warrior, Sword Knight, Tentacle

**Stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (a star layer has smart filters) (3 cards):** Doctor Zomboss, Holo Defense API, Newspaper

**Stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (no room for another star) (2 cards):** Mantis Lords, Tyler

**A styled word in the ability loses its style (2 cards):** Flameblade, Flower Of Decay

**Stat bar: a missing digit was made from the one below it (1 card):** Spinni

**A font the text uses isn't in MDW/Fonts: drawn in a stand-in (1 card):** Dark Mind Eye

**Stars: the edit moves the card to the other "N Stars" folder, but its printed stars aren't redrawn (layout) (1 card):** Cloud Soiree

**Overflow: ability: set at 94% of its size to stay on its panel (1 card):** Chilled
