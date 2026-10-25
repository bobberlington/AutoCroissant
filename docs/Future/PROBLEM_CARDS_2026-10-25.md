# Problem cards, 2026-10-25

Cards are not required to have types or an ability: cards without them are not listed.

- **124 of 764 cards** have at least one website edit that can't bake (a real refusal; changing stars on 575 cards whose stars come from their "N Stars" folder is a folder move, not a bake, and is not counted).
- **4 cards** have an open data problem (Computer Virus has no stats; Twin Emperors' second creature has a different DEF on the website than on the card; 2 cards stuck under "Not parsed yet"); 5 more are ruled acceptable on purpose.
- **308 cards** bake some edit only with a note an editor must look at; nearly all are a title made 3 letters longer reaching the stars, type icons or picture. Only 22 get such a note from a same-length title fix or a one-word ability change.

Source: dry bakes of 11 kinds of edit over all 764 cards (6,712 bakes, AutoCroissant 429adb6, TTSCardMaker cacabfca, nothing written), plus a check of the card data and live website (2026-10-25 05:41 UTC). A card is listed under each reason it hits, so some cards appear more than once. Spot-check: I re-ran the dry bake for 5 refused cards (Alex YIIK title, Clot Black HP, Sonic ability, Magnum Opus removing a type, Lorian HP) and all 5 refused again for the same reason.

> **Being fixed in code (2026-10-25), so no Photoshop work is needed just for baking:**
> - Titles and abilities printed by several layers, split titles, titles named oddly, titles printed twice, several or no visible Ability layers: the website is getting **one text box per layer** (and copies/watermarks of the title follow the main title). Titles that are **pictures** still need retyping as text.
> - **Stars set by the 'N Stars' folder** (575 cards, not counted below): a stars edit will move the card into the matching folder.
> - **Tab stops** in text are being added to the bake.
>
> - **Numbered PNGs sharing one PSD** (5 cards): variant tags (` #N` on layer names), then the bot handles every picture.
>
> Still for card makers or later code work: non-template stat bars, unusual star layouts, picture titles, Sonic's one-row SPD bar.

## 1. Website edits that can't bake

Grouped by the part of the card. Each card shows its folder and the edits the reason blocks. "Who" is who would have to act for the edit to bake.

### Whole card

#### Several card pictures (numbered PNGs such as A_Snack2.png) come from this one PSD (5 cards)

- **Fix (owner, 2026-10-25): variant tags.** Tag each variant's own layers and groups with ` #N` at the end of the name (e.g. `Snack #1`, `Breakfast #2`, `Supper #3`; ` #2,3` for several). Untagged layers are in every variant. Variant 1 is `Name.png`, variant N is `NameN.png`, as today. The bot then renders, bakes and checks every picture from the one PSD (being built; the suggested tags for each card will be worked out from today's PNGs).
- **Who:** Card maker in Photoshop (the tags), then bot code

  - A Snack (Items/Heal/2 Stars): title, ability, stars (add and remove), types (add and remove)
  - All Purpose Medical Kit (Items/Heal/2 Stars): title, ability, stars (add and remove), types (add and remove)
  - NME Stock (Items/Support/2 Stars): title, ability, stars (add and remove), types (add and remove)
  - Regeneration Potion (Items/Heal/2 Stars): title, stars (add and remove), types (add and remove)
  - Watermelon (Items/Heal/2 Stars): title, ability, stars (add and remove), types (add and remove)

#### Sonic's SPD 20 is printed on a one-row bar (1 card)

- **Fix:** The bar shows SPD above 10 without the template's second row, so the read-back after any edit disagrees and every edit is refused; SPD itself needs a second bar row. Rebuild SPD as the template's two-row bar.
- **Who:** Card maker in Photoshop (standardisation)

  - Sonic (Creatures/Other/4 Stars): ability, all four stats, types (add and remove)

### Title

#### The title is printed by more than one layer (9 cards)

- **Fix:** Merge the title into one text layer named Name (line breaks inside one layer are fine), or edit the title in Photoshop.
- **Who:** Card maker in Photoshop

  - Azamoth the Forgotten King (Creatures/ROTMG/5 Stars): title
  - Azamoth the Possessed Vessel (Auxiliary/Creatures): title
  - Massive Moss Charger (Auxiliary/Minions): title
  - Oryx the Exalted God (Creatures/ROTMG/4 Stars): title
  - The Bagel (N.M.E): title
  - The Big Yellow Cat (Creatures/Omori/5 Stars): title
  - The Gates of Ahn'Qiraj (Field/4 Stars): title
  - The Temple of Ahn'Qiraj (Auxiliary/Items): title
  - Warcraftian Shaman (Creatures/World Of Warcraft/4 Stars): title

#### No title layer the bot recognises (9 cards)

- **Fix:** Make the title one visible text layer named Name inside the Text group.
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

#### The title is split over several layers (8 cards)

- **Fix:** Merge the title into one text layer named Name.
- **Who:** Card maker in Photoshop

  - Crowned Magolor (Creatures/Kirby/5 Stars): title
  - Royal Eradicator Main Cannon (Creatures/FTL/5 Stars): title
  - Warcraftian Druid (Creatures/World Of Warcraft/4 Stars): title
  - Warcraftian Hunter (Creatures/World Of Warcraft/4 Stars): title
  - Warcraftian Paladin (Creatures/World Of Warcraft/4 Stars): title
  - Warcraftian Priest (Creatures/World Of Warcraft/4 Stars): title
  - Warcraftian Rogue (Creatures/World Of Warcraft/4 Stars): title
  - WX-78 (Creatures/DST/3 Stars): title

#### The title's words are printed twice (another layer repeats them) (5 cards)

- **Fix:** Delete or hide the duplicate layer, or edit the title in Photoshop.
- **Who:** Card maker in Photoshop

  - Autopsy Report (Auxiliary/Items): title
  - Moose Goose (Creatures/DST/3 Stars): title
  - Mosling (Auxiliary/Minions): title
  - Tainted Eden (Creatures/Isaac/2 Stars): title
  - Wickerbottom (Creatures/DST/3 Stars): title

#### The title layer has another name (not Name, the card's name or its words) (5 cards)

- **Fix:** Rename that one layer to Name. (The owner chose on 2026-10-22 to leave layer names as they are; these are the few the bot still can't match.)
- **Who:** Card maker in Photoshop

  - Alex YIIK (Creatures/Other/5 Stars): title; layer 'this card fucking sucks'
  - Chris & Jordi (Creatures/Everhood/5 Stars): title; layer 'Knight'
  - Lorian (Auxiliary/Minions): title; layer 'Twin Princes'
  - Lothric (Creatures/Souls/5 Stars): title; layer 'Twin Princes'
  - Thunder Thighs (Items/Heal/2 Stars): title; layer 'Snack'

#### Two layers in the title area look like the title (2 cards)

- **Fix:** Merge them into one text layer named Name.
- **Who:** Card maker in Photoshop

  - Dark Matter Swordsman (Creatures/Kirby/3 Stars): title
  - Knight Lost A Lot (Creatures/Everhood/2 Stars): title

#### The title layer is hidden (4 cards)

- **Fix:** Delete the hidden leftover Name layer; the printed title looks like part of the picture, so retype it as a visible text layer named Name to make it editable.
- **Who:** Card maker in Photoshop

  - E. Gadd's Garage (Field/2 Stars): title
  - Man Serpent Shield (Items/Equip/2 Stars): title
  - Sonic (Creatures/Other/4 Stars): title
  - Wigfrid (Creatures/DST/3 Stars): title

#### The bot's drawing of the new title is more than 25% off the card picture (4 cards)

- **Fix:** The bot's renderer draws this title far from how Photoshop drew it, so it won't risk the bake. Improve the renderer for these title styles, or edit the title in Photoshop.
- **Who:** Bot code (renderer); until then edit in Photoshop

  - Dark Tentacle (Auxiliary/Minions): title
  - Dull Knife (Auxiliary/Items): title
  - Shiny Knife (Items/Equip/3 Stars): title
  - Snow Angel (Creatures/Omori/3 Stars): title (typo fix only)

### Ability

#### Several visible ability layers (5 cards)

- **Fix:** Merge them into one text layer named Ability, or edit the ability in Photoshop.
- **Who:** Card maker in Photoshop

  - Lit Bob-omb (Auxiliary/Items): ability
  - Miracle Matter (Creatures/Kirby/3 Stars): ability
  - Tainted Lazarus (Creatures/Isaac/3 Stars): ability
  - Twin Emperors (Creatures/World Of Warcraft/4 Stars): ability
  - Warcraftian Druid (Creatures/World Of Warcraft/4 Stars): ability

#### No visible ability layer (4 cards)

- **Fix:** Make the printed ability one visible text layer named Ability.
- **Who:** Card maker in Photoshop

  - Apex Predator (Creatures/R.E.P.O/2 Stars): ability
  - Bulborb Larva (Auxiliary/Minions): ability
  - Butterfly (Creatures/Kirby/5 Stars): ability
  - Empress Bulblax (Creatures/Other/3 Stars): ability

#### The ability's words are printed twice (1 card)

- **Fix:** Delete or hide the duplicate layer.
- **Who:** Card maker in Photoshop

  - SOMETHING (Creatures/Omori/3 Stars): ability

#### The ability's icons are pictures in gaps of the text, so the edited ability doesn't read back as typed (1 card)

- **Fix:** Replace the picture icons with icon-font tokens (like [fire]).
- **Who:** Card maker in Photoshop

  - Nebulous Precipice (Field/3 Stars): ability

### Stats (HP/DEF/ATK/SPD)

#### The stat bar is not the card template's bar (31 cards)

- **Fix:** Replace the bar row with the template's row.
- **Who:** Card maker in Photoshop

  - Anubisath Guardian (Auxiliary/Minions): HP
  - Blank Card (Items/Active/3 Stars): HP/DEF
  - Bulborb Larva (Auxiliary/Minions): all four stats
  - Chef Saltbaker (Creatures/Cuphead/5 Stars): all four stats
  - Chris & Jordi (Creatures/Everhood/5 Stars): all four stats
  - Clot Black (Auxiliary/Minions): all four stats
  - Clot Blood (Auxiliary/Minions): all four stats
  - Clot Bone (Auxiliary/Minions): all four stats
  - Clot Gold (Auxiliary/Minions): all four stats
  - Clot White (Auxiliary/Minions): all four stats
  - Dark Esau (Auxiliary/Minions): all four stats
  - Dark Mind (Creatures/Kirby/4 Stars): DEF
  - Dirty Bed (Items/Active/2 Stars): HP/DEF
  - Empress Bulblax (Creatures/Other/3 Stars): all four stats
  - Grimora (Creatures/Other/3 Stars): all four stats
  - Minecraft Panda (Creatures/Pandas/2 Stars): all four stats
  - Pizza (Creatures/Other/3 Stars): all four stats
  - Skeleton Mate (Auxiliary/Minions): all four stats
  - Slugcat (Creatures/Other/2 Stars): all four stats
  - Spinni (Creatures/Kirby/3 Stars): all four stats
  - Tainted Bethany (Creatures/Isaac/4 Stars): all four stats
  - Tainted Eden (Creatures/Isaac/2 Stars): all four stats
  - Tainted Eve (Creatures/Isaac/4 Stars): all four stats
  - Tainted Isaac (Creatures/Isaac/3 Stars): all four stats
  - Tainted Jacob (Creatures/Isaac/3 Stars): all four stats
  - Tainted Keeper (Creatures/Isaac/4 Stars): all four stats
  - Tainted Lost (Creatures/Isaac/2 Stars): all four stats
  - Tainted Magdalene (Creatures/Isaac/3 Stars): all four stats
  - Uumuu (Creatures/Hollow Knight/3 Stars): all four stats
  - Whispy Woods (Creatures/Kirby/2 Stars): all four stats
  - Zigg (Creatures/Everhood/3 Stars): all four stats

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

#### Safety check: the stat number's layer style would not move cleanly to the new number (4 cards)

- **Fix:** Probably a bot bug: the style on the stat number changes in a way the bake didn't plan. Needs a look in the bot.
- **Who:** Bot code

  - Lorian (Auxiliary/Minions): HP
  - Mettaton EX (Auxiliary/Creatures): DEF
  - Tutankoopa (Creatures/Mario/3 Stars): HP
  - Vivian (Creatures/Mario/3 Stars): HP

### Stars

#### The stars are laid out in a shape the bot can't follow (not a row, column or rows of two) (13 cards)

- **Fix:** Lay the stars out in a row or column, or add/remove stars in Photoshop. Bot code could learn more shapes (not planned).
- **Who:** Card maker in Photoshop

  - God Tamer (Auxiliary/Minions): adding a star
  - Jack-In-The-Box Zombie (Auxiliary/Minions): adding a star
  - Lorian (Auxiliary/Minions): adding a star
  - Magnum Opus (Auxiliary/Items): adding a star
  - Mettaton NEO (Auxiliary/Creatures): adding a star
  - Michael's Blessing (N.M.E): adding a star
  - Morpho Knight (Auxiliary/Creatures): adding a star
  - Paintbrush (N.M.E): adding a star
  - PoD04Zebra Bloon (Auxiliary/Minions): adding a star
  - PoD09DDT (Auxiliary/Minions): adding a star
  - Polevaulter Zombie (Auxiliary/Minions): adding a star
  - Springtrap (Auxiliary/Minions): adding a star
  - Termination Shock (N.M.E): adding a star

#### One layer draws several stars (3 cards)

- **Fix:** Give each star its own layer.
- **Who:** Card maker in Photoshop

  - Cat (Auxiliary/Minions): stars (add and remove)
  - Living Bread (Auxiliary/Minions): stars (add and remove)
  - Shadow Queen (Auxiliary/Creatures): stars (add and remove)

#### No single group named Stars draws them (3 cards)

- **Fix:** Put all the stars in one group named Stars.
- **Who:** Card maker in Photoshop

  - Nasty Patty (Auxiliary/Items): stars (add and remove)
  - Paint Warrior (Auxiliary/Minions): stars (add and remove)
  - Shell Shield (Auxiliary/Items): stars (add and remove)

#### A new star would touch the title or icons (1 card)

- **Fix:** No room for another star; edit in Photoshop.
- **Who:** Nothing (edit in Photoshop)

  - Bugzzy (Auxiliary/Minions): adding a star

#### A star layer has smart filters, so it can't be moved (1 card)

- **Fix:** Rasterise the star layer (or drop its smart filter).
- **Who:** Card maker in Photoshop

  - Magnet-shroom (N.M.E): adding a star

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

- **Stars on 575 cards** come from the card's "N Stars" folder, not from star layers in the picture; changing them means moving the card to another folder, not baking.
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

- **Fix:** The card data treats each as a copy of another card, but the website only recognises numbered copies (like A Snack2), so no card sync will clear the "Not parsed yet" note. Teach the website these copies, or give each card its own PSD.
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

These edits do bake; the bake carries a note for an editor. Most title notes come from the census's rename, which makes the title 3 letters longer (" II"); a shorter title change may bake clean. Cards marked * got the note even on a same-length typo fix.

| Note | Edits | Cards | Editor must look? |
|---|---|---|---|
| title reaches the stars or type icons | 232 | 228 | yes |
| title prints past its old text onto the picture below | 168 | 165 | yes |
| title would print over a layer just below it | 118 | 116 | yes |
| a layer above the title would cover part of the new title | 15 | 15 | yes |
| title runs past an effect layer clipped to it | 10 | 10 | yes |
| title runs off the card | 7 | 7 | yes |
| ability runs off the card | 7 | 7 | yes |
| ability prints over the picture or an effect layer | 5 | 5 | yes |
| title needs more lines than its box has | 4 | 3 | yes |
| a styled word in the ability loses its style | 1 | 1 | yes |
| ability redrawn whole, small pixel difference | 14 | 14 | no (pixel or font note) |
| type icons baked with a small pixel difference | 3 | 3 | no (pixel or font note) |
| title's saved pixels don't match its named font | 1 | 1 | no (pixel or font note) |
| title redrawn whole, small pixel difference | 1 | 1 | no (pixel or font note) |

**Title reaches the stars or type icons (228 cards):** Abyss Watcher, Ancient Fuelweaver, Anubisath Guardian, Arachnophobia, Assault Panda, Asteroid Field, Auto Croissant, Barry The Barrel, Battle Spear, Become Back Money, BergenTRUCK, Binge Eater, Blood Bomb, Bobberlington, Bobbery, Boggly Woods, Bolts and Screws, Brad Armstrong, Bucket Of Lard, Butterfly, Cabbage, Captain Spaceboy, Cave Of The Past, Chained, Charm Of The Vampire, Claw, Condemned, Corrupted Blood, Cortez, Crimson Flag, Croissant, Crystal Guardian, Crystal King, Crystal Spikes, Cursed Inferno, Dark Bramble, Dark Mind, Dark Nebula, Dark Ritual, Dark Sword, Daroach, Deepnest, Delicate Flower, Delta, Diligent Watch, Doc, Dogma, DooPliss, Door Shield, Drawcia, Drawcia Soul*, Dryya, Dumplings, Dyna Blade, Dynamite, Empress Bulblax, Excess Express, False Knight, Feast, Fire Staff, Flower Of Decay, Flurrie, Fool Eater, Fragile Heart, Frozen In Time, Funky D Luffy, Galacta Warrior, Glaive Main Cannon, Godskin Apostle, Godskin Noble, Gold Fuzzy, Golden Freddy, grand doomer, Green Mage, Greenpath, Grimmchild, Groda, Heavy Lobster, Heavy Mole, Helpful Mannequin, Holo Defense API, Holy Blade, Hooktail Castle, Hornet, HR-D3, hr-h, Humphrey, Ice Dragon, Ice Staff, Infected Crossroads, Ingredient Icy Sugar Cubes, Ipsilon, Irithyll, Jetpack, Jevil, Joel Miller, John Ward, King Dedede, Koro-Sensei, Landia, Leshy, Little Guy, Living Bread, Lord Crump, Macho Grubba, Magic Shield, Mantis Lords, Mari, Marilyn, Masher, Master Hand, Matterling, Maxim Tomato, Meta Knight, Metal Shield, Metronome Lite, Mettaton, Mettaton EX, Mettaton NEO, Miasma, Michael*, Militron, Mini Auto Croissant, Mini Bee, Mini Drill, Monkey D Goofy, Monkey D Letion, Monkey D Pression, Monkey D Vour, Morshu, Moss Charger, Mr Krabs, Mr Softener, Mushroom, Necrodeus, Nefarious Chip, Nightmare, Niiue, NME Salesman, Ominous Red Circle, Overseer, P.A.N.D.A. Mk A, P.A.N.D.A. Mk T, Paint Bomb, Paint Warrior, Pan D Luffy, Pandabal, Pandamonium, Papa Chip, Pike Beam, PoD04Zebra Bloon, PoD05Lead Bloon, PoD06Rainbow Bloon, Project P.A.N.D.A, Pure Vessel, Puzzled Cat, Qiraji Soldier, Quote, Rapier, Rasta Beast, Remote Detonator, Resentful Glare, Retro Kirby, Sabre, Sacred Heart, Second Wind, Shield of the Guilty, Silithid, Skullord, Smeckeldorfed, Smorg, Snake, Snow Angel, SOMETHING, Soul Master, Soul Sanctum, Spaghetti, Spawn of the Forbidden One, Spicy Spray, SpiralHorn Shield, Spirit Flame, Springtrap, Spur, Staff of Purity, Stalwart Shell, Stranger, Sumo Panda, Super Shroom, Supreme Jeh'Oul, Sword, Sword Knight, Sword of Aeons, Tainted Bethany, Tainted Keeper, Tainted Magdalene, Taranza, Termination Shock*, The Bank, The Bull, The Collector, The Core*, The Farlands, The Flash, The Freezer, The Krusty Krab, The Master, The Plaguelands, The Puppet, The Queen, The Ringed City, The Shade, The Structure, Tomorrow I'll, Toriel, Tutankoopa, Unbread Twins, Uumuu, Warrior Dee, Waxwell, Whippy D Cream, White Space, Wilson, Wokey D pronouns, Z!!!!!, Zero, Zero Override, Zero Two, Zompanda

**Title prints past its old text onto the picture below (165 cards):** Anubisath Guardian, Bag of Good, Binge Eater, Blindness, Blood Bomb, Bobberlington, Brad Armstrong, Bubble, Bucket Of Lard, Burdened, Butterfly, Cabbage, Cake Mix, Cheap, Claw, Croissant, Dark Esau, Dark Mind, Dark Nebula, Delicate Flower, Disgusted, Door Shield, Dryya, Dung Defender, Dyna Blade*, Dynamite, Earth Totem, Enraged, Exposed, False Knight, Feast, Fire Staff, Firebomb, Fog Canyon, Freeze Spell, Frog, Galacta Warrior, Gandhi, Godskin Apostle, Godskin Noble, Gold Fuzzy, Golden Freddy, Green Mage, Greenpath, Grimmchild, Groda, Heavy Lobster, Helpful Mannequin, Holo Defense API, Hooktail, Hooktail Castle, Hornet, Ice Dragon, Ice Staff, Igor Roballtowski*, Igor Robotowski, Imp, Infected Crossroads, Inkhole, Ipsilon, Jetpack, Jevil, John Ward, Katarina, Koops, Koro-Sensei*, Lit Bob-omb, Living Bread, Lor Starcutter, Lord Crump, Magic Shield, Marx, Masher, Master Hand, Matterling, Maxim Tomato, Meta Knight, Metal General, Metal Shield, Mettaton, Mettaton EX, Mettaton NEO, Mini Bee, Miracle Matter, Monkey D Letion, Morshu, Moss Charger, Mr Jawsum, Nefarious Chip, Nightmare, Niko, Nosk, Ominous Red Circle, Overseer, Paint Bomb, Paint Warrior, Pandamonium, Party!, Peppy, Pike Beam, PoD01Red Bloon, PoD05Lead Bloon, PoD06Rainbow Bloon, PoD08MOAB, PoD11ZOMG, Pure Vessel, Puzzled Cat, Qiraji Soldier, Quote, Rapier, Rasta Beast, Red Knife, Reflection, Regen, Regret, Reinforced, Remote Detonator, Retro Kirby, Rope, Rusty Knife, Sacred Heart, Sandwich of Rejects, Second Wind, Shield of the Guilty, Silithid, Snow Angel, SOMETHING, Soul Master, Spaghetti, Spice, Springtrap, Spur, Stalwart Shell, Star Dream, Supreme Jeh'Oul, Tai Lung, Tainted Bethany, Tainted Keeper, Tainted Magdalene, Tanjerin, Termination Shock, Thalassophobia, The Bank, The Farlands, The Flash, The Master, The Puppet, The Queen, The Ringed City, The Structure, Toriel, Trick Room, Uumuu, Warcraftian Mage, Warrior Dee, Water Totem, Waxwell, Whacka Bump, Whelp, Whispy Woods, Z!!!!!, Zero, Zero Override, Zoltan, Zompanda

**Title would print over a layer just below it (116 cards):** Abyss Watcher, Ancient Fuelweaver, Arachnophobia, Assault Panda, Asteroid Field, Auto Croissant, Bag of Evil, Barry The Barrel, Battle Spear, Become Back Money, BergenTRUCK, Bobbery, Boggly Woods, Bolts and Screws, Captain Spaceboy, Cave Of The Past, Charm Of The Vampire, Corrupted Blood, Cortez, Crimson Flag, Crystal Guardian, Crystal King, Crystal Spikes, Dark Bramble, Dark Ritual, Dark Sword, Daroach, Deepnest, Delta, Doc, DooPliss, Drawcia, Drawcia Soul*, Dumplings, Empress Bulblax, Energy Ritual, Excess Express, Flame Trap, Flower Of Decay, Flurrie, Fool Eater, Fragile Heart, Funky D Luffy, Glaive Main Cannon, grand doomer, Heavy Mole, Holy Blade, HR-D3, hr-h, Humphrey, Ingredient Icy Sugar Cubes, Irithyll, Joel Miller, King Dedede, Landia, Leshy, Little Guy, Macho Grubba, Mantis Lords, Mari, Marilyn, Metronome Lite, Miasma, Michael, Militron, Mini Auto Croissant, Mini Drill, Monkey D Goofy, Monkey D Pression, Monkey D Vour, Mr Krabs, Mr Softener, Mushroom, Necrodeus, Niiue, NME Salesman, P.A.N.D.A. Mk A, P.A.N.D.A. Mk T, Pan D Luffy, Pandabal, Papa Chip, PoD04Zebra Bloon, Project P.A.N.D.A, Sabre, Sans, Skullord, Smorg, Snake, Soul Sanctum, Spawn of the Forbidden One, Spicy Spray, SpiralHorn Shield, Staff of Purity, Stranger, Sumo Panda, Super Shroom, Sword, Sword Knight, Sword of Aeons, Taranza, Termination Shock*, The Bull, The Collector, The Core*, The Krusty Krab, The Plaguelands, The Shade, Thunderfury Blessed Blade of the Windseeker, Tomorrow I'll, Tutankoopa, Unbread Twins, Whippy D Cream, White Space, Wilson, Wokey D pronouns, Zero Two

**A layer above the title would cover part of the new title (15 cards):** Chained, Condemned, Cursed Inferno, Diligent Watch, Frozen In Time, Leeroy Jenkins, Leshy*, Moss Charger, Mr Jawsum, Resentful Glare, Smeckeldorfed, Spirit Flame, The Farlands, The Freezer, The Structure

**Title runs past an effect layer clipped to it (10 cards):** Air Totem, Cake Mix, Claw, Dogma, Earth Totem, Fire Totem, Miasma, Paint Bomb, Water Totem, Yin Yarn

**Title runs off the card (7 cards):** Fawful, Helpful Mannequin, Indoctrination, King Dedede, Kyle, Stay, Wokey D pronouns

**Ability runs off the card (7 cards):** Chilled, Dumplings, Holy Blade, Ice Storm, Krabby Patty, Love, Pile Of Swords

**Ability prints over the picture or an effect layer (5 cards):** Claw, Iron Spikes, Miasma, Remote Detonator, Zompanda

**Title needs more lines than its box has (3 cards):** Basic Shield, Computer Virus, Pot Full of Spiders!*

**A styled word in the ability loses its style (1 card):** Flameblade
