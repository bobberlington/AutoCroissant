# Unbakeable cards (2026-10-23)

Baking is the bot writing a website or Discord edit of a card's name, stats or ability text into the card's PSD and
re-exporting its PNG with our psd-tools fork (`commands/psd_edit.py`, `commands/card_bake.py`). This list says, for every
card and field, whether a bake works, why not, and what would fix it.

- **Cards:** every card PSD on TTSCardMaker origin/main (`71ddebec`): 764 cards. The 48 rulebook pages are left out.
- **Today** = AutoCroissant main `e837f68` (deployed) + psd-tools fork tag `cards-20261022b`.
- **After** = three pending changes on top:
  - **soft parity** (owner ruling 2026-10-23, being built on `work/soft-parity`): a pixel miss no longer blocks; the
    bake goes ahead with a note. Only a grossly broken render (more than 25% of the box off) or a read-back that doesn't
    match the edit refuses.
  - **colour glyphs** (round 6: fork `r6/integration` e6539a2f + AutoCroissant `work/r6-color-glyphs` eaf446e): built
    and reviewed, waiting for a fork tag and a deploy.
  - **refused-layers** (`work/refused-layers`, in progress): bake the title layer the parser already reads, and a single
    visible Ability layer wherever it sits. Measured on its uncommitted code of 02:51; the branch has committed
    c5c3f19..dc2bd69 since (03:35-03:38) and is still being built, so these numbers are preliminary.
- Everything was checked with the bake's own code, dry: nothing was written to either repo (section 6).

**Since (release check, 2026-10-23, fork `cards-20261023`, TTSCardMaker `0bc4c099` after the badges re-run):**

- refused-layers also bakes a title named after the card outside Text (`7176d6f`): the 58 "no Text group" titles plus
  Ms Mowz, Soul Master, Traitor Lord and The Radiance. Name after all pending changes: about **708 / 56** (not 646/118).
- Section 4: Metal General, Star Dream, Zero Override, Masked Dedede and Dark matter are now redundant (the PSD already
  prints the requested title): nothing to bake. Only PoD11ZOMG bakes ('Z.O.M.G' -> 'Z.O.M.G.', one glyph spliced).
- Section 3: Frog's and Masked Dedede's Name boxes are within the bar on the release fork (no pixel note).
- Doctor Zomboss bakes (2.11 lists it with the split titles): the edit goes into Text/Name, the only visible copy; its
  'images/Name' under the art keeps the old title inside the PSD.
- Ability bakes gave the edited words the layer's commonest colour (Pizza's '1' -> '2' went from yellow to cyan) and,
  with edits far apart, the style of the first changed character to everything between (badges drawn as text). Fixed:
  each changed part takes the style of what it replaces (`psd_edit.edit_styles`, `splice_text`).

**Stars and types (`work/stars-types-bake`, 2026-10-23):** the bake prints them (`psd_edit.plan_stars`,
`plan_types`); a layout it can't follow is refused for that field alone, the card's other fields still bake. Measured
dry with every gate on copies of the frozen corpus (`corpus-6e50c3ee`, 764 cards; fork `cards-20261023`; script and
CSV in `~/Documents/Tools/wt/scratch/followups/stars-types/census-bake/`):

| Edit | Cards | Bake | Refused (why) |
|---|---|---|---|
| stars + 1 | 764 | **146** | 592 the `N Stars` folder sets them (creatures 297, items 182, status effects 64, fields 49: a move, ledger S9); 22 a star layout it can't follow (14 two-over-three, quincunx, diamond and other shapes; 3 several stars in one group; 2 no Stars group; 1 two Stars groups; 1 smart filters; 1 would touch the title); 4 a stale Txt2 story (`txt2_repair` first) |
| stars - 1 | 144 (2+ stars) | **132** | 6 layout, 6 stale Txt2 |
| types add | 764 | **669** | 77 no type icon to line one up with; 4 off the canvas; 7 would touch the title or stars; 6 stale Txt2; 1 Sonic's stats read above 10 |
| types remove | 687 (1+ icon) | **674** | 2 off the canvas; 1 a frame (Magnum Opus's field); 2 render past the 25% floor (Tainted Bethany, Dark Mind Eye); 1 changes pixels outside its box (Crowned Magolor); 6 stale Txt2; 1 Sonic |
| types swap | 687 | 685 planned (not baked) | 1 frame, 1 off the canvas |

Of the 172 cards whose stars the parser counts from the PSD, 146 bake a star more and a star fewer, 159 one of the
two; 665 cards bake both a type added and one removed, 678 one of them. 21 bakes carry a pixel note. No card prints
its type row as icon tokens ([fire] in header text: 0 on the corpus and on TTSCardMaker `96d61368`), so a token row is
refused (`types: printed as icon tokens`). A bake took 9 s (median; 45 s at most on a loaded Mac), two or three
whole-card renders of it for `gate_outside`.

How to read it: section 1 is the one-screen summary. Section 2 has one part per reason: a plain explanation, what would
fix it, and the cards (alphabetical, with their folder). A card counts once per field, under the first reason the bake
stops at today; the last column says what the pending changes do to it. A card can sit in several parts, one per field.
Section 3 lists the boxes that bake but with a known pixel difference. The CSV beside this file
(`UNBAKEABLE_CARDS_2026-10-23.csv`) has one row per card and field that is not a clean bake.

## 1. Summary

| Field | Cards with the field | Today: bake / can't | After soft parity + colour glyphs | After also refused-layers (preliminary) |
|---|---|---|---|---|
| Name | 764 | **486** / 278 | **528** (12 with a pixel note) / 236 | **646** (21 with a pixel note) / 118 |
| Ability | 764 | **382** / 382 | **617** (10 with a pixel note) / 147 | **701** (13 with a pixel note, 1 not measured) / 63 |
| HP | 417 | **372** / 45 | **376** (4 with a pixel note) / 41 | **376** (4 with a pixel note) / 41 |
| DEF | 417 | **372** / 45 | **376** (4 with a pixel note) / 41 | **376** (4 with a pixel note) / 41 |
| ATK | 395 | **353** / 42 | **357** (4 with a pixel note) / 38 | **357** (4 with a pixel note) / 38 |
| SPD | 395 | **352** / 43 | **356** (4 with a pixel note) / 39 | **356** (4 with a pixel note) / 39 |

- **Whole cards:** 275 of 764 cards bake every field they have today; 573 will after the pending changes.
- **Stars and types** baked on no card on this list's day (`psd_edit.NOT_YET`; a pending one refused the whole card).
  Since `work/stars-types-bake` (above) they bake, and a refused one is refused for its field alone.
- **Stats** count only the cards that show that stat (items, fields and status effects without stats are left out).
- **Not counted here** (they depend on the edit, not the card): a stat above 10 (only Royal Eradicator Main Cannon's
  ATK has the two rows for it), new text that no longer fits (section 5), and a card pushed in the last 30 minutes
  (the bake tries again later).

Why, per field (today → after all pending changes; section 2 has the cards):

| Why (section 2) | Name | Ability | HP | DEF | ATK | SPD | Fixed by |
|---|---|---|---|---|---|---|---|
| Several PNGs come from one PSD | 5 → 5 | 5 → 5 |  |  |  |  | card maker (a PSD per PNG) |
| The stats come from a creatures list |  |  | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 | nothing to bake (edit the creatures list) |
| Photoshop's copy of the text is out of date | 4 → 4 | 4 → 4 | 4 → 4 | 4 → 4 | 4 → 4 | 4 → 4 | txt2_repair (one command) |
| No Text group | 69 → 58 |  |  |  |  |  | refused-layers fixes 11 of 69 |
| No title layer the bot recognizes | 52 → 12 |  |  |  |  |  | refused-layers fixes 38 of 52 |
| The title layer has another name | 90 → 10 |  |  |  |  |  | refused-layers fixes 79 of 90 |
| The Name layer is hidden | 4 → 4 |  |  |  |  |  | card maker |
| More than one layer prints the title | 0 → 9 |  |  |  |  |  | Photoshop |
| The title is split over several layers | 0 → 8 |  |  |  |  |  | card maker (one layer) |
| The title's words are printed twice | 0 → 6 |  |  |  |  |  | Photoshop, or relax the check |
| The title would touch the stars or type icons | 1 → 2 |  |  |  |  |  | Photoshop |
| The whole title doesn't fit into one of its layers | 11 → 0 |  |  |  |  |  | card maker (one title layer) |
| The Ability layer is outside the Text group |  | 92 → 0 |  |  |  |  | refused-layers fixes 85 of 92 |
| No visible Ability layer |  | 7 → 7 |  |  |  |  | card maker |
| Several visible Ability layers |  | 5 → 5 |  |  |  |  | card maker |
| The ability's words are printed twice |  | 0 → 1 |  |  |  |  | card maker |
| The ability's icons are pictures |  | 8 → 15 |  |  |  |  | card maker (stoogedings tokens) |
| A badge layer would be lost (BUG 2) |  | 18 → 18 |  |  |  |  | BUG 2 fix (approved, not started) |
| A word layer the status chart has no style for |  | 8 → 8 |  |  |  |  | stays refused (owner, 10-21); Goofy: BUG 2 fix |
| Colour icons would turn one colour |  | 213 → 0 |  |  |  |  | colour glyphs (built, not released) |
| No template stat bar |  |  | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 | card maker (template bars) |
| The stat bar is not the template's |  |  | 30 → 30 | 30 → 30 | 27 → 27 | 27 → 27 | card maker (template bars) |
| Several bars for one stat |  |  | 4 → 4 | 4 → 4 | 4 → 4 | 5 → 5 | Photoshop |
| The bar sits outside the card |  |  | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 | card maker |
| psd-tools draws this box differently from Photoshop | 42 → 0 | 22 → 0 | 4 → 0 | 4 → 0 | 4 → 0 | 4 → 0 | soft parity (being built) |

What stands out:

- Today the two biggest blockers are the colour icons (213 abilities) and the layer layout: titles the bot doesn't
  recognize (211 names) and Ability layers outside the Text group (92). Colour glyphs are built; refused-layers is in
  progress. With all three changes, 58 names are still blocked by "no Text group": nearly all
  are status effects whose title is a top-level layer named after the card, which refused-layers does not look at.
- What is left after the pending changes is mostly card-maker work: stat bars the bot can't use (38 cards),
  pictures instead of icon-font tokens (15 abilities), numbered PNGs (5 cards), plus the 18 badge layers waiting for
  the approved BUG 2 fix (19 gradient badge cards with Miracle Matter; 20 with Monkey D Goofy, the owner's count).
- **A bug in today's bot:** where the title is split over several layers or printed twice, it writes the whole edited
  title into the layer named Name and leaves the others. A dry run on Warcraftian Druid printed "Warcraftian Druidess"
  with the old "Druid" line still under it, and the read-back gate passed. Of the 20 cards with such a title, 9 bake
  today like this, wrongly: Doctor Zomboss, Moose Goose, Mosling, Tainted Eden, WX-78, Warcraftian Druid, Warcraftian
  Hunter, Warcraftian Paladin, Warcraftian Rogue. The other 11 refuse today (2.7): 5 titles run off the card, 4 reach
  the stars or type icons, and Autopsy Report and Warcraftian Shaman fail the read-back. refused-layers refuses all 20
  earlier, as titles printed by several layers, split or printed twice (2.10, 2.11, 2.12), which is right.
- **Four refused edits in Card Edits.json would bake today** (Metal General, Z.O.M.G., Star Dream, Zero Override;
  section 4). Their verdicts come from the old pixel bar and fork, and the schedule treats them as final until the PSD
  changes; a dispatched run without `--pending` re-checks them.

## 2. Why cards can't bake, by reason

### Whole card (every field)

#### 2.1 Photoshop's copy of the text is out of date

Today: **4 cards** (Name 4, Ability 4, HP 4, DEF 4, ATK 4, SPD 4). After the pending changes, **0** of these 24 fields bake.

- **Why:** Photoshop keeps a second copy of every text layer's words (the Txt2 story); on this card it holds older text than the layer, so Photoshop would show the old words after a bake.
- **What would fix it:** Code tool: `python -m commands.txt2_repair` on the card (one commit).

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| PoD08MOAB | Auxiliary/Minions | all | layer Name: Photoshop shows 'M.O.A.B', the layer holds 'M.O.A.B.' | no change |
| PoD09DDT | Auxiliary/Minions | all | layer Name: Photoshop shows 'D.D.T', the layer holds 'D.D.T.' | no change |
| PoD10BFB | Auxiliary/Minions | all | layer Name: Photoshop shows 'B.F.B', the layer holds 'B.F.B.' | no change |
| PoD12BAD | Auxiliary/Minions | all | layer Name: Photoshop shows 'B.A.D', the layer holds 'B.A.D.' | no change |

#### 2.2 Several PNGs come from one PSD

Today: **5 cards** (Name 5, Ability 5). After the pending changes, **0** of these 10 fields bake.

- **Why:** The folder has numbered PNGs (for example A_Snack2.png) exported from this one PSD; a bake re-exports only the PNG named like the PSD, so the bot leaves the card alone rather than let the copies drift apart.
- **What would fix it:** Card maker: give each numbered PNG its own PSD (save a copy as A_Snack2.psd), or delete the numbered PNGs if they are not separate cards. No code change planned.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| A Snack | Items/Heal/2 Stars | Name, Ability | A_Snack2.png, A_Snack3.png | no change |
| All Purpose Medical Kit | Items/Heal/2 Stars | Name, Ability | All_Purpose_Medical_Kit2.png | no change |
| NME Stock | Items/Support/2 Stars | Name, Ability | NME_Stock2.png | no change |
| Regeneration Potion | Items/Heal/2 Stars | Name, Ability | Regeneration_Potion2.png | no change |
| Watermelon | Items/Heal/2 Stars | Name, Ability | Watermelon2.png | no change |

#### 2.3 The stats come from a creatures list

Today: **1 card** (HP 1, DEF 1, ATK 1, SPD 1). After the pending changes, **0** of these 4 fields bake.

- **Why:** The card's entry in Card Edits.json has a creatures list, which sets the stats creature by creature; a baked stat would be overwritten by it.
- **What would fix it:** Edit the creatures list on the website instead; nothing to bake.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| Twin Emperors | Creatures/World Of Warcraft/4 Stars | all four stats | behind it, several bars per stat too (HP 3 bar rows, DEF 2 bar rows, ATK 2 bar rows, SPD 2 bar rows) | no change |

### Name

One message for four parts: when today's title layer is missing, hidden or named otherwise (no Text group; no title
layer the bot recognizes; the title layer has another name; the Name layer is hidden), the bake reports
`unsupported-card: no readable Text/Name layer` (`classify`, through `read_name`). The parts split that message by
what `name_layer` found, because that decides the fix. The parts on titles printed by several layers, split or printed
twice are refusals of work/refused-layers, not of today's bot.

#### 2.4 The title layer has another name

Today: **90 cards** (Name 90). After the pending changes, **79** of them bake.

- **Why:** The printed title is a text layer named after the card (for example 'Hornet') instead of 'Name', so today's bot doesn't know it is the title.
- **What would fix it:** Code, in progress (work/refused-layers): it bakes the layer when its name matches the card's. The owner chose to leave these layer names as they are (2026-10-22). Left after it: the layers whose name differs from the card's ('Twin Princes' on Lorian, 'Knight' on Knight Lost A Lot): edit in Photoshop, or rename that one layer to Name.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Acro | Creatures/Kirby/2 Stars | 'Acro' in layer 'Acro' (in Text) | bakes (refused-layers) |
| Alex YIIK | Creatures/Other/5 Stars | 'Alex YIIK' in layer 'this card fucking sucks' (in Text) | no change |
| Balloon Boy | Creatures/Fnaf/3 Stars | 'Balloon Boy' in layer 'Balloon Boy' (in Text) | bakes (refused-layers) |
| Barry The Barrel | Creatures/Other/2 Stars | 'Barry The Barrel' in layer 'Barry The Barrel' (in Text) | bakes (refused-layers) |
| Basic Shield | Items/Equip/2 Stars | 'Basic Shield' in layer 'Basic Shield' (in Text) | bakes (refused-layers) |
| Beldam | Creatures/Mario/3 Stars | 'Beldam' in layer 'Beldam' (in Text) | bakes (refused-layers) |
| Blooper Tentacle | Auxiliary/Minions | 'Blooper Tentacle' in layer 'Blooper Tentacle' (in Text) | bakes (refused-layers) |
| Bob-ulk | Creatures/Mario/2 Stars | 'Bob-Ulk' in layer 'Bob-Ulk' (in Text) | bakes with a pixel note (refused-layers) |
| Bugzzy | Auxiliary/Minions | 'Bugzzy' in layer 'Bugzzy' (in Text) | bakes (refused-layers) |
| Cheap | Auxiliary/Status Effects/Debuffs/3 Stars | 'Cheap' in layer 'Cheap' (in Text) | bakes (refused-layers) |
| Chris & Jordi | Creatures/Everhood/5 Stars | 'Jordi' in layer 'Knight' (in Text); 'Chris' in layer 'Knight copy' (in Text); '&' in layer 'Knight copy 2' (in Text) | no change |
| Crystal Guardian | Creatures/Hollow Knight/2 Stars | 'Crystal Guardian' in layer 'Crystal Guardian' (in Text) | bakes (refused-layers) |
| Crystal King | Creatures/Mario/4 Stars | 'Crystal King' in layer 'Crystal King' (in Text) | bakes (refused-layers) |
| Curse | Auxiliary/Status Effects/Debuffs/4 Stars | 'Curse' in layer 'Curse' (in Text) | bakes with a pixel note (refused-layers) |
| Dark matter | Creatures/Kirby/3 Stars | 'Dark MAtter' in layer 'Dark Matter' (in Text) | bakes (refused-layers) |
| Dark Matter Swordsman | Creatures/Kirby/3 Stars | 'Swordsman' in layer 'Swordsman' (in Text); 'Dark Matter' in layer 'Dark Matter' (in Text) | no change |
| Dark Mind | Creatures/Kirby/4 Stars | 'Dark Mind' in layer 'Dark Mind' (in Text) | bakes (refused-layers) |
| Dark Mind Eye | Creatures/Kirby/5 Stars | 'Dark Mind Eye' in layer 'Dark Mind Eye' (in Text) | bakes (refused-layers) |
| Daroach | Creatures/Kirby/3 Stars | 'Daroach' in layer 'Daroach' (in Text) | bakes (refused-layers) |
| Despair | Auxiliary/Status Effects/Debuffs/4 Stars | 'Despair' in layer 'Despair' (in Text) | bakes with a pixel note (refused-layers) |
| Disarmed | Auxiliary/Status Effects/Debuffs/3 Stars | 'Disarmed' in layer 'Disarmed' (in Text) | bakes (refused-layers) |
| Doc | Creatures/Kirby/3 Stars | 'Doc' in layer 'Doc' (in Text) | bakes (refused-layers) |
| Door Shield | Items/Equip/2 Stars | 'Door Shield' in layer 'Door Shield' (in Text) | bakes (refused-layers) |
| Drawcia | Creatures/Kirby/3 Stars | 'Drawcia' in layer 'Drawcia' (in Text) | bakes (refused-layers) |
| Dung Defender | Creatures/Hollow Knight/3 Stars | 'Dung Defender' in layer 'Dung Defender' (in Text) | bakes (refused-layers) |
| Dyna Blade | Creatures/Kirby/4 Stars | 'Dyna Blade' in layer 'Dyna Blade' (in Text) | bakes (refused-layers) |
| Edible | Auxiliary/Status Effects/Debuffs/2 Stars | 'Edible' in layer 'Edible' (in Text) | bakes (refused-layers) |
| False Knight | Creatures/Hollow Knight/2 Stars | 'False Knight' in layer 'False Knight' (in Text) | bakes (refused-layers) |
| Fatty Whale | Creatures/Kirby/3 Stars | 'Fatty Whale' in layer 'Fatty Whale' (in Text) | bakes (refused-layers) |
| Final Dark Mind Eye | Creatures/Kirby/3 Stars | 'Final Dark Mind Eye' in layer 'Final Dark Mind Eye' (in Text) | bakes (refused-layers) |
| Flurrie | Creatures/Mario/3 Stars | 'Flurrie' in layer 'Flurrie' (in Text) | bakes (refused-layers) |
| God Tamer | Auxiliary/Minions | 'God Tamer' in layer 'God Tamer' (in Text) | bakes (refused-layers) |
| Gold Fuzzy | Creatures/Mario/2 Stars | 'Gold Fuzzy' in layer 'Gold Fuzzy' (in Text) | bakes (refused-layers) |
| Goombella | Creatures/Mario/2 Stars | 'Goombella' in layer 'Goombella' (in Text) | bakes (refused-layers) |
| Gooper Blooper | Creatures/Mario/2 Stars | 'Gooper Blooper' in layer 'Gooper Blooper' (in Text) | bakes (refused-layers) |
| Halberd Main Cannon | Creatures/Kirby/3 Stars | 'Halberd Main Cannon' in layer 'Halberd Main Cannon' (in Text) | bakes (refused-layers) |
| Heavy Lobster | Creatures/Kirby/3 Stars | 'Heavy Lobster' in layer 'Heavy Lobster' (in Text) | bakes (refused-layers) |
| Heavy Mole | Creatures/Kirby/3 Stars | 'Heavy Mole' in layer 'Heavy Mole' (in Text) | bakes (refused-layers) |
| Hive Knight | Creatures/Hollow Knight/3 Stars | 'Hive Knight' in layer 'Hive Knight' (in Text) | bakes (refused-layers) |
| Hornet | Creatures/Hollow Knight/4 Stars | 'Hornet' in layer 'Hornet' (in Text) | bakes (refused-layers) |
| Ice Dragon | Auxiliary/Minions | 'Ice Dragon' in layer 'Ice Dragon' (in Text) | bakes (refused-layers) |
| Kabula | Creatures/Kirby/3 Stars | 'Kabula' in layer 'Kabula' (in Text) | bakes (refused-layers) |
| Knight Lost A Lot | Creatures/Everhood/2 Stars | 'Knight' in layer 'Knight' (in Text); 'Lost' in layer 'Lost' (in Text); 'A' in layer 'A' (in Text) | no change |
| Kracko | Creatures/Kirby/3 Stars | 'Kracko' in layer 'Kracko' (in Text) | bakes (refused-layers) |
| Lord Crump | Creatures/Mario/3 Stars | 'Lord Crump' in layer 'Lord Crump' (in Text) | bakes (refused-layers) |
| Lorian | Auxiliary/Minions | 'Lorian' in layer 'Twin Princes' (in Text) | no change |
| Lothric | Creatures/Souls/5 Stars | 'Lothric' in layer 'Twin Princes' (in Text) | no change |
| Magic Shield | Items/Equip/4 Stars | 'Magic Shield' in layer 'Magic Shield' (in Text) | bakes (refused-layers) |
| Magnus Von Grapple | Creatures/Mario/2 Stars | 'Magnus Von Grapple' in layer 'Magnus Von Grapple' (in Text) | bakes (refused-layers) |
| Magolor | Creatures/Kirby/3 Stars | 'Magolor' in layer 'Magolor' (in Text) | bakes (refused-layers) |
| Mangle | Creatures/Fnaf/3 Stars | 'Mangle' in layer 'Mangle' (in Text) | bakes (refused-layers) |
| Mantis Lords | Creatures/Hollow Knight/3 Stars | 'Mantis Lords' in layer 'Mantis Lords' (in Text) | bakes (refused-layers) |
| Marilyn | Creatures/Mario/3 Stars | 'Marilyn' in layer 'Marilyn' (in Text) | bakes (refused-layers) |
| Masher | Auxiliary/Minions | 'Masher' in layer 'Masher' (in Text) | bakes (refused-layers) |
| Master Hand | Creatures/Kirby/3 Stars | 'Master Hand' in layer 'Master Hand' (in Text) | bakes (refused-layers) |
| Metal Shield | Items/Equip/3 Stars | 'Metal Shield' in layer 'Metal Shield' (in Text) | bakes (refused-layers) |
| Monster Meat | Items/Heal/1 Stars | 'Monster Meat' in layer 'Monster Meat' (in Text) | bakes (refused-layers) |
| Ms Mowz | Creatures/Mario/2 Stars | 'Ms. Mowz' in layer 'Ms. Mowz' (in Text) | no change |
| Nailsage Sly | Creatures/Hollow Knight/4 Stars | 'Nailsage Sly' in layer 'Nailsage Sly' (in Text) | bakes (refused-layers) |
| Necrodeus | Creatures/Kirby/5 Stars | 'Necrodeus' in layer 'Necrodeus' (in Text) | bakes (refused-layers) |
| Nosk | Creatures/Hollow Knight/4 Stars | 'Nosk' in layer 'Nosk' (in Text) | bakes (refused-layers) |
| Old Bonnie | Creatures/Fnaf/4 Stars | 'Old Bonnie' in layer 'Old Bonnie' (in Text) | bakes (refused-layers) |
| Old Chica | Creatures/Fnaf/4 Stars | 'Old Chica' in layer 'Old Chica' (in Text) | bakes (refused-layers) |
| Old Foxy | Creatures/Fnaf/4 Stars | 'Old Foxy' in layer 'Old Foxy' (in Text) | bakes (refused-layers) |
| Old Freddy | Creatures/Fnaf/4 Stars | 'Old Freddy' in layer 'Old Freddy' (in Text) | bakes (refused-layers) |
| Papyrus | Creatures/Undertale/3 Stars | 'The Great Papyrus' in layer 'The Great Papyrus' (in Text) | no change |
| Pessimism | Auxiliary/Status Effects/Debuffs/3 Stars | 'Pessimism' in layer 'Pessimism' (in Text) | bakes (refused-layers) |
| Po | Creatures/Pandas/4 Stars | 'Po' in layer 'Po' (in Text) | bakes (refused-layers) |
| Pure Vessel | Creatures/Hollow Knight/5 Stars | 'Pure Vessel' in layer 'Pure Vessel' (in Text) | bakes (refused-layers) |
| Sans | Creatures/Undertale/5 Stars | 'snas.' in layer 'snas.' (in Text) | no change |
| Shell Shock | Auxiliary/Status Effects/Debuffs/3 Stars | 'Shell Shock' in layer 'Shell Shock' (in Text) | bakes with a pixel note (refused-layers) |
| Sniffle Thwomp | Creatures/Mario/3 Stars | 'Sniffle Thwomp' in layer 'Sniffle  Thwomp' (in Text) | still refused: the title would touch the stars or type icons (shown once refused-layers lands) |
| Snow Bunny | Items/Heal/2 Stars | 'Snow Bunny' in layer 'Snow Bunny' (in Text) | bakes (refused-layers) |
| Storo | Creatures/Kirby/3 Stars | 'Storo' in layer 'Storo' (in Text) | bakes (refused-layers) |
| The Beast | Creatures/Hollow Knight/3 Stars | 'The Beast' in layer 'The Beast' (in Text) | bakes (refused-layers) |
| The Master | Creatures/Mario/5 Stars | 'The Master' in layer 'The Master' (in Text) | bakes (refused-layers) |
| Thunder Thighs | Items/Heal/2 Stars | 'Thunder thighs' in layer 'Supper' (in Text) | no change |
| Toriel | Creatures/Undertale/4 Stars | 'Toriel' in layer 'Toriel' (in Text) | bakes (refused-layers) |
| Toy Bonnie | Creatures/Fnaf/3 Stars | 'Toy Bonnie' in layer 'Toy Bonnie' (in Text) | bakes (refused-layers) |
| Toy Chica | Creatures/Fnaf/3 Stars | 'Toy Chica' in layer 'Toy Chica' (in Text) | bakes (refused-layers) |
| Toy Freddy | Creatures/Fnaf/3 Stars | 'Toy Freddy' in layer 'Toy Freddy' (in Text) | bakes (refused-layers) |
| Tutankoopa | Creatures/Mario/3 Stars | 'Tutankoopa' in layer 'Tutankoopa' (in Text) | bakes (refused-layers) |
| Twin Emperors | Creatures/World Of Warcraft/4 Stars | 'Twin Emperors' in layer 'Twin Emperors' (in Text) | bakes (refused-layers) |
| Undyne | Creatures/Undertale/4 Stars | 'Undyne' in layer 'Undyne' (in Text) | bakes (refused-layers) |
| Uumuu | Creatures/Hollow Knight/3 Stars | 'Uumuu' in layer 'Uumuu' (in Text) | bakes (refused-layers) |
| Wham Bam Bone | Creatures/Kirby/4 Stars | 'Wham Bam Bone' in layer 'Wham Bam Bone' (in Text) | bakes (refused-layers) |
| Wham Bam Jewel | Creatures/Kirby/4 Stars | 'Wham Bam Jewel' in layer 'Wham Bam Jewel' (in Text) | bakes (refused-layers) |
| Wham Bam Rock | Creatures/Kirby/3 Stars | 'Wham Bam Rock' in layer 'Wham Bam Rock' (in Text) | bakes (refused-layers) |
| Whispy Woods | Creatures/Kirby/2 Stars | 'Whispy Woods' in layer 'Whispy Woods' (in Text) | bakes (refused-layers) |
| Yoshi Kid | Creatures/Mario/2 Stars | 'Yoshi Kid' in layer 'Yoshi Kid' (in Text) | bakes (refused-layers) |

#### 2.5 No Text group

Today: **69 cards** (Name 69). After the pending changes, **11** of them bake.

- **Why:** The card has no layer group named Text, so today's bot can't find the printed title.
- **What would fix it:** Code, in progress (work/refused-layers), but it covers only the 11 cards with a layer named Name somewhere. The other 58 (nearly all status effects) print the title in a top-level layer named after the card; a small extension of the refused-layers title finder would take those too (not planned yet). Card maker alternative: put the title and the Ability layer in a group named Text.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Afraid | Auxiliary/Status Effects/Debuffs/2 Stars | 'Afraid' in layer 'Afraid', top level | no change |
| Blazing | Auxiliary/Status Effects/Debuffs/2 Stars | 'Blazing' in layer 'Blazing', top level | no change |
| Blindness | Auxiliary/Status Effects/Debuffs/1 Stars | 'Blindness' in layer 'Blindness', top level | no change |
| Burdened | Auxiliary/Status Effects/Debuffs/1 Stars | 'Burdened' in layer 'Burdened', top level | no change |
| Burning | Auxiliary/Status Effects/Debuffs/1 Stars | 'Burning' in layer 'Burning', top level | no change |
| Chained | Auxiliary/Status Effects/Debuffs/2 Stars | 'Chained' in layer 'Chained', top level | no change |
| Chilled | Auxiliary/Status Effects/Debuffs/2 Stars | 'Chilled' in layer 'Chilled', top level | no change |
| Condemned | Auxiliary/Status Effects/Debuffs/4 Stars | 'Condemned' in layer 'Condemned', top level | no change |
| Confusion | Auxiliary/Status Effects/Debuffs/2 Stars | 'Confusion' in layer 'Confusion', top level | no change |
| Contracted | Auxiliary/Status Effects/Debuffs/4 Stars | 'Contracted' in layer 'Contracted', top level | no change |
| Corrupted Blood | Auxiliary/Status Effects/Debuffs/3 Stars | 'Corrupted Blood' in layer 'Corrupted Blood', top level | no change |
| Corruption | Auxiliary/Status Effects/Debuffs/3 Stars | 'Corruption' in layer 'Corruption', top level | no change |
| Cursed Inferno | Auxiliary/Status Effects/Debuffs/3 Stars | 'Cursed Inferno' in layer 'Cursed Inferno', top level | no change |
| Diligent Watch | Auxiliary/Status Effects/Buffs | 'Diligent Watch' in layer 'Diligent Watch', top level | no change |
| Disgusted | Auxiliary/Status Effects/Debuffs/1 Stars | 'Disgusted' in layer 'Disgusted', top level | no change |
| Dreamshield | Auxiliary/Items | 'Dreamshield' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Empowered | Auxiliary/Status Effects/Buffs | 'Empowered' in layer 'Empowered', top level | no change |
| Empty | Auxiliary/Status Effects/Debuffs/2 Stars | 'Empty' in layer 'Empty', top level | no change |
| Enfeebled | Auxiliary/Status Effects/Debuffs/1 Stars | 'Enfeebled' in layer 'Enfeebled', top level | no change |
| Engorged | Auxiliary/Status Effects/Debuffs/2 Stars | 'Engorged' in layer 'Engorged', top level | no change |
| Enraged | Auxiliary/Status Effects/Buffs | 'Enraged' in layer 'Enraged', top level | no change |
| Exposed | Auxiliary/Status Effects/Debuffs/2 Stars | 'Exposed' in layer 'Exposed', top level | no change |
| Fragile Strength | Items/Equip/3 Stars | 'Fragile strength' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Frostbite | Auxiliary/Status Effects/Debuffs/2 Stars | 'Frostbite' in layer 'Frostbite', top level | no change |
| Frozen | Auxiliary/Status Effects/Debuffs/3 Stars | 'Frozen' in layer 'Frozen', top level | no change |
| Frozen In Time | Auxiliary/Status Effects/Debuffs/4 Stars | 'Frozen In Time' in layer 'Frozen In Time', top level | no change |
| Glued | Auxiliary/Status Effects/Debuffs/2 Stars | 'Glued' in layer 'Name', top level | bakes (refused-layers) |
| Hardened | Auxiliary/Status Effects/Buffs | 'Hardened' in layer 'Hardened', top level | no change |
| Heavy Blow | Items/Equip/2 Stars | 'Heavy Blow' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Hemorrhaging | Auxiliary/Status Effects/Debuffs/4 Stars | 'Hemorrhaging' in layer 'Hemorrhaging', top level | no change |
| Hexed | Auxiliary/Status Effects/Debuffs/3 Stars | 'Hexed' in layer 'Hexed', top level | no change |
| Hyper | Auxiliary/Status Effects/Buffs | 'Hyper' in layer 'Hyper', top level | no change |
| Imprisoned | Auxiliary/Status Effects/Debuffs/3 Stars | 'Imprisoned' in layer 'Imprisoned', top level | no change |
| Infected | Auxiliary/Status Effects/Debuffs/4 Stars | 'Infected' in layer 'Infected', top level | no change |
| Insane | Auxiliary/Status Effects/Debuffs/3 Stars | 'Insane' in layer 'Insane', top level | no change |
| Inspired | Auxiliary/Status Effects/Buffs | 'Inspired' in layer 'Inspired', top level | no change |
| Kel | Creatures/Omori/3 Stars | 'Kel' in layer 'Kel' (in Group 1) | no change |
| Kingsoul | Items/Equip/4 Stars | 'Kingsoul' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Loony | Auxiliary/Status Effects/Debuffs/2 Stars | 'Loony' in layer 'Loony', top level | no change |
| Mari | Creatures/Omori/4 Stars | 'Mari' in layer 'Name' (in text) | bakes (refused-layers) |
| Marked For Death | Auxiliary/Status Effects/Debuffs/4 Stars | 'Marked For Death' in layer 'Marked For Death', top level | no change |
| Marked For Life | Auxiliary/Status Effects/Buffs | 'Marked For Life' in layer 'Marked For Life', top level | no change |
| Mr Jawsum | Creatures/Omori/3 Stars | 'Mr. Jawsum' in layer 'Mr. Jawsum' (in Group 3/Group 2) | no change |
| Newspaper | Items/Equip/3 Stars | 'Newspaper' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Paralyzed | Auxiliary/Status Effects/Debuffs/2 Stars | 'Paralyzed' in layer 'Paralyzed', top level | no change |
| Peppy | Auxiliary/Status Effects/Buffs | 'Peppy' in layer 'Peppy', top level | no change |
| Plague | Auxiliary/Status Effects/Debuffs/2 Stars | 'Plague' in layer 'Plague', top level | no change |
| Poison | Auxiliary/Status Effects/Debuffs/1 Stars | 'Poison' in layer 'Poison', top level | no change |
| Quick Slash | Items/Equip/3 Stars | 'Quick Slash' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Recursion | Auxiliary/Status Effects/Debuffs/4 Stars | 'Recursion' in layer 'Recursion', top level | no change |
| Regen | Auxiliary/Status Effects/Buffs | 'Regen' in layer 'Regen', top level | no change |
| Reinforced | Auxiliary/Status Effects/Buffs | 'Reinforced' in layer 'Reinforced', top level | no change |
| Resentful Glare | Auxiliary/Status Effects/Debuffs/3 Stars | 'Resentful Glare' in layer 'Resentful Glare', top level | no change |
| Restrained | Auxiliary/Status Effects/Debuffs/2 Stars | 'Restrained' in layer 'Restrained', top level | no change |
| Sickness | Auxiliary/Status Effects/Debuffs/1 Stars | 'Sickness' in layer 'Sickness', top level | no change |
| Smeckeldorfed | Auxiliary/Status Effects/Debuffs/1 Stars | 'Smeckeldorfed' in layer 'Smeckeldorfed', top level | no change |
| Smoldering | Auxiliary/Status Effects/Debuffs/1 Stars | 'Smoldering' in layer 'Smoldering', top level | no change |
| Softened | Auxiliary/Status Effects/Debuffs/1 Stars | 'Softened' in layer 'Softened', top level | no change |
| Spirit Flame | Auxiliary/Status Effects/Debuffs/2 Stars | 'Spirit Flame' in layer 'Spirit Flame', top level | no change |
| Stalwart Shell | Items/Equip/3 Stars | 'Stalwart shell' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Stuffed | Auxiliary/Status Effects/Debuffs/1 Stars | 'Stuffed' in layer 'Stuffed', top level | no change |
| Suffocation | Auxiliary/Status Effects/Debuffs/3 Stars | 'Suffocation' in layer 'Suffocation', top level | no change |
| Tainted Lazarus | Creatures/Isaac/3 Stars | 'Tainted Lazarus' in layer 'Name' (in Laz/Laz) | bakes (refused-layers) |
| Tanjerin | Creatures/Bug Fables/5 Stars | 'Tanjerin' in layer 'Tanjerin' (in text/Group 2) | no change |
| Thickened | Auxiliary/Status Effects/Buffs | 'Thickened' in layer 'Thickened', top level | no change |
| Thinned | Auxiliary/Status Effects/Debuffs/1 Stars | 'Thinned' in layer 'Thinned', top level | no change |
| Unbreakable Strength | Auxiliary/Items | 'Unbreakable strength' in layer 'Name' (in Types/Text) | bakes (refused-layers) |
| Vigor | Auxiliary/Status Effects/Buffs | 'Vigor' in layer 'Vigor', top level | no change |
| Withering | Auxiliary/Status Effects/Debuffs/4 Stars | 'Withering' in layer 'Withering', top level | no change |

#### 2.6 No title layer the bot recognizes

Today: **52 cards** (Name 52). After the pending changes, **38** of them bake.

- **Why:** The Text group has no layer named Name and no other text layer near the top, so today's bot can't tell which layer is the printed title (it may sit outside the group, be a picture, or carry another name).
- **What would fix it:** Code, in progress (work/refused-layers): bake the one layer the parser reads as the title. Card maker alternative: make the title one visible text layer named Name inside the Text group.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| ABBI | Creatures/Omori/5 Stars | no visible text layer at the top | no change |
| Baba | Creatures/Other/5 Stars | no visible text layer at the top | no change |
| Bench | Field/1 Stars | 'Bench' in layer 'Name', top level | bakes (refused-layers) |
| Bulborb Larva | Auxiliary/Minions | 'Bulborb Larva' in layer 'Name', top level | bakes (refused-layers) |
| Captain Spaceboy | Creatures/Omori/3 Stars | 'Captain Spaceboy' in layer 'Name' (in Stat Txt) | bakes (refused-layers) |
| Computer Virus | Creatures/Kirby/3 Stars | 'Computer Virus appears!' in layer 'Name', top level | bakes (refused-layers) |
| Dark Nebula | Creatures/Kirby/4 Stars | 'Dark Nebula' in layer 'Name', top level | bakes (refused-layers) |
| Dark Ritual | Items/Equip/4 Stars | 'Dark ritual' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Dark Sword | Items/Equip/3 Stars | 'Dark sword' in layer 'Name', top level | bakes (refused-layers) |
| Electric Dart | N.M.E | 'Electric Dart' in layer 'Name', top level | bakes (refused-layers) |
| Empress Bulblax | Creatures/Other/3 Stars | 'Empress Bulblax' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Fawful | Creatures/Mario/4 Stars | 'FAWFUL' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Future Niiue | Creatures/Other/2 Stars | 'Future Niiue' in layer 'Name', top level | bakes (refused-layers) |
| Fuzzy Horde | Auxiliary/Minions | 'Fuzzy Horde' in layer 'Name', top level | bakes (refused-layers) |
| Golden Bomb | Items/Attack/4 Stars | 'Golden bomb' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Golden Metronome | Items/Negation/4 Stars | 'Golden Metronome' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Green Mage | Creatures/Everhood/4 Stars | 'Green Mage' in layer 'Name', top level | bakes (refused-layers) |
| Grey Prince Zote | Creatures/Hollow Knight/4 Stars | 'Invincible Fearless Sensual Mysterious E' in layer 'Invincible Fearless Sensual Mysterious Enchanting Vigorous Dili', top level | no change |
| Grodus X | Auxiliary/Minions | 'Grodus X' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Holy Card | Items/Equip/2 Stars | no visible text layer at the top | no change |
| Hooktail | Creatures/Mario/2 Stars | 'Hooktail' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Indoctrination | Auxiliary/Items | 'Indoctrinati0n' in layer 'Name' (in Types) | bakes (refused-layers) |
| Leshy | Creatures/Other/5 Stars | 'Leshy' in layer 'Name', top level | bakes (refused-layers) |
| Lor Starcutter | Creatures/Kirby/4 Stars | 'Lor Starcutter' in layer 'Name', top level | bakes (refused-layers) |
| Macho Grubba | Creatures/Mario/4 Stars | 'Macho Grubba' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Mimic | Items/Support/4 Stars | 'Mimic' in layer 'Name', top level | bakes (refused-layers) |
| Minecraft Panda | Creatures/Pandas/2 Stars | no visible text layer at the top | no change |
| Morpho Knight | Auxiliary/Creatures | 'Morpho Knight' in layer 'Name', top level | bakes (refused-layers) |
| NME Salesman | Creatures/Kirby/4 Stars | 'NME SALESMAN' in layer 'Name', top level | bakes (refused-layers) |
| omori | Creatures/Omori/5 Stars | 'OMORI' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Pandanoko | Creatures/Pandas/2 Stars | no visible text layer at the top | no change |
| Pontiff Sulyvahn | Creatures/Souls/4 Stars | 'Pontiff Sulyvahn' in layer 'Name', top level | bakes (refused-layers) |
| Rasta Beast | Creatures/Everhood/4 Stars | 'Pasta Beast' in layer 'Name', top level | bakes (refused-layers) |
| Raw Cock | Creatures/Mario/3 Stars | 'Raw Kawk' in layer 'Name', top level | bakes (refused-layers) |
| Sacred Heart | Items/Equip/4 Stars | 'Sacred Heart' in layer 'Name', top level | bakes (refused-layers) |
| Sir Grodus | Creatures/Mario/4 Stars | 'Grodus' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Skullord | Creatures/Kirby/3 Stars | 'Skullord' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| Smorg | Creatures/Mario/4 Stars | 'Smorg' in layer 'Name' (in Text/Stat Txt) | bakes (refused-layers) |
| SOMETHING | Creatures/Omori/3 Stars | 'SOMETHING' in layer 'Name', top level | bakes (refused-layers) |
| Soul Master | Creatures/Hollow Knight/4 Stars | 'Soul Master' in layer 'Soul Master', top level | no change |
| Stranger | Creatures/Omori/4 Stars | 'Stranger' in layer 'Name', top level | bakes (refused-layers) |
| The Big Yellow Cat | Creatures/Omori/5 Stars | 'The' in layer 'Name' (in Name); 'Cat' in layer 'Name' (in Name); 'Yellow' in layer 'Name' (in Name) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| The Last Cabbage Merchant | Creatures/Other/3 Stars | no visible text layer at the top | no change |
| The Radiance | Creatures/Hollow Knight/5 Stars | 'The Radiance' in layer 'The Radiance' (in Group 1) | no change |
| The Spire | Field/3 Stars | 'The Spire' in layer 'Name', top level | bakes with a pixel note (refused-layers) |
| Tomorrow I'll | Items/Equip/2 Stars | 'Tomorrow I’ll' in layer 'Name', top level | bakes (refused-layers) |
| Traitor Lord | Creatures/Hollow Knight/4 Stars | 'Traitor Lord' in layer 'Traitor Lord', top level | no change |
| Tyler | Creatures/Other/5 Stars | no visible text layer at the top | no change |
| Waxwell | Creatures/DST/3 Stars | 'Waxwell' in layer 'Name', top level | bakes (refused-layers) |
| Wickerbottom | Creatures/DST/3 Stars | 'Wickerbottom' in layer 'Name', top level | still refused: the title's words are printed twice (shown once refused-layers lands) |
| Wolfgang | Creatures/DST/4 Stars | no visible text layer at the top | no change |
| Zigg | Creatures/Everhood/3 Stars | 'Zigg' in layer 'Name', top level | bakes (refused-layers) |

#### 2.7 The whole title doesn't fit into one of its layers

Today: **11 cards** (Name 11). After the pending changes, **0** of them bake.

- **Why:** The title is split over several text layers or printed twice (sections on split and doubled titles below). Today's bot writes the whole edited title, as the website shows it, into the one layer named Name; on these cards that refuses: the title runs off the card, reaches the stars or type icons, or the read-back still doesn't match the edit ('idempotence: still to bake: name').
- **What would fix it:** Card maker: merge the title into one text layer named Name (line breaks inside one layer are fine). work/refused-layers refuses these cards earlier, as split or doubled titles; no code change would bake them.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Autopsy Report | Auxiliary/Items | the read-back still differs from the edit (still to bake: name); title in: 'Type: Attack Evidence submitted by Miles' in layer 'Type: Attack Evidence submitted by Miles Edgeworth.', top level; 'Autopsy Report' in layer 'Name' (in Text) | still refused: the title's words are printed twice (shown once refused-layers lands) |
| Azamoth the Forgotten King | Creatures/ROTMG/5 Stars | runs off the card; title in: 'Forgotten King' in layer 'Name' (in Text); 'Azamoth The' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| Azamoth the Possessed Vessel | Auxiliary/Creatures | runs off the card; title in: 'Possessed Vessel' in layer 'Name' (in Text); 'Azamoth The' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| Crowned Magolor | Creatures/Kirby/5 Stars | reaches the stars or type icons; title in: 'Crowned' in layer 'Name' (in Text); 'Magolor' in layer 'Magolor' (in Text) | still refused: the title is split over several layers (shown once refused-layers lands) |
| Massive Moss Charger | Auxiliary/Minions | runs off the card; title in: 'Moss Charger' in layer 'Name' (in Text); 'Massive' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| Oryx the Exalted God | Creatures/ROTMG/4 Stars | runs off the card; title in: 'Exalted God' in layer 'Name' (in Text); 'Oryx The' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| The Bagel | N.M.E | runs off the card; title in: 'The Bagel' in layer 'Name' (in Text); 'Or... is it' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| The Gates of Ahn'Qiraj | Field/4 Stars | reaches the stars or type icons; title in: 'The gates of' in layer 'Name' (in Text); 'AHN’QiRAj' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| The Temple of Ahn'Qiraj | Auxiliary/Items | reaches the stars or type icons; title in: 'The Temple of' in layer 'Name' (in Text); 'AHN’QiRAj' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |
| Warcraftian Priest | Creatures/World Of Warcraft/4 Stars | reaches the stars or type icons; title in: 'Warcraftian' in layer 'Name' (in Text); 'Priest' in layer 'Priest' (in Text) | still refused: the title is split over several layers (shown once refused-layers lands) |
| Warcraftian Shaman | Creatures/World Of Warcraft/4 Stars | the read-back still differs from the edit (still to bake: name); title in: 'Warcraftian' in layer 'Name' (in Text); 'Shaman' in layer 'Name' (in Text) | still refused: more than one layer prints the title (shown once refused-layers lands) |

#### 2.8 The Name layer is hidden

Today: **4 cards** (Name 4). After the pending changes, **0** of them bake.

- **Why:** The layer called Name is hidden, so it isn't what the card prints; the printed title is something else.
- **What would fix it:** Not fixed by refused-layers. Card maker: delete the hidden Name layer (on Wigfrid it is a leftover reading 'Wickerbottom', on E. Gadd's Garage 'Glitz Pit'). On these four cards no visible text layer prints the title (it looks like part of the picture), so to make the title editable, retype it as a text layer named Name.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| E. Gadd's Garage | Field/2 Stars | hidden Name reads 'Glitz Pit'; no visible title text (a picture?) | no change |
| Man Serpent Shield | Items/Equip/2 Stars | hidden Name reads 'Man Serpent Shield'; no visible title text (a picture?) | no change |
| Sonic | Creatures/Other/4 Stars | hidden Name reads 'Sonic'; no visible title text (a picture?) | no change |
| Wigfrid | Creatures/DST/3 Stars | hidden Name reads 'Wickerbottom'; no visible title text (a picture?) | no change |

#### 2.9 The title would touch the stars or type icons

Today: **1 card** (Name 1). After the pending changes, **0** of them bake.

- **Why:** Re-typesetting the title runs it into the star or type icons beside it.
- **What would fix it:** Edit in Photoshop (shrink or move the title). No code change planned.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Jack-In-The-Box Zombie | Auxiliary/Minions | 'jack-in-the-box zombie' in layer 'Name' (in Text) | no change |

Cards that meet this reason only once the pending changes land (1; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| Sniffle Thwomp | Creatures/Mario/3 Stars | Name | 'Sniffle Thwomp' in layer 'Sniffle  Thwomp' (in Text) | today: the title layer has another name |

#### 2.10 More than one layer prints the title

Today: none (this reason shows only once refused-layers lands).

- **Why:** Several visible layers read as the title (an epithet layer such as Grey Prince Zote's, a copy kept for an effect, or two layers both called Name), so the bot can't tell which one to change.
- **What would fix it:** Edit the title in Photoshop, or merge the title into one text layer. No code change planned.

Cards that meet this reason only once the pending changes land (9; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| Azamoth the Forgotten King | Creatures/ROTMG/5 Stars | Name | 'Forgotten King' in layer 'Name' (in Text); 'Azamoth The' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| Azamoth the Possessed Vessel | Auxiliary/Creatures | Name | 'Possessed Vessel' in layer 'Name' (in Text); 'Azamoth The' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| Massive Moss Charger | Auxiliary/Minions | Name | 'Moss Charger' in layer 'Name' (in Text); 'Massive' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| Oryx the Exalted God | Creatures/ROTMG/4 Stars | Name | 'Exalted God' in layer 'Name' (in Text); 'Oryx The' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| The Bagel | N.M.E | Name | 'The Bagel' in layer 'Name' (in Text); 'Or... is it' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| The Big Yellow Cat | Creatures/Omori/5 Stars | Name | 'The' in layer 'Name' (in Name); 'Cat' in layer 'Name' (in Name); 'Yellow' in layer 'Name' (in Name) | today: no title layer the bot recognizes |
| The Gates of Ahn'Qiraj | Field/4 Stars | Name | 'The gates of' in layer 'Name' (in Text); 'AHN’QiRAj' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| The Temple of Ahn'Qiraj | Auxiliary/Items | Name | 'The Temple of' in layer 'Name' (in Text); 'AHN’QiRAj' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |
| Warcraftian Shaman | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Shaman' in layer 'Name' (in Text) | today: the whole title doesn't fit into one of its layers |

#### 2.11 The title is split over several layers

Today: none (this reason shows only once refused-layers lands).

- **Why:** The title is spread over several text layers (a word or a line each), so a new title can't be written into one of them.
- **What would fix it:** Card maker: merge the title into one text layer (line breaks inside one layer are fine).

Cards that meet this reason only once the pending changes land (8; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| Crowned Magolor | Creatures/Kirby/5 Stars | Name | 'Crowned' in layer 'Name' (in Text); 'Magolor' in layer 'Magolor' (in Text) | today: the whole title doesn't fit into one of its layers |
| Royal Eradicator Main Cannon | Creatures/FTL/5 Stars | Name | 'Main Cannon' in layer 'Name' (in Text); 'Eradicator' in layer 'Eradicator' (in Text); 'Royal' in layer 'Royal' (in Text) | today: psd-tools draws this box differently from photoshop |
| Warcraftian Druid | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Druid' in layer 'Druid' (in Text) | today: bakes (into one of the title's layers only) |
| Warcraftian Hunter | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Hunter' in layer 'Hunter' (in Text) | today: bakes (into one of the title's layers only) |
| Warcraftian Paladin | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Paladin' in layer 'Paladin' (in Text) | today: bakes (into one of the title's layers only) |
| Warcraftian Priest | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Priest' in layer 'Priest' (in Text) | today: the whole title doesn't fit into one of its layers |
| Warcraftian Rogue | Creatures/World Of Warcraft/4 Stars | Name | 'Warcraftian' in layer 'Name' (in Text); 'Rogue' in layer 'Rogue' (in Text) | today: bakes (into one of the title's layers only) |
| WX-78 | Creatures/DST/3 Stars | Name | 'wx-' in layer 'Name' (in Text); '78' is a picture (smart-object layer '78' in Text) | today: bakes (into one of the title's layers only) |

#### 2.12 The title's words are printed twice

Today: none (this reason shows only once refused-layers lands).

- **Why:** Another visible text layer prints the same words as the title (a second copy elsewhere on the card), so a bake would change one and leave the other.
- **What would fix it:** Edit in Photoshop; or, if the second copy may differ from the title, relax the check in work/refused-layers.

Cards that meet this reason only once the pending changes land (6; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| Autopsy Report | Auxiliary/Items | Name | another visible layer 'Name' prints the same words | today: the whole title doesn't fit into one of its layers |
| Doctor Zomboss | Creatures/Other/5 Stars | Name | another visible layer 'Name' prints the same words | today: bakes (into one of the title's layers only) |
| Moose Goose | Creatures/DST/3 Stars | Name | another visible layer 'Name' prints the same words | today: bakes (into one of the title's layers only) |
| Mosling | Auxiliary/Minions | Name | another visible layer 'Name' prints the same words | today: bakes (into one of the title's layers only) |
| Tainted Eden | Creatures/Isaac/2 Stars | Name | another visible layer 'Name copy' prints the same words | today: bakes (into one of the title's layers only) |
| Wickerbottom | Creatures/DST/3 Stars | Name | another visible layer 'Name' prints the same words | today: no title layer the bot recognizes |

### Ability

#### 2.13 Colour icons would turn one colour

Today: **213 cards** (Ability 213). After the pending changes, **213** of them bake.

- **Why:** The ability uses the stoogedings icon font, whose icons are colour pictures; the live psd-tools draws them in a single colour, so a bake would change how every icon looks.
- **What would fix it:** Built, waiting for release: round 6 colour glyphs (fork r6/integration e6539a2f + AutoCroissant work/r6-color-glyphs).

After the pending changes: **bakes (colour glyphs)** (209): Air Totem (Auxiliary/Items), Alex YIIK (Creatures/Other/5 Stars), Assault Panda (Creatures/Pandas/2 Stars), Auto Croissant (Creatures/Other/2 Stars), Azamoth the Forgotten King (Creatures/ROTMG/5 Stars), Azamoth the Possessed Vessel (Auxiliary/Creatures), Barry The Barrel (Creatures/Other/2 Stars), Beldam (Creatures/Mario/3 Stars), Blasphemous Blade (Items/Equip/4 Stars), Blockade (Items/Active/4 Stars), Blood Bomb (Items/Attack/3 Stars), Blue Marker (Items/Support/3 Stars), Bob-ulk (Creatures/Mario/2 Stars), Bobberlington (Creatures/Other/5 Stars), Bolts and Screws (Items/Heal/1 Stars), Cabbage (Auxiliary/Items), Cake Mix (Items/Heal/2 Stars), Cave Of The Past (Field/4 Stars), Charge (Items/Equip/2 Stars), Chef Saltbaker (Creatures/Cuphead/5 Stars), Chris & Jordi (Creatures/Everhood/5 Stars), Clock (Items/Active/2 Stars), Cloud Soiree (Field/1 Stars), Crowned Magolor (Creatures/Kirby/5 Stars), Curse (Auxiliary/Status Effects/Debuffs/4 Stars), Dark Bramble (Field/4 Stars), Dark Daroach (Creatures/Kirby/4 Stars), Dark Mind (Creatures/Kirby/4 Stars), Dark Nebula (Creatures/Kirby/4 Stars), Daroach (Creatures/Kirby/3 Stars), Deep Focus (Items/Equip/3 Stars), Defcon Panda (Creatures/Pandas/3 Stars), Doc (Creatures/Kirby/3 Stars), Dogma (Creatures/Isaac/4 Stars), DooPliss (Creatures/Mario/4 Stars), Dr Robotnik's Roboticizer (Items/Support/1 Stars), Drawcia (Creatures/Kirby/3 Stars), Dumplings (Items/Heal/3 Stars), Dung Defender (Creatures/Hollow Knight/3 Stars), Dyna Blade (Creatures/Kirby/4 Stars), Earth Totem (Auxiliary/Items), Energy Ritual (Items/Support/3 Stars), Fahr Outpost (Field/2 Stars), Fatty Whale (Creatures/Kirby/3 Stars), Fawful (Creatures/Mario/4 Stars), Feast (Items/Heal/3 Stars), Fire Totem (Auxiliary/Items), Flurrie (Creatures/Mario/3 Stars), Forsaken City (Field/2 Stars), Fragile Heart (Items/Equip/3 Stars), Funky D Luffy (Creatures/Monkey D Folder/4 Stars), Fuzzy Horde (Auxiliary/Minions), Galacta Knight (Creatures/Kirby/5 Stars), Gandhi (Creatures/Other/5 Stars), Gasoline (Items/Support/2 Stars), Glaive Main Cannon (Creatures/FTL/4 Stars), Glowing Womb (Items/Equip/2 Stars), Godskin Apostle (Auxiliary/Minions), Godskin Noble (Creatures/Souls/4 Stars), Goombella (Creatures/Mario/2 Stars), Green Mage (Creatures/Everhood/4 Stars), Greenpath (Field/2 Stars), Grimmchild (Auxiliary/Minions), Groda (Creatures/Pandas/4 Stars), GrubSong (Items/Equip/1 Stars), Halberd Main Cannon (Creatures/Kirby/3 Stars), Halberd Reactor (Creatures/Kirby/3 Stars), Hiveblood (Items/Equip/3 Stars), Hooktail (Creatures/Mario/2 Stars), Hooktail Castle (Field/2 Stars), Hornet (Creatures/Hollow Knight/4 Stars), hr-h (Creatures/Kirby/4 Stars), Igor Roballtowski (Auxiliary/Creatures), Igor Robotowski (Creatures/Other/2 Stars), Inkhole (Auxiliary/Items), Ipsilon (Auxiliary/Minions), Isma (Creatures/Hollow Knight/4 Stars), Jetpack (Auxiliary/Minions), Jevil (Creatures/Undertale/4 Stars), Kalecgos (Creatures/World Of Warcraft/4 Stars), King Dedede (Creatures/Kirby/4 Stars), Koops (Creatures/Mario/2 Stars), Kracko (Creatures/Kirby/3 Stars), Kyle (Creatures/Other/5 Stars), Landia (Creatures/Kirby/5 Stars), Little Guy (Creatures/Other/2 Stars), Lor Starcutter (Creatures/Kirby/4 Stars), Lord Crump (Creatures/Mario/3 Stars), Magic Hammer (Items/Equip/3 Stars), Magic Shield (Items/Equip/4 Stars), Magnum Opus (Auxiliary/Items), Magolor (Creatures/Kirby/3 Stars), Master Hand (Creatures/Kirby/3 Stars), Mega Titan (Creatures/Kirby/3 Stars), Meta Knight (Creatures/Kirby/4 Stars), Metal General (Creatures/Kirby/3 Stars), Metronome (Items/Negation/2 Stars), Mettaton EX (Auxiliary/Creatures), Michael (Creatures/Other/5 Stars), Miles Edgeworth (Creatures/Other/4 Stars), Militron (Creatures/Other/4 Stars), Minecraft Panda (Creatures/Pandas/2 Stars), Mini Auto Croissant (Auxiliary/Minions), Monkey D Vour (Creatures/Monkey D Folder/4 Stars), Morshu (Creatures/Other/4 Stars), Ms Mowz (Creatures/Mario/2 Stars), Nasus (Creatures/Other/4 Stars), Nightmare (Creatures/Kirby/5 Stars), Nightmare King Grimm (Creatures/Hollow Knight/5 Stars), Niko (Creatures/Other/3 Stars), NME Salesman (Creatures/Kirby/4 Stars), Nuclear Proliferation (Items/Equip/4 Stars), Ominous Red Circle (Items/Deploy/2 Stars), Overseer (Creatures/Other/3 Stars), P.A.N.D.A. Mk A (Creatures/Pandas/3 Stars), P.A.N.D.A. Mk T (Creatures/Pandas/3 Stars), Pan D Luffy (Creatures/Monkey D Folder/3 Stars), Pandabal (Creatures/Pandas/3 Stars), Pandanoko (Creatures/Pandas/2 Stars), Pandevourer (Creatures/Pandas/4 Stars), Pandora's Gym (Field/1 Stars), Pep Drink (Items/Heal/2 Stars), Peter Panda (Creatures/Pandas/2 Stars), Po (Creatures/Pandas/4 Stars), PoD03White Bloon (Auxiliary/Minions), PoD05Lead Bloon (Auxiliary/Minions), Poison Circle (Items/Deploy/1 Stars), Prince Of Darkness (Creatures/BTD/5 Stars), Project P.A.N.D.A (Creatures/Pandas/3 Stars), Pure Vessel (Creatures/Hollow Knight/5 Stars), Rasta Beast (Creatures/Everhood/4 Stars), Raw Cock (Creatures/Mario/3 Stars), Retro Kirby (Creatures/Kirby/3 Stars), Rock Bottom (Field/3 Stars), Rogueport (Field/1 Stars), Royal Eradicator Main Cannon (Creatures/FTL/5 Stars), Sandwich of Rejects (Items/Heal/2 Stars), Sans (Creatures/Undertale/5 Stars), Sharp Baldur (Auxiliary/Minions), Shield Master Pi (Creatures/Pandas/3 Stars), Shield of Ogmur (Items/Equip/4 Stars), Sigma (Auxiliary/Minions), Sir Grodus (Creatures/Mario/4 Stars), Smorg (Creatures/Mario/4 Stars), Sniffle Thwomp (Creatures/Mario/3 Stars), Soul Master (Creatures/Hollow Knight/4 Stars), Soul Sanctum (Field/2 Stars), Spawn of the Forbidden One (Auxiliary/Minions), Spice (Items/Support/2 Stars), Steady Body (Items/Equip/1 Stars), Storo (Creatures/Kirby/3 Stars), Stubborn King (Creatures/Other/2 Stars), Sumo Panda (Creatures/Pandas/3 Stars), Supreme Jeh'Oul (Creatures/Other/3 Stars), Sword of Aeons (Items/Equip/4 Stars), Tainted Eden (Creatures/Isaac/2 Stars), Tainted Eve (Creatures/Isaac/4 Stars), Tainted Isaac (Creatures/Isaac/3 Stars), Tainted Jacob (Creatures/Isaac/3 Stars), Tainted Lost (Creatures/Isaac/2 Stars), Tainted Magdalene (Creatures/Isaac/3 Stars), The Abyss Watchers (Creatures/Souls/3 Stars), The Claw (N.M.E), The Collector (Creatures/Hollow Knight/3 Stars), The Farlands (Field/2 Stars), The Forbidden One (Creatures/Other/5 Stars), The Freezer (Field/1 Stars), The Gates of Ahn'Qiraj (Field/4 Stars), The Hive (Field/4 Stars), The Hydrant (Items/Support/2 Stars), The Kestrel (Creatures/FTL/4 Stars), The Kingchin (Creatures/Other/4 Stars), The Krusty Krab (Field/3 Stars), The Lich King (Creatures/World Of Warcraft/5 Stars), The Panda Future (Field/4 Stars), The Plaguelands (Field/3 Stars), The Queen (Creatures/Other/5 Stars), The Ringed City (Field/1 Stars), The Spire (Field/3 Stars), The Structure (Field/4 Stars), The Temple of Ahn'Qiraj (Auxiliary/Items), Time Riddle (Items/Equip/4 Stars), Toriel (Creatures/Undertale/4 Stars), Traitor Lord (Creatures/Hollow Knight/4 Stars), Tutankoopa (Creatures/Mario/3 Stars), Tyler (Creatures/Other/5 Stars), Utica Stock Exchange (Field/2 Stars), Vivian (Creatures/Mario/3 Stars), Waddle Dee (Creatures/Kirby/3 Stars), Warcraftian Mage (Creatures/World Of Warcraft/4 Stars), Warcraftian Paladin (Creatures/World Of Warcraft/4 Stars), Warcraftian Priest (Creatures/World Of Warcraft/4 Stars), Warcraftian Shaman (Creatures/World Of Warcraft/4 Stars), Warcraftian Warlock (Creatures/World Of Warcraft/4 Stars), Warcraftian Warrior (Creatures/World Of Warcraft/4 Stars), Water Totem (Auxiliary/Items), Wham Bam Jewel (Creatures/Kirby/4 Stars), Wham Bam Rock (Creatures/Kirby/3 Stars), Wokey D pronouns (Creatures/Monkey D Folder/3 Stars), YIIK Panda (Creatures/Pandas/2 Stars), Yoshi Kid (Creatures/Mario/2 Stars), You are going DOWN you Sputnik! (Items/Counter/3 Stars), Z!!!!! (Creatures/Other/5 Stars), Ze'mer (Creatures/Hollow Knight/5 Stars), Zenith Martlet (Creatures/Undertale/5 Stars), Zero (Creatures/Kirby/4 Stars), Zero Two (Creatures/Kirby/5 Stars), Zigg (Creatures/Everhood/3 Stars), Zoltan (Creatures/FTL/2 Stars).

After the pending changes: **bakes with a pixel note (colour glyphs)** (4): Cortez (Creatures/Mario/4 Stars), Masked Dedede (Creatures/Kirby/5 Stars), Panseida (Creatures/Pandas/3 Stars), The Radiance (Creatures/Hollow Knight/5 Stars).

#### 2.14 The Ability layer is outside the Text group

Today: **92 cards** (Ability 92). After the pending changes, **85** of them bake.

- **Why:** The Ability text layer sits outside the Text group, and today's bot only edits an Ability layer directly inside Text.
- **What would fix it:** Code, in progress (work/refused-layers): accept the single visible Ability layer wherever it is. Card maker alternative: drag the Ability layer into the Text group.

After the pending changes: **bakes (refused-layers)** (81): Blazing (Auxiliary/Status Effects/Debuffs/2 Stars), Burdened (Auxiliary/Status Effects/Debuffs/1 Stars), Burning (Auxiliary/Status Effects/Debuffs/1 Stars), Chained (Auxiliary/Status Effects/Debuffs/2 Stars), Condemned (Auxiliary/Status Effects/Debuffs/4 Stars), Confusion (Auxiliary/Status Effects/Debuffs/2 Stars), Contracted (Auxiliary/Status Effects/Debuffs/4 Stars), Corrupted Blood (Auxiliary/Status Effects/Debuffs/3 Stars), Crystal Guardian (Creatures/Hollow Knight/2 Stars), Cursed Inferno (Auxiliary/Status Effects/Debuffs/3 Stars), Diligent Watch (Auxiliary/Status Effects/Buffs), Disgusted (Auxiliary/Status Effects/Debuffs/1 Stars), Dreamshield (Auxiliary/Items), Empty (Auxiliary/Status Effects/Debuffs/2 Stars), Enfeebled (Auxiliary/Status Effects/Debuffs/1 Stars), Engorged (Auxiliary/Status Effects/Debuffs/2 Stars), Exposed (Auxiliary/Status Effects/Debuffs/2 Stars), Forgotten Crossroads (Field/1 Stars), Fragile Strength (Items/Equip/3 Stars), Frozen (Auxiliary/Status Effects/Debuffs/3 Stars), Frozen In Time (Auxiliary/Status Effects/Debuffs/4 Stars), Glued (Auxiliary/Status Effects/Debuffs/2 Stars), God Tamer (Auxiliary/Minions), Hardened (Auxiliary/Status Effects/Buffs), Heavy Blow (Items/Equip/2 Stars), Hemorrhaging (Auxiliary/Status Effects/Debuffs/4 Stars), Hexed (Auxiliary/Status Effects/Debuffs/3 Stars), Humphrey's Belly (Auxiliary/Items), Hyper (Auxiliary/Status Effects/Buffs), Imprisoned (Auxiliary/Status Effects/Debuffs/3 Stars), Indoctrination (Auxiliary/Items), Infected (Auxiliary/Status Effects/Debuffs/4 Stars), Insane (Auxiliary/Status Effects/Debuffs/3 Stars), Inspired (Auxiliary/Status Effects/Buffs), Kel (Creatures/Omori/3 Stars), Kingsoul (Items/Equip/4 Stars), Leshy (Creatures/Other/5 Stars), Loony (Auxiliary/Status Effects/Debuffs/2 Stars), Lothric (Creatures/Souls/5 Stars), Mantis Lords (Creatures/Hollow Knight/3 Stars), Marked For Death (Auxiliary/Status Effects/Debuffs/4 Stars), Marked For Life (Auxiliary/Status Effects/Buffs), Marx (Creatures/Kirby/5 Stars), Mr Jawsum (Creatures/Omori/3 Stars), Necrodeus (Creatures/Kirby/5 Stars), Newspaper (Items/Equip/3 Stars), Paralyzed (Auxiliary/Status Effects/Debuffs/2 Stars), Peppy (Auxiliary/Status Effects/Buffs), Pix (Creatures/Kirby/3 Stars), Poison (Auxiliary/Status Effects/Debuffs/1 Stars), Quick Slash (Items/Equip/3 Stars), Recursion (Auxiliary/Status Effects/Debuffs/4 Stars), Regen (Auxiliary/Status Effects/Buffs), Reinforced (Auxiliary/Status Effects/Buffs), Resentful Glare (Auxiliary/Status Effects/Debuffs/3 Stars), Restrained (Auxiliary/Status Effects/Debuffs/2 Stars), Shadow Chopper (Auxiliary/Minions), Shadow Miner (Auxiliary/Minions), Skullord (Creatures/Kirby/3 Stars), Smeckeldorfed (Auxiliary/Status Effects/Debuffs/1 Stars), Smoldering (Auxiliary/Status Effects/Debuffs/1 Stars), Snow Angel (Creatures/Omori/3 Stars), Softened (Auxiliary/Status Effects/Debuffs/1 Stars), Spamton Neo (Creatures/Undertale/5 Stars), Spirit Flame (Auxiliary/Status Effects/Debuffs/2 Stars), Stalwart Shell (Items/Equip/3 Stars), Stotch (Creatures/Other/2 Stars), Stranger (Creatures/Omori/4 Stars), Stuffed (Auxiliary/Status Effects/Debuffs/1 Stars), Suffocation (Auxiliary/Status Effects/Debuffs/3 Stars), Tanjerin (Creatures/Bug Fables/5 Stars), The Bank (N.M.E), The Beast (Creatures/Hollow Knight/3 Stars), The Big Yellow Cat (Creatures/Omori/5 Stars), Thickened (Auxiliary/Status Effects/Buffs), Thinned (Auxiliary/Status Effects/Debuffs/1 Stars), Tomorrow I'll (Items/Equip/2 Stars), Unbreakable Strength (Auxiliary/Items), Vigor (Auxiliary/Status Effects/Buffs), Withering (Auxiliary/Status Effects/Debuffs/4 Stars), Yin Yarn (Creatures/Kirby/5 Stars).

After the pending changes: **still refused: the ability's icons are pictures (shown once refused-layers lands)** (7): Afraid (Auxiliary/Status Effects/Debuffs/2 Stars), Blindness (Auxiliary/Status Effects/Debuffs/1 Stars), Chilled (Auxiliary/Status Effects/Debuffs/2 Stars), Corruption (Auxiliary/Status Effects/Debuffs/3 Stars), Frostbite (Auxiliary/Status Effects/Debuffs/2 Stars), Plague (Auxiliary/Status Effects/Debuffs/2 Stars), Sickness (Auxiliary/Status Effects/Debuffs/1 Stars).

After the pending changes: **bakes with a pixel note (refused-layers)** (3): Empowered (Auxiliary/Status Effects/Buffs), Enraged (Auxiliary/Status Effects/Buffs), Mari (Creatures/Omori/4 Stars).

After the pending changes: **bakes (pixels not measured) (refused-layers)** (1): Bugzzy (Auxiliary/Minions).

#### 2.15 A badge layer would be lost (BUG 2)

Today: **18 cards** (Ability 18). After the pending changes, **0** of them bake.

- **Why:** An 'Ability FX: ...' layer paints the timing badges or icons over the ability with a gradient; a bake would delete it and the bot can't rebuild it yet.
- **What would fix it:** Code fix approved, not started: rebuild the Ability FX layers from the new text's word boxes (NEXT_IDEAS section 5, idea 4; needs text geometry and colour glyphs). Until then, edit in Photoshop. There are 19 gradient badge cards: these 18 and Miracle Matter ('Ability FX: (1GCD)'; listed under several visible Ability layers, its first refusal). With Monkey D Goofy ('Ability FX: 1#1', listed under word layers) they are the owner's 20 BUG 2 cards.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Final Dark Mind Eye | Creatures/Kirby/3 Stars | Ability FX: (OD) | no change |
| Heavy Mole | Creatures/Kirby/3 Stars | Ability FX: (1CD) | no change |
| Martlet | Creatures/Undertale/3 Stars | Ability FX: (2CD) | no change |
| Mini Doomer | Auxiliary/Minions | Ability FX: (BOT) | no change |
| Morpho Knight | Auxiliary/Creatures | Ability FX: (1CD) / (3U) / (OYT) / (ENDT) | no change |
| Pandamonium | Items/Active/4 Stars | Ability FX: (1GCD) | no change |
| Party! | Auxiliary/Items | Ability FX: (BOGT) | no change |
| PoD02Black Bloon | Auxiliary/Minions | Ability FX: (1GCD) | no change |
| PoD06Rainbow Bloon | Auxiliary/Minions | Ability FX: (OD) | no change |
| PoD11ZOMG | Auxiliary/Minions | Ability FX: (1GCD) / (2U) / (2CD) | no change |
| Star Cannon | Items/Active/3 Stars | Ability FX: (1GCD) / (4U) / (OYT) | no change |
| Stay | Creatures/Other/3 Stars | Ability FX: (WOF) | no change |
| Taranza | Creatures/Kirby/3 Stars | Ability FX: (2CD) | no change |
| Turtle | Auxiliary/Minions | Ability FX: (1GCD) | no change |
| Voidwalker | Auxiliary/Minions | Ability FX: (1GCD) | no change |
| Warcraftian Hunter | Creatures/World Of Warcraft/4 Stars | Ability FX: (OFS) / (Action) | no change |
| Wham Bam Bone | Creatures/Kirby/4 Stars | Ability FX: (OFS) / (BOGT) | no change |
| Whispy Woods | Creatures/Kirby/2 Stars | Ability FX: (1CD) | no change |

#### 2.16 The ability's icons are pictures

Today: **8 cards** (Ability 8). After the pending changes, **0** of them bake.

- **Why:** The type icons in the ability are separate picture layers, not the stoogedings icon font, so the bot can't place [type] icons in retyped text (the website shows them as [fire]-style tokens).
- **What would fix it:** Card maker / standardization: move the card's icons to stoogedings tokens (the B1/B2 batches). No code change planned.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Electric Dart | N.M.E |  | no change |
| Elemental World | Field/3 Stars |  | no change |
| Error Room | Field/4 Stars |  | no change |
| Irithyll | Field/2 Stars |  | no change |
| Nebulous Precipice | Field/3 Stars |  | no change |
| Palace of Shadow | Field/4 Stars |  | no change |
| The Core | Field/2 Stars |  | no change |
| Zess T Deluxe | Items/Heal/4 Stars |  | no change |

Cards that meet this reason only once the pending changes land (7; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| Afraid | Auxiliary/Status Effects/Debuffs/2 Stars | Ability |  | today: the ability layer is outside the text group |
| Blindness | Auxiliary/Status Effects/Debuffs/1 Stars | Ability |  | today: the ability layer is outside the text group |
| Chilled | Auxiliary/Status Effects/Debuffs/2 Stars | Ability |  | today: the ability layer is outside the text group |
| Corruption | Auxiliary/Status Effects/Debuffs/3 Stars | Ability |  | today: the ability layer is outside the text group |
| Frostbite | Auxiliary/Status Effects/Debuffs/2 Stars | Ability |  | today: the ability layer is outside the text group |
| Plague | Auxiliary/Status Effects/Debuffs/2 Stars | Ability |  | today: the ability layer is outside the text group |
| Sickness | Auxiliary/Status Effects/Debuffs/1 Stars | Ability |  | today: the ability layer is outside the text group |

#### 2.17 A word layer the status chart has no style for

Today: **8 cards** (Ability 8). After the pending changes, **0** of them bake.

- **Why:** A hand-made word layer over the ability (an overlay or FX layer for a word such as Smoldering or +2 DEF) has no style on the status chart, so the bot can't recreate it after an edit.
- **What would fix it:** Owner decision 2026-10-21: stays refused; edit in Photoshop or remove the overlay by hand. Except Monkey D Goofy: its 'Ability FX: 1#1' layer styles a '1' in the ability, and the owner counts the card with BUG 2, so the approved BUG 2 fix (rebuild the Ability FX layers) is meant to cover it.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Clot Gold | Auxiliary/Minions | 1 Become Back Money | no change |
| Computer Virus | Creatures/Kirby/3 Stars | 1st: Slime, 2nd: Puppet, 3rd: Magician | no change |
| Fire Staff | Items/Equip/2 Stars | Smoldering | no change |
| Flameblade | Items/Equip/3 Stars | Attacks inflict 2T | no change |
| Ice Staff | Items/Equip/2 Stars | Chilled | no change |
| Monkey D Goofy | Creatures/Monkey D Folder/4 Stars | Ability FX: 1#1 (the owner counts this card with BUG 2) | no change |
| Mr Krabs | Creatures/Other/4 Stars | 1 Become Back Money, 2T | no change |
| The Watcher | Creatures/In Sound Mind/3 Stars | Despair | no change |

#### 2.18 No visible Ability layer

Today: **7 cards** (Ability 7). After the pending changes, **0** of them bake.

- **Why:** No visible text layer is called Ability (it is hidden, named something else, or the ability is a picture), so there is nothing for the bot to retype.
- **What would fix it:** Card maker: make the printed ability one visible text layer named Ability. No code change planned.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Abyss Watcher | Auxiliary/Minions | hidden: Text/Ability | no change |
| Apex Predator | Creatures/R.E.P.O/2 Stars | hidden: Text/Ability Txt/Ability | no change |
| Bulborb Larva | Auxiliary/Minions | hidden: Text/Ability | no change |
| Butterfly | Creatures/Kirby/5 Stars | hidden: Text/Ability Txt/Ability | no change |
| Empress Bulblax | Creatures/Other/3 Stars | hidden: Text/Ability | no change |
| Shadow Duelist | Auxiliary/Minions | hidden: Ability | no change |
| Warrior Dee | Auxiliary/Minions | hidden: Text/Ability | no change |

#### 2.19 Several visible Ability layers

Today: **5 cards** (Ability 5). After the pending changes, **0** of them bake.

- **Why:** Two or more visible text layers are called Ability (one per creature or per paragraph), so the bot can't tell which one an edit belongs to.
- **What would fix it:** Card maker: merge them into one Ability layer, or edit this card in Photoshop. No code change planned.

| Card | Folder | Detail | After the pending changes |
|---|---|---|---|
| Lit Bob-omb | Auxiliary/Items | 2 layers | no change |
| Miracle Matter | Creatures/Kirby/3 Stars | 5 layers; also a badge layer 'Ability FX: (1GCD)' (BUG 2) | no change |
| Tainted Lazarus | Creatures/Isaac/3 Stars | 2 layers | no change |
| Twin Emperors | Creatures/World Of Warcraft/4 Stars | 5 layers | no change |
| Warcraftian Druid | Creatures/World Of Warcraft/4 Stars | 2 layers | no change |

#### 2.20 The ability's words are printed twice

Today: none (this reason shows only once refused-layers lands).

- **Why:** Another visible text layer prints the same words as the Ability layer, so a bake would change one and leave the other.
- **What would fix it:** Card maker: delete or hide the duplicate layer (or edit in Photoshop).

Cards that meet this reason only once the pending changes land (1; another reason stops them first today, or today's bot does not look this closely):

| Card | Folder | Field | Detail | Today |
|---|---|---|---|---|
| SOMETHING | Creatures/Omori/3 Stars | Ability | also in layer 'Attacks are piercing, and can attack twi...' | today: bakes |

### Stats (HP, DEF, ATK, SPD)

#### 2.21 The stat bar is not the template's

Today: **31 cards** (HP 30, DEF 30, ATK 27, SPD 27). After the pending changes, **0** of these 114 fields bake.

- **Why:** The bar's group exists but doesn't hold the template's ten masked digit layers named 0-9 (one digit layer only, digits 1-10, an extra 'gauge' or 'Dark' layer, a missing or doubled digit). Renaming the group would not help: none of these rows has template digits under another name.
- **What would fix it:** Card maker: replace the row with the template's row. No code change planned.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| Anubisath Guardian | Auxiliary/Minions | HP | 'Hp Dark': extra 'Layer 1' | no change |
| Blank Card | Items/Active/3 Stars | HP, DEF | 'hp': digits numbered 1-10 (the other bars alike) | no change |
| Bulborb Larva | Auxiliary/Minions | all four stats | 'Hp Dark': extra 'gauge', '10' (the other bars alike) | no change |
| Chef Saltbaker | Creatures/Cuphead/5 Stars | all four stats | 'hp' holds only '8' (the other bars alike) | no change |
| Chris & Jordi | Creatures/Everhood/5 Stars | all four stats | 'hp': no digit layers (holds 'hp', 'Layer 8 copy 7', ...) (the other bars alike) | no change |
| Clot Black | Auxiliary/Minions | all four stats | 'hp': digit layers all named '0' (the other bars alike) | no change |
| Clot Blood | Auxiliary/Minions | all four stats | 'hp': digit layers all named '0' (the other bars alike) | no change |
| Clot Bone | Auxiliary/Minions | all four stats | 'HP': digit layers all named '0' (the other bars alike) | no change |
| Clot Gold | Auxiliary/Minions | all four stats | 'hp': digit layers all named '0' (the other bars alike) | no change |
| Clot White | Auxiliary/Minions | all four stats | 'hp': digit layers all named '0' (the other bars alike) | no change |
| Dark Esau | Auxiliary/Minions | all four stats | 'HP' holds only 'Dark', '5' (the other bars alike) | no change |
| Dark Mind | Creatures/Kirby/4 Stars | DEF | 'Def Dark': extra '5 copy' | no change |
| Dirty Bed | Items/Active/2 Stars | HP, DEF | 'hp': digits numbered 1-10 (the other bars alike) | no change |
| Empress Bulblax | Creatures/Other/3 Stars | all four stats | 'Hp Dark': extra 'gauge', '10' (the other bars alike) | no change |
| Grimora | Creatures/Other/3 Stars | all four stats | 'hp' holds only '5' (the other bars alike) | no change |
| Minecraft Panda | Creatures/Pandas/2 Stars | all four stats | 'Darks': digit layers all named '0' | no change |
| Pizza | Creatures/Other/3 Stars | all four stats | 'HP' holds only '6' (the other bars alike) | no change |
| Skeleton Mate | Auxiliary/Minions | all four stats | 'HP' holds only '2' (the other bars alike) | no change |
| Slugcat | Creatures/Other/2 Stars | all four stats | 'Hp Dark' holds only '4', 'Exposure' (the other bars alike) | no change |
| Spinni | Creatures/Kirby/3 Stars | all four stats | 'Hp Dark': no digit 5 (the other bars alike) | no change |
| Tainted Bethany | Creatures/Isaac/4 Stars | all four stats | 'HP' holds only 'Dark', '7' (the other bars alike) | no change |
| Tainted Eden | Creatures/Isaac/2 Stars | all four stats | 'HP' holds only 'Dark', '6' (the other bars alike) | no change |
| Tainted Eve | Creatures/Isaac/4 Stars | all four stats | 'HP' holds only 'Dark', '10' (the other bars alike) | no change |
| Tainted Isaac | Creatures/Isaac/3 Stars | all four stats | 'HP' holds only 'Dark', '5' (the other bars alike) | no change |
| Tainted Jacob | Creatures/Isaac/3 Stars | all four stats | 'HP' holds only 'Dark', '5' (the other bars alike) | no change |
| Tainted Keeper | Creatures/Isaac/4 Stars | all four stats | 'HP' holds only '0', '5' (the other bars alike) | no change |
| Tainted Lost | Creatures/Isaac/2 Stars | all four stats | 'HP' holds only 'Dark', '1'; 'DEF' holds only 'Dark', '0', 'Gradiant' ... | no change |
| Tainted Magdalene | Creatures/Isaac/3 Stars | all four stats | 'HP' holds only 'Layer 1 copy 11', '7' (the other bars alike) | no change |
| Uumuu | Creatures/Hollow Knight/3 Stars | all four stats | 'Hp Dark' holds only '6' (the other bars alike) | no change |
| Whispy Woods | Creatures/Kirby/2 Stars | all four stats | 'HP DARK': digits numbered 1-10 (the other bars alike) | no change |
| Zigg | Creatures/Everhood/3 Stars | all four stats | 'hp': only digit 5, the rest layers like 'Layer 8 copy 9' (the other bars alike) | no change |

#### 2.22 Several bars for one stat

Today: **5 cards** (HP 4, DEF 4, ATK 4, SPD 5). After the pending changes, **0** of these 17 fields bake.

- **Why:** The card has more than one bar for this stat (two creatures, a shared-health bar, stacked rows), so the bot can't tell which to change. (Two rows named Top and Bottom inside one 'Hp Dark' group, for stats above 10, are supported.)
- **What would fix it:** Edit in Photoshop. No code change planned.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| False Knight | Creatures/Hollow Knight/2 Stars | all four stats | bars: 'Empty', 'Filled' | no change |
| Pix | Creatures/Kirby/3 Stars | all four stats | bars: '3', '2', '1' | no change |
| Sonic | Creatures/Other/4 Stars | SPD | bars: 'Spd Dark', '2', '1' | no change |
| Tainted Lazarus | Creatures/Isaac/3 Stars | all four stats | bars: 'Hp Dark', 'Light', 'Hp Dark', 'Light' (the other bars alike) | no change |
| Ze'mer | Creatures/Hollow Knight/5 Stars | all four stats | bars: 'hp', 'Hp Dark' (the other bars alike) | no change |

#### 2.23 No template stat bar

Today: **1 card** (HP 1, DEF 1, ATK 1, SPD 1). After the pending changes, **0** of these 4 fields bake.

- **Why:** The card prints this stat, but not as the template's stat bar (a group such as 'Hp Dark' holding ten masked digit layers 0-9), so there is no digit for the bot to switch.
- **What would fix it:** Card maker: rebuild the bars from the card template. Code idea (NEXT_IDEAS section 5b D, after template fill): rebuild them automatically; not approved yet.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| Computer Virus | Creatures/Kirby/3 Stars | all four stats |  | no change |

#### 2.24 The bar sits outside the card

Today: **1 card** (HP 1, DEF 1, ATK 1, SPD 1). After the pending changes, **0** of these 4 fields bake.

- **Why:** The bar's digit layers lie below the edge of the card, so there is nothing on the card for a bake to redraw (the bake would stop with an error).
- **What would fix it:** Card maker: move the stat bars onto the card, or delete them if the card shows its stats another way.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| Green Mage | Creatures/Everhood/4 Stars | all four stats | digit masks below the 2400 px canvas | no change |

### Pixel check

#### 2.25 psd-tools draws this box differently from Photoshop

Today: **68 cards** (Name 42, Ability 22, HP 4, DEF 4, ATK 4, SPD 4). After the pending changes, **79** of these 80 fields bake.

- **Why:** Before baking, the bot redraws the box with psd-tools and compares it with the card's PNG; on this card the difference is above today's bar (0.5% bad pixels, mean 2.0).
- **What would fix it:** Approved 2026-10-23 and being built (work/soft-parity): a pixel miss no longer blocks; the bake goes ahead with a note in the report and commit. Only a grossly broken render (more than 25% of the box off) or a wrong read-back refuses.

| Card | Folder | Field | Detail | After the pending changes |
|---|---|---|---|---|
| Black Light Bulb | Items/Active/3 Stars | Name | bad 0.76% mean 0.52 max 72 | bakes (round 6 brings it under the bar) |
| Boggly Woods | Field/2 Stars | Name | bad 2.36% mean 2.95 max 255 | bakes (round 6 brings it under the bar) |
| Bolts and Screws | Items/Heal/1 Stars | Name | bad 2.05% mean 1.32 max 132 | bakes with a pixel note (soft parity) |
| Burly Bomb | Items/Attack/3 Stars | Ability | bad 0.67% mean 1.38 max 159 | bakes (round 6 brings it under the bar) |
| Copper Sword | Items/Equip/1 Stars | Ability | bad 2.84% mean 2.16 max 164 | bakes (round 6 brings it under the bar) |
| Corruptor | Auxiliary/Minions | all four stats | bad 3.49% mean 3.04 max 38 (the other bars alike) | bakes with a pixel note (soft parity) |
| Credit Card | Items/Support/3 Stars | Ability | bad 0.84% mean 0.62 max 255 | bakes with a pixel note (soft parity) |
| Danger Badges | Items/Equip/2 Stars | Ability | bad 1.72% mean 1.70 max 239 | bakes (round 6 brings it under the bar) |
| Dark Bramble | Field/4 Stars | Name | bad 5.92% mean 7.60 max 255 | bakes with a pixel note (soft parity) |
| Dark Sword | Items/Equip/3 Stars | Ability | bad 0.85% mean 0.57 max 125 | bakes (round 6 brings it under the bar) |
| Dynamite | Items/Attack/4 Stars | Ability | bad 1.88% mean 1.23 max 214 | bakes (round 6 brings it under the bar) |
| Elmadan | Creatures/Other/3 Stars | Name | bad 3.52% mean 2.29 max 109 | bakes with a pixel note (soft parity) |
| Energy Ritual | Items/Support/3 Stars | Name | bad 0.76% mean 1.09 max 255 | bakes (round 6 brings it under the bar) |
| Fireball | Items/Attack/3 Stars | Ability | bad 0.86% mean 0.55 max 219 | bakes (round 6 brings it under the bar) |
| Fog Canyon | Field/2 Stars | Name | bad 1.19% mean 1.08 max 255 | bakes (round 6 brings it under the bar) |
| Forget Me Now | Items/Support/3 Stars | Ability | bad 1.27% mean 0.72 max 160 | bakes with a pixel note (soft parity) |
| Frog | Creatures/Everhood/5 Stars | Name, HP, DEF, ATK, SPD | bad 0.00% mean 4.30 max 18; bad 0.00% mean 6.55 max 22; bad 0.00% mean 4.21 max 23; bad 0.00% mean 6.31 max 24; bad 0.00% mean 6.06 max 23 | bakes with a pixel note (soft parity) |
| Glaive Main Cannon | Creatures/FTL/4 Stars | Name | bad 1.88% mean 1.65 max 255 | bakes (round 6 brings it under the bar) |
| Godskin Apostle | Auxiliary/Minions | Name | bad 3.72% mean 2.25 max 127 | bakes (round 6 brings it under the bar) |
| Godskin Noble | Creatures/Souls/4 Stars | Name | bad 3.87% mean 2.49 max 123 | bakes (round 6 brings it under the bar) |
| Greenpath | Field/2 Stars | Name | bad 1.56% mean 0.98 max 176 | bakes with a pixel note (soft parity) |
| Grimmchild | Auxiliary/Minions | Name | bad 3.04% mean 1.67 max 172 | bakes (round 6 brings it under the bar) |
| Hadal Stew | Items/Heal/3 Stars | Ability | bad 1.79% mean 1.49 max 195 | bakes (round 6 brings it under the bar) |
| Hive Knight | Creatures/Hollow Knight/3 Stars | ATK | bad 0.00% mean 2.02 max 14 | bakes with a pixel note (soft parity) |
| Hooktail Castle | Field/2 Stars | Name | bad 3.51% mean 3.82 max 215 | bakes (round 6 brings it under the bar) |
| Ice Dragon | Auxiliary/Minions | Ability | bad 2.04% mean 1.93 max 137 | bakes (round 6 brings it under the bar) |
| Infected Crossroads | Field/3 Stars | Name | bad 5.05% mean 5.89 max 255 | bakes (round 6 brings it under the bar) |
| Inner Eye | Items/Equip/3 Stars | Ability | bad 0.55% mean 0.35 max 127 | bakes with a pixel note (soft parity) |
| Irithyll | Field/2 Stars | Name | bad 2.22% mean 1.37 max 152 | bakes (round 6 brings it under the bar) |
| Iron Spikes | Items/Counter/3 Stars | Ability | bad 1.01% mean 0.67 max 96 | bakes (round 6 brings it under the bar) |
| Iron Sword | Items/Equip/2 Stars | Ability | bad 1.53% mean 0.96 max 114 | bakes (round 6 brings it under the bar) |
| Jetpack | Auxiliary/Minions | Name | bad 0.00% mean 2.46 max 27 | bakes with a pixel note (soft parity) |
| Kalecgos | Creatures/World Of Warcraft/4 Stars | HP, DEF, SPD | bad 0.01% mean 4.15 max 25 (the other bars alike) | bakes with a pixel note (soft parity) |
| Katarina | Creatures/Other/4 Stars | Name | bad 0.89% mean 1.16 max 252 | bakes (round 6 brings it under the bar) |
| Keelhaul Key | Field/1 Stars | Ability | bad 2.65% mean 1.71 max 170 | bakes (round 6 brings it under the bar) |
| Kyle | Creatures/Other/5 Stars | Name | bad 2.59% mean 3.47 max 253 | bakes (round 6 brings it under the bar) |
| Little Guy | Creatures/Other/2 Stars | Name | bad 1.43% mean 0.70 max 40 | bakes with a pixel note (soft parity) |
| Machine Gun | Auxiliary/Items | Name | bad 0.90% mean 0.69 max 61 | bakes (round 6 brings it under the bar) |
| Masher | Auxiliary/Minions | Ability | bad 1.82% mean 0.98 max 78 | bakes (round 6 brings it under the bar) |
| Masked Dedede | Creatures/Kirby/5 Stars | Name | bad 0.00% mean 3.07 max 14 | bakes with a pixel note (soft parity) |
| Mechanical Shield | Items/Equip/3 Stars | Ability | bad 1.27% mean 0.89 max 79 | bakes (round 6 brings it under the bar) |
| Mr Softener | Items/Infliction/2 Stars | Ability | bad 0.51% mean 0.49 max 82 | bakes (round 6 brings it under the bar) |
| Pile Of Swords | Items/Support/4 Stars | Name | bad 2.56% mean 2.33 max 199 | bakes with a pixel note (soft parity) |
| Pizza | Creatures/Other/3 Stars | Ability | bad 2.23% mean 1.27 max 84 | bakes with a pixel note (soft parity) |
| PoD06Rainbow Bloon | Auxiliary/Minions | Name | bad 0.00% mean 2.48 max 15 | bakes (round 6 brings it under the bar) |
| Polar Star | Auxiliary/Items | Name | bad 1.15% mean 0.78 max 61 | bakes (round 6 brings it under the bar) |
| Quote | Creatures/Other/4 Stars | Name | bad 2.34% mean 2.95 max 203 | bakes (round 6 brings it under the bar) |
| Red Knife | Auxiliary/Items | Name | bad 1.87% mean 3.11 max 87 | bakes with a pixel note (soft parity) |
| Royal Eradicator Main Cannon | Creatures/FTL/5 Stars | Name | bad 2.02% mean 3.46 max 43 | bakes with a pixel note (soft parity); refused-layers would then refuse it: the title is split over several layers |
| Sabre | Auxiliary/Minions | Ability | bad 0.00% mean 6.37 max 14 | bakes with a pixel note (soft parity) |
| Salt | Auxiliary/Items | Name | bad 3.21% mean 2.03 max 80 | bakes (round 6 brings it under the bar) |
| Shellphone | Items/Equip/1 Stars | Ability | bad 1.05% mean 0.79 max 176 | bakes (round 6 brings it under the bar) |
| Snake | Auxiliary/Items | Name | bad 1.62% mean 0.97 max 61 | bakes (round 6 brings it under the bar) |
| Soul Sanctum | Field/2 Stars | Name | bad 1.09% mean 1.01 max 255 | bakes (round 6 brings it under the bar) |
| Spicy Spray | Items/Support/2 Stars | Name | bad 0.63% mean 0.74 max 56 | bakes (round 6 brings it under the bar) |
| Spur | Auxiliary/Items | Name | bad 0.87% mean 0.72 max 61 | bakes (round 6 brings it under the bar) |
| Stopwatch | Items/Negation/2 Stars | Name | bad 0.73% mean 1.03 max 167 | bakes (round 6 brings it under the bar) |
| The Farlands | Field/2 Stars | Name | bad 2.58% mean 2.69 max 255 | bakes (round 6 brings it under the bar) |
| The Freezer | Field/1 Stars | Name | bad 1.99% mean 1.61 max 217 | bakes with a pixel note (soft parity) |
| The Hive | Field/4 Stars | Name | bad 3.63% mean 5.05 max 255 | bakes (round 6 brings it under the bar) |
| The Hydrant | Items/Support/2 Stars | Name | bad 3.78% mean 3.59 max 180 | bakes (round 6 brings it under the bar) |
| The Kingchin | Creatures/Other/4 Stars | Name | bad 1.86% mean 3.57 max 255 | bakes (round 6 brings it under the bar) |
| The Panda Future | Field/4 Stars | Name | bad 3.36% mean 5.08 max 253 | bakes (round 6 brings it under the bar) |
| Titanium Sword | Items/Equip/3 Stars | Ability | bad 2.87% mean 2.08 max 154 | bakes (round 6 brings it under the bar) |
| White Space | Field/3 Stars | Name | bad 3.07% mean 4.62 max 255 | bakes (round 6 brings it under the bar) |
| Wooden Spikes | Items/Counter/1 Stars | Name | bad 1.02% mean 0.69 max 127 | bakes (round 6 brings it under the bar) |
| Yin Yarn | Creatures/Kirby/5 Stars | all four stats | bad 0.13% mean 4.36 max 32 (the other bars alike) | bakes with a pixel note (soft parity) |
| Zero Override | Creatures/Kirby/5 Stars | Ability | bad 0.01% mean 3.94 max 73 | bakes with a pixel note (soft parity) |


## 3. Bakes, but with a known pixel difference

These boxes bake once soft parity lands, with a note in the bake report and the commit ("baked with a pixel difference:
bad X%, mean Y, max Z"). Today the pixel check refuses most of them (last column). The numbers are psd-tools' render on
the round-6 fork against the card's PNG, over the box the edit can reach; the bar was 0.5% bad pixels and mean 2.0. The
worst is 10.00% bad, far under the 25% floor, so none is refused. They are the fork's to-do list, not the card makers'.
Not unbakeable: informational.

| Card | Folder | Field | Bad pixels | Mean | Max | Today |
|---|---|---|---|---|---|---|
| Bob-ulk | Creatures/Mario/2 Stars | Name (box new with refused-layers) | 0.66% | 6.33 | 28 | refused: the title layer has another name |
| Bolts and Screws | Items/Heal/1 Stars | Name | 1.33% | 0.87 | 40 | refused: pixel check |
| Corruptor | Auxiliary/Minions | HP | 3.49% | 2.98 | 38 | refused: pixel check |
| Corruptor | Auxiliary/Minions | DEF | 4.34% | 3.58 | 38 | refused: pixel check |
| Corruptor | Auxiliary/Minions | ATK | 4.33% | 3.57 | 38 | refused: pixel check |
| Corruptor | Auxiliary/Minions | SPD | 5.12% | 4.22 | 38 | refused: pixel check |
| Cortez | Creatures/Mario/4 Stars | Ability | 0.00% | 2.12 | 18 | refused: colour icons would turn one colour |
| Credit Card | Items/Support/3 Stars | Ability | 0.84% | 0.62 | 255 | refused: pixel check |
| Curse | Auxiliary/Status Effects/Debuffs/4 Stars | Name (box new with refused-layers) | 1.94% | 0.94 | 40 | refused: the title layer has another name |
| Dark Bramble | Field/4 Stars | Name | 0.90% | 0.47 | 41 | refused: pixel check |
| Dark Ritual | Items/Equip/4 Stars | Name (box new with refused-layers) | 0.67% | 0.74 | 40 | refused: no title layer the bot recognizes |
| Despair | Auxiliary/Status Effects/Debuffs/4 Stars | Name (box new with refused-layers) | 2.31% | 1.11 | 40 | refused: the title layer has another name |
| Elmadan | Creatures/Other/3 Stars | Name | 0.72% | 0.57 | 98 | refused: pixel check |
| Empowered | Auxiliary/Status Effects/Buffs | Ability (box new with refused-layers) | 1.73% | 0.71 | 34 | refused: the ability layer is outside the text group |
| Empress Bulblax | Creatures/Other/3 Stars | Name (box new with refused-layers) | 0.46% | 3.12 | 29 | refused: no title layer the bot recognizes |
| Enraged | Auxiliary/Status Effects/Buffs | Ability (box new with refused-layers) | 0.52% | 0.67 | 27 | refused: the ability layer is outside the text group |
| Forget Me Now | Items/Support/3 Stars | Ability | 1.27% | 0.72 | 160 | refused: pixel check |
| Frog | Creatures/Everhood/5 Stars | Name | 0.00% | 4.29 | 18 | refused: pixel check |
| Frog | Creatures/Everhood/5 Stars | HP | 0.00% | 6.55 | 22 | refused: pixel check |
| Frog | Creatures/Everhood/5 Stars | DEF | 0.00% | 4.21 | 23 | refused: pixel check |
| Frog | Creatures/Everhood/5 Stars | ATK | 0.00% | 6.31 | 24 | refused: pixel check |
| Frog | Creatures/Everhood/5 Stars | SPD | 0.00% | 6.06 | 23 | refused: pixel check |
| Golden Bomb | Items/Attack/4 Stars | Name (box new with refused-layers) | 4.83% | 5.34 | 128 | refused: no title layer the bot recognizes |
| Golden Metronome | Items/Negation/4 Stars | Name (box new with refused-layers) | 4.57% | 4.35 | 127 | refused: no title layer the bot recognizes |
| Greenpath | Field/2 Stars | Name | 0.64% | 0.32 | 41 | refused: pixel check |
| Hive Knight | Creatures/Hollow Knight/3 Stars | ATK | 0.00% | 2.02 | 14 | refused: pixel check |
| Inner Eye | Items/Equip/3 Stars | Ability | 0.55% | 0.35 | 127 | refused: pixel check |
| Jetpack | Auxiliary/Minions | Name | 0.00% | 2.15 | 27 | refused: pixel check |
| Kalecgos | Creatures/World Of Warcraft/4 Stars | HP | 0.01% | 4.14 | 25 | refused: pixel check |
| Kalecgos | Creatures/World Of Warcraft/4 Stars | DEF | 0.00% | 2.79 | 25 | refused: pixel check |
| Kalecgos | Creatures/World Of Warcraft/4 Stars | SPD | 0.00% | 4.86 | 25 | refused: pixel check |
| Little Guy | Creatures/Other/2 Stars | Name | 1.44% | 0.70 | 40 | refused: pixel check |
| Mari | Creatures/Omori/4 Stars | Ability (box new with refused-layers) | 0.00% | 4.22 | 16 | refused: the ability layer is outside the text group |
| Masked Dedede | Creatures/Kirby/5 Stars | Name | 0.00% | 2.95 | 14 | refused: pixel check |
| Masked Dedede | Creatures/Kirby/5 Stars | Ability | 0.00% | 4.59 | 24 | refused: colour icons would turn one colour |
| omori | Creatures/Omori/5 Stars | Name (box new with refused-layers) | 10.00% | 6.71 | 73 | refused: no title layer the bot recognizes |
| Panseida | Creatures/Pandas/3 Stars | Ability | 0.73% | 0.74 | 117 | refused: colour icons would turn one colour |
| Pile Of Swords | Items/Support/4 Stars | Name | 1.47% | 1.15 | 199 | refused: pixel check |
| Pizza | Creatures/Other/3 Stars | Ability | 2.23% | 1.27 | 84 | refused: pixel check |
| Red Knife | Auxiliary/Items | Name | 1.87% | 3.11 | 87 | refused: pixel check |
| Sabre | Auxiliary/Minions | Ability | 0.00% | 6.37 | 14 | refused: pixel check |
| Shell Shock | Auxiliary/Status Effects/Debuffs/3 Stars | Name (box new with refused-layers) | 2.93% | 1.35 | 40 | refused: the title layer has another name |
| The Freezer | Field/1 Stars | Name | 0.86% | 0.38 | 40 | refused: pixel check |
| The Radiance | Creatures/Hollow Knight/5 Stars | Ability | 0.08% | 2.23 | 48 | refused: colour icons would turn one colour |
| The Spire | Field/3 Stars | Name (box new with refused-layers) | 0.00% | 3.52 | 19 | refused: no title layer the bot recognizes |
| Yin Yarn | Creatures/Kirby/5 Stars | HP | 0.13% | 4.36 | 32 | refused: pixel check |
| Yin Yarn | Creatures/Kirby/5 Stars | DEF | 0.06% | 2.50 | 32 | refused: pixel check |
| Yin Yarn | Creatures/Kirby/5 Stars | ATK | 0.08% | 4.03 | 32 | refused: pixel check |
| Yin Yarn | Creatures/Kirby/5 Stars | SPD | 0.12% | 4.21 | 32 | refused: pixel check |
| Zero Override | Creatures/Kirby/5 Stars | Ability | 0.01% | 3.95 | 73 | refused: pixel check |

## 4. Edits waiting in Card Edits.json

The entries with a `_psd` verdict on origin/main, run again through `classify` and `apply_jobs` (a dry bake: nothing
committed). "After" is refused-layers on the round-6 fork, with soft parity applied to the pixel check.

| Entry | Edit | Verdict in the file (2026-10-22) | Dry run today | After the pending changes |
|---|---|---|---|---|
| Dark matter | name → 'Dark Matter' | unsupported-card: no readable Text/Name layer | refused: no readable Text/Name layer | bakes (dry run) |
| Masked Dedede | name → 'Masked Dedede' | parity: baseline bad 0.11% mean 3.01 max 136 | refused: parity: baseline bad 0.00% mean 3.07 max 14 | bakes with a pixel note (bad 0.00% mean 2.95 max 14) |
| Metal General | name → 'Metal General' | overflow: name reaches the stars or type icons | bakes (dry run) | bakes (dry run) |
| NME Stock | name → 'NME Stock™' | duplicate: numbered PNGs share the PSD: NME_Stock2.png | refused: numbered PNGs share the PSD (NME_Stock2.png) | refused: numbered PNGs share the PSD (NME_Stock2.png) |
| Papyrus | name → 'The Great Papyrus' | unsupported-card: no readable Text/Name layer | refused: no readable Text/Name layer | refused: title layer named 'The Great Papyrus', not Name or the card's name |
| PoD11ZOMG | name → 'Z.O.M.G.' | parity: baseline bad 0.03% mean 1.16 max 46 | bakes (dry run) | bakes (dry run) |
| Sans | name → 'snas.' | unsupported-card: no readable Text/Name layer | refused: no readable Text/Name layer | refused: title layer named 'snas.', not Name or the card's name |
| Star Dream | name → 'Star Dream' | overflow: name reaches the stars or type icons | bakes (dry run) | bakes (dry run) |
| Zero Override | name → 'Zero Override' | parity: baseline bad 1.96% mean 1.71 max 127 | bakes (dry run) | bakes (dry run) |

- Metal General, Z.O.M.G., Star Dream and Zero Override bake today: their verdicts are from 2026-10-22 (old bar and fork).
  The schedule (`card_bake.final`) skips a refused verdict until the PSD changes, so they wait for a dispatched run.
- Papyrus and Sans ask for the title the card already prints ('The Great Papyrus', 'snas.'). There is nothing to redraw;
  the bot just doesn't read those layers as the title (their layer names differ from the card's name).
- Masked Dedede bakes with a pixel note once soft parity lands. NME Stock needs its numbered PNG sorted out (2.2).

## 5. Limits that depend on the edit (not unbakeable)

- **Little room for longer ability text.** On the round-6 fork, appending " Deal 1 [fire] damage." to the ability of
  the 213 stoogedings cards refused on 50 of them: the text runs off the card, or "(box)" doesn't fit its text box.
  Smaller edits to these cards bake. Air Totem, Blue Marker, Bob-ulk, Cabbage, Chef Saltbaker, Clock, Cloud Soiree, Curse, Dark Mind, Deep Focus, Dr Robotnik's Roboticizer, Earth Totem, Fahr Outpost, Feast, Fire Totem, Galacta Knight (box), Gasoline, Greenpath, GrubSong, Hiveblood, Isma, Kyle, Magic Hammer, Magic Shield, Metronome, Nuclear Proliferation (box), Ominous Red Circle, Pan D Luffy, Pandevourer, Panseida, Pep Drink, Po (box), Poison Circle, Pure Vessel, Rock Bottom, Rogueport, Sandwich of Rejects, Soul Sanctum, Spice, Steady Body, Storo, Sword of Aeons, The Claw, The Farlands, The Hydrant, The Panda Future, The Ringed City, The Spire, Utica Stock Exchange, Water Totem.
- **Stats above 10** need two bar rows (Top and Bottom in one `<Stat> Dark` group); only Royal Eradicator Main Cannon's
  ATK has them. Above 20 never.
- **A title that grows into the stars or type icons** refuses (`overflow`). Jack-In-The-Box Zombie's unchanged title
  already touches them (2.9).
- **Try again later:** a PSD or PNG pushed in the last 30 minutes, more than 40 bakes in a run, or a branch that moved
  while the run worked. These are not verdicts.

## 6. How this was measured

- Scripts and raw outputs: `~/Documents/Tools/wt/scratch/unbakeable/` (`bakecheck.py` runs the bake's own steps short of
  rendering: card gates, `template_rows` + `set_stat`, `name_layer`/`ability_layer` + `set_text_layer` with the printed
  title and the website's ability text; `paritycheck.py`; `analyze.py` merges; `gen.py` and `assemble.py` write this
  file and the CSV; `~/Documents/Tools/wt/scratch/unbakeable-fix/scripts/fulltitle.py` re-checks the split and
  doubled titles). Code snapshots: main `e837f68`, `work/r6-color-glyphs` `eaf446e`, `work/refused-layers` `c31343d` +
  uncommitted (02:51; committed since as `c5c3f19`..`dc2bd69`, not re-measured).
- Corpus: the frozen copy of TTSCardMaker `6e50c3ee`. Only two card PSDs changed on origin/main since (Clot White and
  Springtrap, `7b48db05`, Txt2 title stories): both were re-checked from origin/main and their story refusal is gone.
- Pixel check: the round-6 bot-gate scans with the bot's own measuring code and per-card ICC
  (`wt/scratch/r6-critic/botgate/scan_base_eaf.json` today, `scan_final_eaf.json` after). The 101 boxes that miss the bar
  today were measured again on the live fork tag: identical. New refused-layers boxes: its own scan
  (`wt/scratch/ideas/refused-layers/scan_new.json`, worst 10.0% bad); one box (Bugzzy's ability) is not measured yet. A
  stat box is the whole bar row, a little larger than a bake's.
- Fonts: every font the titles and abilities use resolves exactly from MDW/Fonts (no EarthboundZero or other font
  refusal). MyriadArabic-Regular is named by a run on 3 cards (Ancient Altar, Dark Mind Eye, grand doomer) but draws
  nothing, so it doesn't refuse.
- The check edits each field to its current text (the printed title; the website's ability text, which on picture-icon
  cards includes the [type] tokens an editor would send) and a stat one step down, so a refusal comes from the card,
  not from the edit. The 20 titles split over several layers or printed twice were checked again with the whole
  title the website shows (a real edit sends all of it, not one layer's words), through `classify` and `apply_jobs`
  with every gate after saving, on today's code and on round 6 (same verdicts): the title as it is, for the layout
  gates, and with a letter added, for the read-back (`classify` skips an unchanged title). The added letter also
  pushes Moose Goose's title into the stars or type icons; that depends on the edit's length (section 5), so it
  counts as baking.
- Not run on every card: the gates after saving (layer bytes, Txt2 stories, read-back, re-render). They ran in the dry
  bakes of section 4, in the Warcraftian Druid, Mosling and Oryx trials, on the split and doubled titles, and in round
  6's [fire] bake on the 213 stoogedings cards (160 baked; 50 stopped on length, section 5; 2 on the old pixel bar;
  1 on an unstyled FX layer).
- Earlier list: `~/Documents/Tools/wt/scratch/ideas/unbakeable_2026-10-23.txt` (frozen bot code without per-card ICC;
  this one supersedes it).
