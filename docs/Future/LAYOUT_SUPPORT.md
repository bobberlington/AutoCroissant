# Layout support: non-template stat bars and unusual star layouts

**Status:** approved by the owner (10-24 ~18:30, "get support built rather than refusals"); designed 2026-10-25; built
2026-10-25 on branch work/layout ("Built" at the end: what the bake does, per family, and what still refuses). Queue item 5 (after the website fixes). Cards: `docs/Future/PROBLEM_CARDS_2026-10-25.csv` (31 stat-bar rows,
21 star rows). Families below come from a read-only layer scan of those PSDs on 10-25 (TTSCardMaker 577ea1e3).

**Goal.** A website stat or stars edit bakes on these 52 cards, flagged where the bake changed the layout, instead of
being refused.

**Rulings that bind it:** content and layout never refuse; only waits, impossible cases and integrity gates do (owner,
10-24 ~18:30). Card makers keep layout freedom: the bake follows their layout where it can. One PSD rewrite per card
per pass; never over a teammate's newer PSD.

## Stat bars (31 cards; `commands/psd_edit.py` `template_rows` / `_row` refuse "non-template row")

| Family | Cards | What the row is | Bake rule |
|---|---:|---|---|
| A. Template row, one deviation | 4 | Anubisath Guardian (a text layer inside `Hp Dark`), Dark Mind (a second `5 copy`), Spinni (no `5`), Whispy Woods (digits 1-10) | `_row` accepts extra non-digit layers (left alone), a duplicated digit (both toggled), 1-10 naming (shifted); a missing digit refuses only when the new value needs it |
| B. One dark layer named the value | 6 | Chef Saltbaker, Grimora, Pizza, Skeleton Mate, Slugcat (+ an Exposure layer), Uumuu (exposure) | move the layer by (new - old) cell pitches along the bar and rename it to the new value; pitch from the card's bar box / 10. A layer too short for a lower value: template swap (below) |
| C. `Dark` layer + a masked layer named the value | 9 | Tainted Bethany, Eden, Eve, Isaac, Jacob, Keeper, Lost, Magdalene; Dark Esau | one rule for the set (Isaac cards share it): rename the value layer and move its mask edge by the pitch, as B |
| D. One smart object per cell | 7 | Blank Card, Dirty Bed (HP/DEF only), Clot Black, Blood, Bone, Gold, White | template swap |
| E. Unnamed cell copies | 2 | Chris & Jordi, Zigg | template swap |
| F. Others | 3 | Minecraft Panda (one `Darks` group for all four stats), Bulborb Larva, Empress Bulblax (not matched by the scan) | dump each first; template swap by default |

**Template swap** (D, E, F, and B's fallback): when the stat's bar box matches the card kind's template bar within
2 px, the bake imports the template's `<Stat> Dark` group (fork `import_layer`, the parts editor's +ability machinery),
sets the value, deletes the old row's layers, and flags "stat bar replaced with the template's". The parity gate
judges the pixels; a box that doesn't match refuses (impossible case).

Build order by cards per rule: C (9), B (6), A (4), then the swap (12). Out of scope: Sonic's one-row SPD 20 and the
4 cards with several bars per stat (False Knight, Pix, Tainted Lazarus, Ze'mer): Photoshop.

## Stars (21 cards; `psd_edit` `_star_layouts`, `_star_units`)

| Reason | Cards | Bake rule |
|---|---:|---|
| Shape not a row, column or rows of two | 7 | God Tamer, Michael's Blessing (dice five), Lorian (2 over 3), Morpho Knight (2-1-2), Magnum Opus, Mettaton NEO (diamond), Termination Shock (triangle): `_star_layouts` learns these as patterns (a dice face per count; rows of n with an offset); a shape it still can't continue is re-laid as rows of two at the group's top-right corner, flagged, when it stays clear of the title and icons |
| Stars inside a nested group | 6 | Jack-In-The-Box Zombie, PoD04 Zebra Bloon, PoD09 DDT, Polevaulter Zombie, Springtrap, Paintbrush: read the visible leaf layers under `Stars` at any depth; the new star goes beside its neighbour in the same subgroup |
| One layer draws several stars | 3 | Cat, Living Bread, Shadow Queen: split the layer into one layer per alpha island (one per star), then the normal rules; flagged |
| No single `Stars` group | 3 | Nasty Patty, Paint Warrior, Shell Shield: take the star layers `psd_details` finds (the star detector) wherever they sit; the new star goes in the last one's parent |
| No room; smart filter | 2 | Bugzzy, Magnet-shroom: stay Photoshop (real limits) |

Folder-starred cards (575) move folders instead (parts editor, S9); this design is for printed stars only.

## Verification (lean)

- The census's dry bakes on these 52 cards (stat ±1, add and remove a star): refusals 31 -> 0 for bars (bar box
  mismatches aside), 21 -> 2 for stars.
- One live bake per family (A, B, C, swap, each star rule), opened in Photoshop once.
- Tests: one fixture per family in `tests/test_psd_edit.py`, no more.
