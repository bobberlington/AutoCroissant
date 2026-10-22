# Standardization rehearsal (2026-10-21, ledger §4 item 1)

**Status:** done on copies, 2026-10-21 22:49 to 2026-10-22 00:15. Every gate is green or has a known cause. Nothing under
`~/Desktop/TTSCardMaker` was written. The ledger is `CARD_STANDARDIZATION_CAMPAIGN.md` (entries L1-L16). **The expected diff list
in section 5 is pinned to TTSCardMaker 9f0c9ec.** While the rehearsal ran, a teammate pushed 178 PSDs (section 6, item 1), so the
day must first rebuild the manifest at its own HEAD and re-run the rehearsal on the cards that changed.

## 1. Inputs

| Input | Value |
|---|---|
| Card tree | TTSCardMaker `9f0c9ec`. It has the same PSD blobs as the manifest's basis `6359c54` (no PSD commit between them). |
| Copies | 425 PSDs and 433 PNGs: every path in the manifest, each with its PNG. They went to `/private/tmp/rehearsal/cards/` and to a pristine set in `orig/`. Each copy's git blob equals `git ls-tree HEAD` and the manifest's `blob`: 0 mismatches. The per-file sha list is in `inputs.json`. |
| Manifest | `~/Documents/Tools/psd-research/std/rehearsal_manifest.json` (blob `e7679c70`) |
| Fork scripts | `std_common.py d9bb67dc`, `l2_template_rows.py e3b400a2`, `l3_sonic.py 42b24a94`, `l5_ability.py 3bad1883`, `l6_name.py 622c7b70`, `l11_pages.py 65158533`, `l14_stars.py d36b601e` (git blob shas) |
| Photoshop scripts | `std_l1.jsx 5d7c5402`. `Effect Layers.jsx f01a683d` and `Effect Layers - All Cards.jsx ee432d80` were copied to `/private/tmp/rehearsal/scripts/`. Next to them sit copies of the chart (`Satus Effect Colors.psd 98a6b65a`) and the two `.aco` files. |
| Fork | `/tmp/wt-layer-edit` `feat/layer-edit` `26ed79c8`, clean. A snapshot (`/private/tmp/rehearsal/fork-src`) ran both the scripts and every render. |
| Parser | AutoCroissant `commands/` at `7d6c512` (unchanged through `39da9ed`), stock psd-tools 1.25 (`~/.venvs/ac-py314-psd125`). The vocabulary came from the clone: `Types/`, `Types/Stars/` and `stoogedings.json`, none of which changed during the run. |
| Photoshop | 2023 (24.0.0). The ping showed 0 documents open before every step, and every card script found 0 open at its start. |

Artifacts are kept in `~/Documents/Tools/psd-research/rehearsal-2026-10-21/` (12 MB): `inputs.json`, `before.json`, `after.json`,
`corpus_before.json`, `corpus_head.json`, `gates/`, `steps/` (parses after each phase), `tools/`, `scripts/`, `logs/`,
`std-logs/` and `score-results/`. The PSD copies stay in `/private/tmp/rehearsal/`, which a reboot clears. To rebuild them, run
`git cat-file blob <sha>` for each sha in `inputs.json`.

## 2. What ran (the day's order)

| Step | Tool | Cards | Written | Time |
|---|---|---|---|---|
| Pre-batch parse | `tools/parse_tree.py` (golden format plus the pipeline view) | 425 copies; also all 813 of the live tree | none | 10 s; 16 s |
| L7 pass 1 | `Effect Layers - All Cards.jsx` (`processCard` + `exportPng`), one card per script, 300 s timeout | 408 (the batch skips the 17 rulebook pages) | 43 PSD + PNG | 1434 s |
| L1 | `std_l1.jsx` walk, saved in place + PNG (Tainted_Lazarus left out) | 41 | 41 PSD + PNG | 186 s |
| L7 pass 2 | same as pass 1 | 408 | 1 (Palace_of_Shadow) | 950 s |
| L11 pages (Photoshop) | none: all 14 pages are the owner's (decision log) | 0 | none | |
| L2 | `l2_template_rows.py` | 100 | 79 | 157 s |
| L3 | `l3_sonic.py` | 2 | 1 | 1 s |
| L5 | `l5_ability.py` | 34 | 34 | 285 s |
| L6 | `l6_name.py` (without `--undecided`) | 142 | 90 | 42 s |
| L4 | none: the pipeline moves `lfx2` on toggle | 0 | none | |
| L11 structure | `l11_pages.py` | 17 | 5 | 3 s |
| L14 | `l14_stars.py` | 2 | 2 | 1 s |
| Recheck | all six scripts with `--dry-run` | | 0 ops left | |
| PNG export | Photoshop `exportPng` recipe on the 23 fork cards whose pixels change | 23 | 23 PNG | 38 s |
| Open check | Photoshop opens and closes each 0-pixel fork-written PSD | 172 | none | 114 s |
| Gates | parse, goldens, composer, harness (fork renders of `orig/` and `cards/`), Photoshop truth (section 4) | 425 | | about 25 min |

In total, 264 PSDs and 102 PNGs were written: 81 by Photoshop and 195 by the fork scripts, with 12 cards written by both.

Every Photoshop step ran with no error, no timeout and no dialog. That covers 1,177 card scripts, counting the baseline exports and
probes of section 4.

Wrappers live in `/private/tmp/rehearsal/scripts/` and `tools/ps_driver.py`:
- The All Cards script is used unedited.
- One line in the `Effect Layers.jsx` copy is patched: line 529 hides converted overlays instead of deleting them (ledger L7).
- `std_l1_card.jsx` is the `std_l1.jsx` walk unchanged. Only its I/O changed: it opens the copy, saves it in place, exports the
  PNG, and on error closes only its own document.

Each Photoshop step writes the blob of every PSD it changed to `cards/.std-logs/photoshop-*.json`. The fork scripts' stale check
reads those logs, so it accepts the cards Photoshop rewrote.

## 3. Per-entry results

- **L7.** Exactly the manifest's 43 cards changed, and 365 stayed untouched. The overlays were hidden, not deleted. Parse
  unchanged and 0 `EFFECT LAYER` problems. Pass 2 rebuilt only Palace_of_Shadow, whose FX layer L1 had made stale. Every other
  card was unchanged, so the batch is idempotent.
- **L1.** 41 of 41 cards migrated: every token placed, and as many icon layers hidden as tokens. The ability parse is
  byte-identical on all 41 (no gap-whitespace drift either), and no icon-token problems appeared.
  - The Single-line composer was kept.
  - The new font icons now take the `Text` group's stroke and shadow. That is a look change inside the ability region.
- **L2.** 79 written:
  - 7 slider cards got template rows; their pixels change (Dark_Esau and 6 Tainted cards).
  - 8 deco cards got `art <n>` renames.
  - 4 L2b cards and 61 L2c cards had their bars renamed.
  - Sans had its hazard renames.

  hp, def, atk and spd are unchanged everywhere. The pipeline now accepts the rows on the slider and L2b cards. Of the 36 L2a
  cards, 19 are not done by the script:
  - Grimora and Skeleton_Mate: Photoshop by hand.
  - Tainted_Isaac and Tainted_Eden: the owner.
  - 17 "drawn bar art this script cannot rebuild" (Photoshop per card, though the manifest says fork): Chef_Saltbaker, Zigg,
    Chris_&_Jordi, Ze'mer, Slugcat, Pizza, the 5 Clots, False_Knight, Minecraft_Panda, Bulborb Larva, Empress Bulblax,
    Dirty_Bed, Blank_Card.
- **L3.** Sonic became Top + Bottom with markers. It still parses spd 20, STATS TOO HIGH stays, and the pipeline now reads
  "ok 2 row(s) reads 20". The fork render is identical. Royal Eradicator was left alone, as the manifest says.
- **L5.** All 34 moved under `Text`:
  - 16 change pixels; their PNGs were re-exported.
  - 8 are 0 pixels and fork-identical.
  - 10 differ by at most 1 level in the fork render, and Photoshop exports them identically.

  The ability text is unchanged, and the pipeline now accepts the ability on all 34.
- **L6.** 90 L6a titles renamed to `Name`. Knight_Lost_A_Lot is held. The 4 L6a moves are held. The 45 L6b cards (Wigfrid among
  them) are held as not decided. The 6 L6c cards are listed for Photoshop. 7 cards gain a `display_name` (section 5).
- **L11.** The 6 hidden layers on 4 pages were deleted; the characters lost equal the manifest's (480, 250, 815, 562). `tornado`
  was renamed `wind`. The 14 owner pages were untouched.
- **L14.** God_Tamer now reads 3 stars and Bubble 1; NO STARS FOUND is gone. The fork render is identical.
- **L4, L9, L10, L12, L13, L15:** no PSD change. L10 was checked: 412 visible Ability and 402 visible Name layers, all
  Single-line, before and after.
- **L8, L16: not rehearsed.** L8 runs through `card_bake`, which reads GitHub, so its rows come from the manifest. L16 is the
  team's re-export.

## 4. Gates

- **Parser gate: green.**
  - Copies, after vs before: 425 cards and 264 written. There are 20 field differences on 14 cards, all of them expected (section
    5), and 0 unexpected.
  - Every manifest check holds: the stat values, `display_name`, 0 icon-token problems on L1 and 0 `EFFECT LAYER` problems on L7.
  - The copies parse equal to the live 813-card pre-batch parse at 9f0c9ec. So the 813-card gate equals the same 20 differences,
    with every other field of all 813 cards unchanged.
  - The only intermediate red was after L1 and before L7 pass 2: Palace_of_Shadow showed `EFFECT LAYER OUT OF DATE`.
- **Harness gate, 0-pixel entries: green.** Fork renders of each copy after the batch were compared with the same renders of the
  pristine originals (fork snapshot 26ed79c8):
  - 152 of the 162 PSDs written with no intended pixel change are bit-identical.
  - The other 10 are the L5 moves expected to change 0 pixels. Their fork render differs by at most 1 level, and Photoshop's
    exports before and after are identical.
  - All 161 untouched copies are identical.
- **Harness gate, pixel-changing entries.** These are the 102 re-exported PNGs: 43 L7, 41 L1, 16 L5 and 7 L2, with 5 cards in
  two entries.
  - **Photoshop's own output, green.** A Photoshop 2023 export of each original PSD was compared with the export after the batch.
    The change stays inside each entry's regions on all 102: ability and ability_fx for L7; ability, ability_fx and the hidden
    icons' boxes for L1; ability for L5; the stat rows for L2.
  - **Fork renderer vs those exports, with re-export drift removed.** Of 166 touched regions, 106 are within the bar (bad ≤ 0.5%,
    mae ≤ 1.0), 49 are outside it but not worse than before, and 11 are worse, on 10 cards.

    | Card | Entry | Region | Before (bad%, mae) | After (bad%, mae) |
    |---|---|---|---|---|
    | Tomorrow_I'll | L5 | ability | 0.193, 0.598 | 8.805, 6.355 |
    | Stotch | L5 | ability | 0.133, 0.592 | 2.541, 1.265 |
    | Spamton_Neo | L5 | ability | 2.097, 0.883 | 3.442, 1.055 |
    | Snow_Angel | L5 | ability | 0.121, 0.794 | 0.598, 1.396 |
    | The_Big_Yellow_Cat | L5 | ability | 0.091, 1.12 | 0.429, 1.39 |
    | Yin_Yarn | L5 | ability | 0.003, 1.736 | 0.146, 1.268 |
    | Marx | L5 + L7 | ability | 7.828, 5.442 | 8.577, 6.888 |
    | Marx | L5 + L7 | ability_fx | 8.505, 5.323 | 8.83, 5.6 |
    | Panseida | L7 | ability_fx (new) | none | 2.965, 2.628 |
    | Hexed | L7 | ability_fx (new) | none | 0.005, 1.345 |
    | Credit_Card | L7 | ability_fx (new) | none | 0.761, 1.029 |

    Every one of these is a renderer gap, not a PSD fault:
    - The L5 cards gained the `Text` group's DropShadow. That effect is in the harness's `MISSING_FX` list.
    - The new chart FX layers carry an OuterGlow, also in `MISSING_FX`.
    - Marx is class D.

    Photoshop shows the intended look on all 10. The 60 raw FAILs against the official PNGs are this list plus the re-export drift
    (section 6, item 3).
- **Goldens: green.** 25 golden cards are among the copies.
  - 22 parse to their golden. 15 of those PSDs were written by the batch but parse the same.
  - 3 differ exactly as the manifest expects: A_Snack's `display_name` becomes "A snack", God_Tamer's stars 0 -> 3 with the problem
    gone, and 20_Creature_Types gains `wind` with the problem gone.
  - On the day, re-pin the 18 written goldens with `tests/make_goldens.py`.

## 5. Expected per-card diff list (the day's live gate must equal this)

Basis: TTSCardMaker 9f0c9ec. On the live tree, all 813 cards must keep every other field (`card`, `alt_names`, `validator`).
There are 26 rows on 20 cards: 20 rows on 14 cards were rehearsed and 6 come from the manifest. Machine-readable:
`gates/expected_diff.json`.

| Card | Field | Before | After | Entry |
|---|---|---|---|---|
| Auxiliary/Minions/God_Tamer.psd | stars | 0 | 3 (PENDING: the card prints 5) | L14 |
| Auxiliary/Minions/God_Tamer.psd | problem | NO STARS FOUND (…) | none | L14 |
| Auxiliary/Minions/God_Tamer.psd | validator | NO STARS FOUND (…) | [] | L14 |
| N.M.E/Bubble.psd | stars | 0 | 1 | L14 |
| N.M.E/Bubble.psd | problem | NO STARS FOUND (…) | none | L14 |
| N.M.E/Bubble.psd | validator | NO STARS FOUND (…) | [] | L14 |
| Auxiliary/Rulebook/20_Creature_Types.psd | types | ice, fire, panda, rock, machine | ice, fire, panda, rock, wind, machine | L11 |
| Auxiliary/Rulebook/20_Creature_Types.psd | problem | MISSPELT TYPE: tornado | none | L11 |
| Auxiliary/Rulebook/20_Creature_Types.psd | validator | MISSPELT TYPE: tornado | [] | L11 |
| Auxiliary/Rulebook/09_TakingYourTurn.psd | ability | 1935 characters | 1455: loses "A creature can choose any creature on the playing field to ATTACK…" (254) and "A creature can use its action to go into DEFENSE MODE…" (226) | L11 |
| Auxiliary/Rulebook/11_Abilities.psd | ability | 1119 characters | 869: loses "PASSIVES / Passive abilities are abilites that are in constant effect…" (250) | L11 |
| Auxiliary/Rulebook/12_Reactions2.psd | ability | 1665 characters | 850: loses "Although reactions happen in reverse order, anything that deals damage…" (815) | L11 |
| Auxiliary/Rulebook/22_Negation.psd | ability | 2690 characters | 2128: loses "- If an item is negated, it means it cannot be played or is taken out…" (562) | L11 |
| Creatures/Hollow Knight/4 Stars/Grey_Prince_Zote.psd | display_name | none | "Invincible Fearless Sensual Mysterious Enchanting Vigorous Diligent Overwhelming gorgeous Passionate Terrifying Beautiful Powerful" (HELD: epithet only; the title "GREY PRINCE ZOTE" is two more layers) | L6a |
| Creatures/Mario/2 Stars/Ms_Mowz.psd | display_name | none | "Ms. Mowz" | L6a |
| Creatures/Omori/3 Stars/Mr_Jawsum.psd | display_name | none | "Mr. Jawsum" | L6a |
| Creatures/Undertale/3 Stars/Papyrus.psd | display_name | none | "The Great Papyrus" | L6a |
| Creatures/Undertale/5 Stars/Sans.psd | display_name | none | "snas." | L6a |
| Items/Heal/2 Stars/A_Snack.psd | display_name | none | "A snack" | L6a |
| Items/Heal/2 Stars/Thunder_Thighs.psd | display_name | none | "Thunder thighs" | L6a |
| Auxiliary/Minions/Clot_White.psd | display_name | "White CLot" | "White Clot" | L8 (pipeline, not rehearsed) |
| Auxiliary/Minions/PoD08MOAB.psd | display_name | "M.O.A.B" | "M.O.A.B." | L8 (pipeline, not rehearsed) |
| Auxiliary/Minions/PoD09DDT.psd | display_name | "D.D.T" | "D.D.T." | L8 (pipeline, not rehearsed) |
| Auxiliary/Minions/PoD10BFB.psd | display_name | "B.F.B" | "B.F.B." | L8 (pipeline, not rehearsed) |
| Auxiliary/Minions/PoD11ZOMG.psd | display_name | "Z.O.M.G" | "Z.O.M.G." | L8 (pipeline, not rehearsed) |
| Auxiliary/Minions/PoD12BAD.psd | display_name | "B.A.D" | "B.A.D." | L8 (pipeline, not rehearsed) |

Three kinds of item add rows of their own once decided: the owner's items (L6c retypes, L6e, L5 blank and hidden, the L11 owner
pages, the by-hand L2a cards), the held ones (Knight_Lost_A_Lot, L6b with Wigfrid) and the L1 additions of section 6, item 5. The
parse is unchanged on all 41 L1 cards and all 43 L7 cards, and hp, def, atk and spd are unchanged on every card.

## 6. Unexpected differences and their causes

1. **The tree moved during the rehearsal.** Zomoroko pushed "cooldowns (2)" to "(5)" between 22:39 and 23:06: 178 PSDs, 115 of
   them among the manifest's 425. A read-only parse of HEAD `73904db` (`corpus_head.json`) shows the change is `ability` text only
   on all 178 (old timing tags became cooldown tags), with no pipeline-view change.
   - The structural expectations above still hold on those cards. Among them are the L6a cards Ms_Mowz, Mr_Jawsum, Papyrus and
     Sans.
   - But the manifest's blob shas are stale for 115 cards, and the fork scripts refuse such cards by design.
   - The day's pre-batch parse must be taken at the day's HEAD.
   - The edit also breaks `tests/test_parser_goldens.py::test_certified_goldens_are_current` on the live tree right now: The_Freezer
     changed upstream. When it is re-pinned, `test_certified_values` must expect "(1GCD)" instead of "(1PGT)".
2. **L1 made an FX layer stale.** The reflow made Palace_of_Shadow's `Ability FX: +2 DEF` stale. The second L7 pass rebuilt it, so
   the day runs L7, then L1, then L7 again. The second pass is only needed on the three L1 ∩ L7 cards: Palace_of_Shadow,
   Gasoline and The_Hydrant.
3. **Re-exporting moves pixels outside the edit.** Exporting the unchanged original PSD with Photoshop 2023 differs from the
   official PNG on 64 of the 102 cards (1.06 M px over 24 levels). The differences sit at the edges of header icons, stars and
   titles, and on the same cards in untouched regions.
   - 75 PNGs come out RGB instead of RGBA. Eight official PNGs had semi-transparent edges that the re-export makes opaque:
     Skullord 7200 px, Engorged 4796, Warcraftian_Mage 2753, Panseida 2508, Monkey_D_Goofy 2364, Stranger 1306,
     Doctor_Zomboss 1306, Stotch 29.
   - Monkey_D_Goofy's harness ICC vote flips from converted to raw.
   - The batch did not cause any of this; re-exporting did.
4. **Fork rounding of ±1 level** on the 10 L5 moves of section 4. Photoshop exports them identically.
5. **Two inline icon cards are missing from L1's list.** Alex_YIIK's `Dark` (`[dark]`) and Pandevourer's two `Machine` layers
   (`[machine]`, `[machine]`) are inline icons. The L2 script says so, and the parser injects them. A probe on scratch copies
   (`smoke/l1extra/`) migrated both with a byte-identical parse. Add them to L1 (43 cards).
6. **17 L2a cards are not done by the script** although the manifest marks them fork (section 3).
7. **The font has new badges that `std_l1.jsx` does not know.** stoogedings `8ec1b925` (cooldown and uses badges) landed during
   the run. `std_l1.jsx` and `l1_check.py` still match only the old tags in `TAG`. Four L1 cards already carry the new tags: The
   Freezer, Nebulous Precipice, Dark Bramble and Error Room use (1GCD). Before the day, `TAG` needs
   `[1-9X]G?CD|[1-9X]U|(?:-1|[0-9X])\/[1-9X]G?CD`, and Photoshop needs the rebuilt font installed and a restart.

## 7. What the day still needs from the owner

- **The go, on a quiet tree.** Teammates are mid-push. Once they stop, the manifest is rebuilt at that HEAD (blobs and expected),
  and this rehearsal is re-run on the changed cards before the live batch.
- **The open answers:**
  - Knight_Lost_A_Lot: held.
  - Grey_Prince_Zote: held (check 2026-10-22: its L6a rename reads only the epithet block; the printed title is the
    separate "GREY PRINCE" and "ZOTE" layers, the same partial-title case as Knight_Lost_A_Lot).
  - Wigfrid's "Wigfrig": typo?
  - God_Tamer's star count: the parser merges 5 flowers into 3 (row pending, not expected, until answered).
  - Bubble's star count: 1?
  - L6b: move 45 `Name` layers under `Text`?
  - The six case-typo titles (L6e).
  - Tainted_Isaac (6 hearts drawn, named 5) and Tainted_Eden (4 drawn, named 7).
- **Accept the re-export look:** the drift and alpha changes of section 6, item 3, on 102 PNGs. The alternative is exporting on
  the team's Photoshop.
- **Photoshop work by hand:**
  - the 17 drawn-bar L2a cards, plus Grimora and Skeleton_Mate;
  - the six L6c retypes;
  - the L5 blank (20) and hidden (5) content;
  - the 14 L11 pages, once the decision log rules.
- **Photoshop access to the clone.** A script started with `open -a` cannot read `~/Desktop`. Either grant Photoshop Files &
  Folders access, start the two batches by hand, or run them on `/private/tmp` copies and copy the results back by blob check (as
  here).
- **Not the owner's:**
  - The fork scripts' guard refuses the clone; they need an explicit live mode, with their `.std-logs` kept outside the repo.
  - The day's order gains three steps: L7 again after L1; a Photoshop PNG export of the 23 fork cards whose pixels change; and a
    Photoshop export of the pre-batch PSDs, so the harness gate is drift-free.
  - The harness gate takes the 10 renderer-gap cards above at this rehearsal's numbers, plus the user's Photoshop check.
  - L8 gets its own dry run through `card_bake`.

## 8. Reproduce

```sh
R=/private/tmp/rehearsal; STD=~/Documents/Tools/psd-research/std; FORK="PYTHONPATH=$R/fork-src ~/Documents/Tools/psd-tools/.venv/bin/python"
/usr/bin/python3 -I $R/tools/copy_inputs.py                                  # copies + inputs.json
~/.venvs/ac-py314-psd125/bin/python -I $R/tools/parse_tree.py $R/cards $R/before.json --pipeline
/usr/bin/python3 -I $R/tools/ps_driver.py L7 $R/lists/l7_all.txt             # then L1 lists/l1.txt, then L7 again
cd $STD; for s in l2_template_rows l3_sonic l5_ability l6_name l11_pages l14_stars; do eval $FORK $s.py $R/cards rehearsal_manifest.json; done
/usr/bin/python3 -I $R/tools/ps_driver.py export $R/lists/fork_pixel.txt; /usr/bin/python3 -I $R/tools/ps_driver.py open $R/lists/fork_zero.txt
~/.venvs/ac-py314-psd125/bin/python -I $R/tools/parse_tree.py $R/cards $R/after.json --pipeline
/usr/bin/python3 -I $R/tools/parse_gate.py $R/before.json $R/after.json $R/gates/parse_gate
~/.venvs/ac-py314-psd125/bin/python -I $R/tools/golden_gate.py $R/cards $R/gates/golden_after.json
# harness: score_cards.py copies in $R/score_orig, $R/score (ROOT = orig/, cards/), $R/score_psbase; then tools/render_gate.py and tools/ps_gate.py
```
