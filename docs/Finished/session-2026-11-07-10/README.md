# Session 2026-11-07 to 11-10: linked pictures, automatic flags, clean cards, card history

From the 11-06 wrap-up (AutoCroissant dc9f6745, 2026-11-06 09:27 PST; TTSCardMaker d460db0e) to 11-10 ~05:00. The owner
answered questions on 11-07, 11-08 and 11-09 (morning and evening); the rest ran unattended, with research rounds beside.
387 AutoCroissant commits; 313 TTSCardMaker commits (most ours; Card sync's and Card guards' bot commits and 10 by card makers among them).

Heads at the end: AutoCroissant cff351a4 plus this wrap-up's docs (deployed: jetson cff351a, fork @170a3fce),
TTSCardMaker 581b61c1, fork tag `cards-20261110` = 170a3fce (released 11-10). The Desktop TTSCardMaker clone was
replaced by a fresh blobless full-history clone. What is left: `docs/NEXT_SESSION_PLAN.md`.

## Landed

| What | Where | Notes |
|---|---|---|
| Alt rollout fixes | TC 1912f245, 0e343d5d, 1d8217ad, e1c5cb3d, 1dceda4f, 813b1f6d | Owner's 11-07 rulings: ATK and DEF bars back on their rows (18 cards, misnamed art such as Peter Panda's), dark cells as see-through as before (23), closest look for Royal Eradicator, Kalecgos, Corruptor; Hive Knight and Tyler restored; Palace of Shadow and Inkhole colours; Retro Kirby's data fixed |
| Templates, several creatures | AC 77db6f09, a8f476e0, ea3aefa2; TC 4b587b5d, 6f0dfadb | The Alt templates took the old paths (Creature_Template.psd, Raid_Boss_Template.psd). 2 and 3 creature templates on 0-20 bars. Every creature's bars are read, baked and edited |
| Automatic flags | AC 12f3c68d, 0fe5c091, ef2f1d78, 150ca8be; TC 4e81dcb0, f966e160, 138af311, a6eff11c | `hand` and `bars` are set and cleared by card_checks, each with a one-sentence note; sweeps keep people's flags. The problem CSV and .md are deleted. A title in the picture, an added type or star: not flags (AC 09c1d51e, eef7173d) |
| Rulebook | TC 411af016, 6e61f8a5, 746a9c8a; AC 125ecc16 | The Stoogedings chapter is on the website only (`{.web}`): not in the TTS PDF, not in the bot |
| Font from trimmed Types | TC b1c07712 | stoogedings rebuilt from Croissant's 18 re-saved Types files, pushed alone |
| PSD size sort | TC dd616258 | The card browser sorts by PSD size; the size is shown on the card page |
| Fork pin | fork d8882e5f; AC cae910b7 | `cards-20261107`: research rounds 1-3 plus links-1107 (`link`/`relink`, `strip_merged`) for the big pass |
| Big pass: linked pictures | AC f6dc63ae, ee26ff79, 7f1ba665, c4385338; TC db8b71f7, 17 batches 040700fb..2cf328c6 | 11-07/08, one rewrite per PSD: 793 PSDs (768 cards, 8 templates, 17 MDW/Unfinished). Every image smart object is linked (stars and types to Types/, the rest to MDW/Linked as "<Card> - <layer>"). Sub/superscript text made normal, merged image stripped. Stat bars stay embedded. 937 MDW/Linked files added; PSDs 16.94 -> 11.10 GB. Links are dated 2000-01-01 UTC; hooks in ~/Downloads date the files |
| Links upkeep | AC 0dfc0646, 90d03921, 23618b4e, dc853b17; TC 66efec76, 2ae8384c, dad0eb0b, 95358cbb, c7a4b2b7 | The guards (all silent) convert later pushes, delete unused linked files, merge byte-identical ones, and put back deleted files that are still linked. Card sync redraws the cards using a changed linked file (proved: c7a4b2b7 -> Card sync 0375789a). Fills, adjustments and gradients stay embedded (17 cards back); Springtrap's star stays shared (13). MDW/Assets and Croissant's Special Folder linked: 677 -> 463 MB |
| Records on the data branch | AC ef0a69a2, b63c5c6e, 7044cdba, e4ed55bf; TC 4334eb4d, c9c2e739, 71242879 | Render and spot records live on the data branch and Card sync keeps them; AC's seed copies are gone |
| Status words | AC dac5656a, 7b02036c, e3588300, 5dc76e95; TC c665f522 | The guards give capitalised chart status words their chart colour (and the chart's FX for styled ones), silently. Sweep of 17 cards |
| Text colours | AC 5b002d78, a8d52a82, 652f1b2c, 8a902b75; TC b2409ea0, 7531d1a7, f418120e | Website colour menu (stored as Card Edits `colors`, baked). Card pages, TTS Card details and the editor mock-up show the card's own accent colours; low-contrast text gets a thin outline |
| Fonts Photoshop can save | TC ce778210, 7a36c265; AC b161f5ac | The 11 cards Photoshop 2023 could not save (Hero Cap, Love, Motivation, Regret, Terry Hintz, Joel Miller, Future Niiue, Elmadan, Niiue, Cave Of The Past, The Maelstrom): the fault was in EarthboundZero.ttf (Findings). Font repaired, hand flags cleared. Fourside, Onett and Twoson fixed the same way. The guards repair a pushed font |
| Stoogedings upkeep | TC 76b58bf6, 37361aea, e8258a04; AC bb57f284 | The Actions rebuild the font after a Types/ push, incrementally: unchanged glyphs copied byte for byte. When a glyph changes, Card sync redraws the text layers using it |
| Single-line Composer | TC 593afe1d, 6edc3ad0, ce94082e, 01f05876, 83e6191a; AC 6dd396c2, 3fe7b229 | 316 cards set where no line moves; Duel Flag fixed. The guards set it on later pushes |
| Titles | AC 0fe3bf16; TC 99594156, 429c48df, 05148972, d93bef5d | Split titles supported (164 cards' title parts named Name). Man Serpent Shield's title is text now. Hidden wrong-name layers dropped |
| Stats | AC 07dcc495, 4597e015; TC 380b856e, ac57d748, 12bc1331, ad0430c1, 6009f069 | Stats drawn as counted pictures bake. Custom displays cap edits at 10 or their room, refused at edit time. Sonic, Computer Virus, Monkey D Letion, Sans, The Thin Man, Pure Vessel, Retro Kirby, Tyler, Ze'mer fixed |
| Edits | AC eef1b6b4, 875a75d4, 05a4f0ce, 039721b4, ca32e8ee | Added stars and types step down until they fit. Stale Ability FX layers (Marx's Glued) re-cut. A paint or texture on the stars (Paint Bomb, Paint Warrior, Yin Yarn) follows added stars |
| Browser | TC a6eff11c; AC 72b4876c | The Morgue (hidden by default), Needs handling at the bottom, 4 useless flags removed. Auxiliary items take their item type from the icon (new Field type) |
| Hidden layers, folders | AC a4ebd5df, 2930c694, 28a60fc3, 15ba6265, 4a638081, 0563de46; TC 5743332b, e29ab18c, 2d64f477, 22394c30, 7d1b1b48, 13139db4, 30cb59a2, 4b475464, 2c49f963 | 11-09, 8 batches, 755 PSDs. Hidden type and star layers deleted, Types icons relinked to Types/ wherever they sit, hidden exact art copies and dead data removed (Jar of BB 63.8 -> 2.6 MB). Card folders follow the templates (new Overlay Text and Overlay folders). Parse unchanged; 36 cards change visibly (icons redrawn from Types/). The guards keep it on pushes |
| Clean pass 2 | AC b13610c2, 6f12e7c5, 97da324c, b3f53660, 2c60183c; TC 570ac911, ea2fee7b, a2f5d245 | 242 cards. Hidden leftovers on 225 (adjustment and fill layers, "Line if needed"/"LINE" shapes, stray name copies, empty layers). Hidden Ability copies on 9, after a parser fix to read the visible pieces. Folders on 17 (10 keep their look via Overlay Text). Middle-of-picture icons filed on 5. Parse identical on all 242 |
| Card history relevance | AC c20a6a9b, 6613dd0d (`commands/png_changes.py`); TC d1d23510, 76fdceb4 | "Updated" dates and Recently updated move only on real changes. An image version is listed only when about 5% of pixels visibly change or the data changes; the rest fold under "minor changes" (live site 22:19: 2,445 minor versions on 668 cards). Formatting-only text versions merged |
| Status effect cards | AC 1c5f08c4, 5cbf2922, 6f4a4fd1, 7443d785; TC e4aa8ff9, c01334aa | 64 status cards and Status_Effect_Template.psd: visible text in Krabby Patty (icons stay stoogedings), titles in chart colour with chart FX for styled ones, own stroke kept. The guards keep it on pushes |
| Proposal forum votes | AC 9e1e3c2c, 0660e009, 0331245c (deployed 0331245) | Toggleable module (`/forum_votes`): Open tag, reactions, "Closing Soon" with a role ping at 6 days, close, lock and archive at 7. Live test 11-09 22:12-22:32 passed (👍 won; note, Closed, locked, archived). Then (owner): no forward; the results channel gets one message, the outcome and the post's link with the proposal's first message quoted (AC 19f39f4c); none when 👎 wins or ties or nobody voted; :abstain: never counts. The test post is kept |
| Guards' sparse copy | AC 7443d785, 7735f28d (deployed 7735f28); TC c01334aa, 4fa6f6a4 | status_font.py and psd_details.py added to the guards' file list. Without psd_details.py the folders fix ran on no push (Findings) |
| Research: Actions | AC a8194f86, 9b5c14b4, f80a9714, e97d49b1, 28c23d50, c2d26b39; TC 7e39788c, d44ac853, 912706d3, d59ecaf6, ee97d961 | Card data parses in its download workers (forked workers on a runner). The guards check each blob as it lands and land past 32 MB by git. Site build deploys in its build job and hands a stuck deploy on. oxipng is cached, with a fallback download |
| Research: bake and previews | AC e452efc4, 23ebbe39, ce913bc8, 2da97f75, 6cba30dd, 657a29f9, 8a1787e3; TC 160a79c4 + 20 Mock-up commits | Previews dispatch first, check every second, bake while fonts restore. Mock-up closer to the bake (titles, comp bars past 10, colours). An Ability off the card's bottom moves up. Parser version 7 (how text is set) |
| Research: card data and TTS | AC b3a3ea05, 694bcdf5, ae1b9d69, 8bdceb93, c6c08e5e, b892f1db; TC ac96c562, 265db42a, f780c721, 0e5ff719, 1fbeb441 | ABILITY_EXCLUSIONS gone. Parser fixes (type icons only inside the Ability, one-line labels, quotes kept). Busy bar rows read by cell lightness. TTS: five Rules buttons, details show only the stats a card has, CDMG / true DMG / reactive link their chapters; TTS checks take about 3 minutes |
| Research: query and site | AC bd9093ba, 62d07950, 14335c83, 3c93c730, c24a17d7, 445e0037, 42ab06c4; TC e8547c48, 586371ee, c7c9ce44, ae94e31f | /query: guesses only cards, one "did you mean" reply, `types==` works, grids in the website's order, curly quotes. Site: stat word aliases, Esc leaves a typed field first, Compare Remove keeps the shown cards, phone close button |
| Future docs | AC 60f9e231, c3d76d5c, 1a3d8788, dca8abc7, f34bea85 | docs/Future: CARD_CREATOR, SIMULATOR, TTS_CARD_FORMS (Stage 0 test passed), CARD_TIMELINE (plan). Skill descriptions shortened |
| Card makers | TC 77f42dc5, ba4a3784, c756b58a, 3c0511d5, 26cc7a66, 637fb655, 8a40d477, 91360b31 | Croissant (linked Alt template and trimmed Types, Daco removal, Springtrap, soul sanctum), Zomoroko (The Great Card Renaming, 78 website edits), netverk (art sources) |
| Despair, Withering titles | TC 04cb8bd0 | Owner 11-09 ~22:20: their black titles on their own black stroke get a white stroke (same size and position), card only; the guards keep a title's own stroke |
| Second pictures | AC 496138f3 (deployed); TC c6f19837 | Gust of Trees and Stubborn Queen are drawn from their base PSDs: the layers tagged #1/#2, the duplicate's PNG name kept (a dup_of PNG maps to picture n); a website name edit bakes into its own picture; a person's pushed duplicate is kept. Stubborn Queen redrawn with today's bars. Their hand flags gone; the other 6 second pictures were already tagged |
| Research ships at the wrap-up | bake r16-20: AC ..9d78d7f6, TC ..e994daa0; card data r5-8: AC 4d45d47a, TC 9cc4c412; site r4-5: AC 7b9ce83b, TC 6a4b8d49 + 3accc453 | Rounds stopped at boundaries: psd r15, bake r20, site r5, card data r8, actions r25. The actions r21-25 ship did not land (see Left). Gains table: Findings |
| Fork release | fork 170a3fce = `cards-20261110`; AC cff351a4 (deployed) | Rounds 1-15 + pass-through clip + lfx2 set_data + an x86 vibrance fix. Proven on arm64 (Mac, jetson) and the GitHub x86_64 runner (AVX-512): fork suite green, all 6 compiled modules build from the git URL; AC 768/768 parse identical; guards dry runs clean. Moves Matterling and The Big Yellow Cat nearer Photoshop |
| Skills | AC 7af73f75, cff351a4 | Status card standard, picture-change rule, incremental font build, the Card-sweep trailer, the guards' sparse-list rule, the pass-through rule; the new pin |
| Fresh TC clone | ~/Desktop/TTSCardMaker | Blobless from GitHub, full history (was shallow), the same git limits and gc settings; the tests' old blobs copied from the old clone (no download); AC suite 925 passed, 1 skipped, as before. The old clone (76 GB) deleted; its unlanded branches bundled in `scratch/s1109/wrap/` |

## Findings

- **Photoshop 2023's "General Photoshop error" on 11 cards was a font fault.** In MDW/Fonts/EarthboundZero.ttf the bdat/bloc
  tables shared EBDT/EBLC's data. The PSDs were fine. Splitting the tables fixed every save (TC ce778210). Fourside, Onett
  and Twoson had the same fault.
- **Photoshop marks a link "changed" unless the file's date equals the link's stored date exactly (UTC).** None of 14
  stored variants stopped it. Hence the fixed 2000-01-01 date and the hooks. A real edit is still noticed by its size.
- **Linked .psb files appear** because a smart object made from layers stores a Photoshop document.
- **Photoshop draws a pass-through group with any clipped layer (even a hidden one) as if it had a layer style.** Deleting such
  a clip changes the picture: never delete one (AC 2c60183c). Fork branch `work/passthrough-clip` (78e765ee) matches it.
- **`psd_links._tree_at` ran `ls-tree` without `-z`.** Git quotes non-ASCII paths, so 2 linked files (Crystal Guardian's,
  Dr Robotnik's Roboticizer's) would never redraw their cards. Fixed in AC dc853b17.
- **The guards run from a sparse copy of AC (`SPARSE_FILES`).** A module imported only where it is used must be listed.
  `folder_reads` imports psd_details, and `standard_folders` catches any exception as a refusal. On GitHub the missing
  module therefore refused every folders fix quietly (AC 7735f28d).
- **A commit without `Card-sweep: <kind>` posts its cards as "New art".** The 8 hidden-layer and folders batches and clean
  pass 2 batches 1-2 had no trailer, so Card sync posted about 950 cards (755 + 194) to Discord. Batch 3/3 and the status
  font sweep carry it (memory `maintenance-pushes-silent`).
- **Croissant's push ba4a3784 deleted a linked file that 58 cards use** (Daco_35508). It was restored, and the guards now put
  back deleted files that are still linked. G4's missing-file restore had been broken since 6037f26d (fixed in 90d03921).
- **Photoshop alerts sat over the owner's game (11-08).** A research lane's test copies had an lfx2 block at version 1 (the
  fork's `set_data`), and Photoshop 2023 reads only version 0. AC 8066fe94 writes version 0. Photoshop now runs in the
  background (`open -g`) with an alert watcher (memory `photoshop-no-interrupt`).
- **The Linux runner can't rebuild the committed stoogedings byte for byte** (GH-F1 red), hence the incremental build.
- **ZIP channel compression:** Photoshop reads it but saves RLE again. It saves about 24% in git, history grows about 4.8x on
  Photoshop pushes, and Quick Look shows the cards transparent. Not adopted.
- **The big pass's own cost:** GitHub's repo size went 28.08 -> 33.57 GiB, not yet counting the last batch. The Desktop
  clone held 14.55 GiB of loose objects (no repack allowed).
- **Before the relevance rule**, 762 of 776 cards showed "updated 11-09" (maintenance passes) in a local build.
- **zsh `$c:refs/heads/main` drops the `:r`;** write `${c}:refs/heads/main`. Three batch pushes failed in 0 s on it.
- **Process hygiene:**
  - `pkill -f` on the shell-snapshot path killed every shell in the session (memory `never-pkill-shell-snapshot`).
  - 30 orphaned Python processes (pool workers, http servers) were found on 11-09.
  - A lane printed another app's process environment, which held a token. Never list other processes' command lines or
    environments.
- **Research gains** (each row checked against its source; full table in `scratch/s1109/wrap/gains.json`): fork, 40-card
  sample, base -> round 15: render -68%, unedited save -89%, open -62%, AC parse -55%, render memory -43% (median),
  error vs Photoshop 2023 -52%. Actions: guards' pushed-file fetch -86%, site build's data wait -85%, bakes -66%.
  Bakes: star/icon edits that bake 13 -> 35 of 41; folder cards whose star edit can't redraw 86 -> 11. Card data:
  renamed cards found by old name 23 -> 70 of 70; TTS deck buttons spawning the wrong pile 25 -> 0 of 30. Website:
  contrast failures 11,591 -> 94; a pasted ability in /query_ability 199 ms -> 0.3 ms.
- **The fork's compiled code is not bit-identical across CPUs.** On x86_64 a NaN's and a zero's sign differ from arm64,
  and on AVX-512 numpy's own float32 maths differ in the last bit; one numpy path also raised IndexError on x86 (a NaN
  index). The Mac and the jetson are both arm64, so only the GitHub runner showed it. Fixed and proven on the runner
  with a one-off workflow on a scratch branch (deleted).
- **Resuming a multi-lane workflow after editing its caps re-ran a lane's old rounds** (research-1109, cardsdata r6);
  stopped within minutes, no damage. Stop multi-lane workflows and ship with a fresh small workflow (memory
  `workflow-resume-multilane`).
- **zsh does not split unquoted variables**: a `git bundle create $B $LIST` failed and the deletions after it ran; the
  rejected round tips were bundled again from their commit ids. Write multi-step shell as `set -e` bash scripts.
- **A full-history clone reaches more history**: `card_guards.earlier_fonts` walked font versions the shallow clone never
  had and fell back to GitHub (blocked in tests); the 7 old font blobs were fetched in one batch.

## Owner

- 11-07 and 11-08 rulings: memory `owner-decisions-2026-11-07`. Highlights:
  - Flags replace the CSV and are automatic; the templates were renamed.
  - Linking and normal text in one rewrite, Maximize Compatibility off, off-card pixels kept.
  - Status words follow the chart; fills, adjustments and gradients stay embedded (Springtrap's star excepted).
  - Single-line everywhere; custom stat caps refused at edit time; split titles supported.
  - The font build need not be identical (incremental); no TTS JPEGs.
- 11-09 rulings: memory `owner-decisions-2026-11-09`.
  - Clean-up kinds, folders (Overlay Text for 10 cards), cards only.
  - History relevance: the 5% cut stays, don't re-propose.
  - Status cards in Krabby Patty; the proposal forum spec.
  - PSDs over 50 MB are left alone.
- Wrap-up (11-09 ~19:45):
  - Research workflows stop at a round boundary; the fork release happens in the wrap-up.
  - Next session: research rounds at once, then card timelines, then open bugs.
  - Keep the forum test post.
- 11-09 ~22:20: Despair and Withering titles get a white stroke, on the card only (done).
- 11-09 ~22:30: second pictures tagged, PNG names kept (replaces the 11-08 "stay flagged"); ~22:40: forum, no forward.
- 11-09 ~23:20: lower the box load while the owner plays (orphans stopped, rounds dropped, our work reniced); 11-10:
  speed up the second-pictures checks; replace the clone now.
- Standing feedback:
  - Memories `maintenance-pushes-silent`, `photoshop-no-interrupt`, `never-pkill-shell-snapshot`.
  - Memory `resource-limits`: 11-09 "the box is yours"; a third workflow is fine.

## Left

All in `docs/NEXT_SESSION_PLAN.md` ("Waiting from this session"): the actions r21-25 ship (Pages version commits
unproven, one-try data refetch), the research rounds' card fixes (Clot Gold's spare HP layers, five typos), psd round
15's strikes sweep, the card data ship's two optional dispatches, and the first card workflow run on the new fork pin
(it builds the fork on the runner; revert AC cff351a4's pin line if it fails). Then: research rounds again, card
timelines from Discord, open bugs.
