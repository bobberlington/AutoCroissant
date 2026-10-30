# Session 2026-10-30 afternoon: overlay badges, TTS names and panels, History revert, rulebook to scale

AutoCroissant ..d9f15d5, TTSCardMaker ..9ecd43b1 (then Card sync's save commits 05add25f, 30a050f7). No deploy owed:
the bot commits touch only the guards (card_standards, which the bot doesn't import), tests and skills.

## Owner asks, in order

- **Butterfly's badges printed as blobs** under its Gradient Overlay. Its ability is in `Section 1` / `Section 2`
  (a hidden `Ability` holds the joined text); hdckighfkvhvgmk's Photoshop re-save of 10-25 had no Stylistic Alternates
  flag, and the 10-25 badge FX retirement only re-baked `Ability` layers. New guard pixel fix `overlay badges`
  (`card_standards.one_colour_badges`, `salt_layers`): any visible text layer under a Color/Gradient/Pattern overlay
  whose badges or `:sticker`s in stoogedings lack the flag gets it (the Txt2 story, as the bake writes it), those glyphs
  spliced, the preview patched, gated; the guards re-export the PNG silently. Card-only (`CARD_PIXELS`). The kind name
  has no hyphen: `card_sweep.kinds_of` reads "-" as a space. A sweep of all 764 cards found Butterfly alone (0 refused);
  landed as TTSCardMaker 41433b7f with `Card-sweep: overlay badges`. The bake already set the flag on part layers
  (they go through `set_text_layer(which="ability")`), and the site's mock-up follows a part's own overlay.
- **Croissant2's three "Rulebook: Death" edits** (picture moves) reverted: 2e4c9dd1.
- **The Silithid button** was missing on The Gates of Ahn'Qiraj ("summon a silithid", lower case): `tts_details` matched
  card names case-sensitively. `build_site.name_links` ports text.js `makeRenderer`'s matching and `isRef` (a lowercase
  one-word name only after its verb: summon/resummon/revive for minions, inflict/cleanse/stacks of for statuses); equal
  to text.js on all 772 cards (node check); 22 cards' `m` changed (gains like Silithid and Polevaulter Zombie, false
  hits like "Cat:" and "Flip" gone). A duplicate's link spawns its original. Test: `test_tts_details_for_the_save`.
- **TTS panels built after load**, not on first use: `prepare()` runs 2 s after tts.json arrives (decode + the seated
  players' panels, one `UI.setXmlTable`); `seatPanels(color)` builds a later sitter's 2 s after `onPlayerChangeColor`;
  `ensurePanels` takes a list. lua_mock's checks rewritten for it (Teal seated later).
- **History "Revert to this version"** (agent, bffadeb4): rulebook chapters (editors), the ban list (admins), cards (the
  History block and Image versions). One commit through `commitFiles`; cards' tree entries point at the old blob SHAs
  (`{sha}`, nothing uploaded): the PSD and the PNGs printed from it at today's paths (renames followed by Image
  versions); Card Edits.json untouched. Texts with no known commit (7 of 1,070) and picture-only duplicates' texts get
  no button. Tested against mocks only: no real GitHub write yet.
- **Rulebook zoom** (agent, 9ecd43b1): chapters are drawn to scale. `.rb-book` is a size container, chapter text 17px on
  a 1108px page (cqw), every px in `--rb-px`, max 1300px, 9px floor (editor preview: none), phones flow at 17px;
  pictures without `width=` take `--w` (natural px). Live Death measured identical at 100/150/200% zoom. Trade-offs:
  zooming in doesn't enlarge a page that fills its window until about 175% on a 1400px window (the contents column
  grows), and the editor preview's text is about 10px.

## Checks

- AutoCroissant full suite 660 passed (before the push); site tests after the merges; esbuild bundle and CSS minify ok.
- TTS checks: lua_mock and updater_idempotent failed only while the save still held the old Global.lua; 24 of 24 on
  the rebuilt save (30a050f7) at the wrap-up.
- Tabletop Simulator's own Saves folder has no copy of the save (its list still names one copied in at 04:45; the file
  is gone). The owner was asked whether to copy the rebuilt save there; no answer yet.
