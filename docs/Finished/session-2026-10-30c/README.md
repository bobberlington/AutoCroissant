# Session 2026-10-30 evening to 10-31: rulebook pictures, names and mentions, guards R14, TTS panels

AutoCroissant ..1658cc6, TTSCardMaker ..098bd28a (Card sync's 09a590d5, 07:22 UTC, before it: the TTS save with the
optimized Global.lua), data 679b1309. The Saves folder copy of the save (an unmodified 10-30 afternoon copy) was updated
to 09a590d5 at the wrap-up. No deploy owed: the bot commits touch the guards (card_standards, card_guards), the parser (run by Card sync
in Actions), tests, docs and skills.

## Rulebook (TTSCardMaker site/assets)

- **Make space amid text** (rulebook-edit.js `spots`, `within`, `sideFor`): a picture dropped on a list or a paragraph
  goes between the items or lines it lands on (a list splits, numbers kept as typed); amid text the middle third
  floats to the nearer side (text can't wrap both sides); arrow keys step a block, an item or a line.
- **Text wraps around pictures**: a list right after a floated picture wraps around it (`:is(.rb-fig.rb-left,
  .rb-fig.rb-right) + .rb-list { display: block }`; other lists keep flow-root, so nothing else moved); a left float
  followed by a list keeps room for the markers (margin and shape-margin 2em / 2.8em for ol).
- **See-through pictures**: build_site `picture_outline` traces 37 transparent pictures (alpha past half, 32 bands) to
  polygon points in %; rules.json `images[file][3]`; rulebook.js puts `shape-outside: polygon(...) content-box` on a
  floated one (`ctx.shape`), CSS shape-margin .9em. Chrome sizes `shape-outside: url()` at the image's natural size,
  hence polygons.
- **Pictures hang out of their panel** (owner: as on the Photoshop pages): `.rb-panel` no longer flow-root; `.rb-sec`
  and the new `.rb-solo-wrap` (a panel without a tab) hold the overhang; a panel with a solid floated picture (no
  outline: screenshots, diagrams) holds it (`holds`, `.rb-holds`).
- **`out=N`**: N% of a floated picture hangs out of its panel's side, the panel moves in by as much (`outOf`, `hangOf`).
  Getting Started: Kirby `{.left width=30 out=62 dy=-6}`.
- Examples were shown in `~/Documents/Tools/wt/scratch/rbe-test/examples/` (deleted at the wrap-up).

## Card names and mentions

- **EXACT_CASE** (text.js, owner: temporary, back to false 10-31, NOT yet done): website links and mentions only for
  names typed in their own case. The TTS buttons (`build_site.name_links`) kept any case.
- **Guard R14** (owner: report only, no one-time fix): card names in ability text typed in another case
  (`card_standards.miscased_names`, `rule_name_case`; names from data `db/stats.json` via `card_guards.shown_cards`;
  no data: skipped). 9 cards at the start (Gates of Ahn'Qiraj "silithid", Doctor Zomboss, Fawful "jetpack", Mettaton,
  Nefarious Chip, Overseer, Papa Chip, Sans, Azamoth the Forgotten King). docs/Campaigns/PUSH_GUARDS.md row.
- **Split titles keep printed case** (psd_parser `_printed_title`): fall back to the file name only when the pieces
  hold its words in another order (The Big Yellow Cat). 5 cards re-parsed by a Card sync `cards=` dispatch (both
  Azamoths, Oryx, Gates/Temple of Ahn'Qiraj -> "The gates of AHN'QiRAj").
- **Editor names** (edit.js `namesLine`): "On the site now" + "File:" under Name; "These boxes make: ..." under the title
  boxes (was the misleading "Name on the site").
- **Self-mentions**: a card naming itself is in its own Mentions / Mentioned by, links itself, and gets its own TTS spawn
  button (11 cards); `selfName` removed from text.js; R14 counts a card's own name; Unused still means no OTHER card.
- **Unused** now for every Auxiliary kind (grid.js `AUX_KINDS`, `unusedTip`): buffs, debuffs, minions, auxiliary items,
  auxiliary creatures; duplicates never. 25 cards with any-case matching (8 aux items, 7 minions, 10 statuses).

## Website, other

- Rendered previews: a trash button discards the preview shown (preview.js `discard`).
- A card-image overlay of mention squiggles was discussed and dropped by the owner.

## TTS (TTS/Global.lua)

- Buttons: Rules wrap onto 2 rows (never shortened); spawn buttons on up to 2 rows, spread evenly when they overflow,
  the widest shortened with "..." and the whole name as a `tooltip`.
- Optimized: `set()`/`setValue()` send only changed attributes, one `UI.setAttributes` call per element (cache cleared on
  rebuild; the search boxes' text set directly); the save's cards indexed a few containers a frame after the panels
  are built (`indexSome`, `indexLater`); Find a card searches a presorted list. Calls into the game: open 5 cards 503 ->
  131, again 503 -> 72, A+/A- 212 -> 42, finder + 6 keys 221 -> 90, rulebook + 6 keys 352 -> 100 (lua_mock counts).
- Not verified in the game: `tooltip` on buttons, `UI.setAttributes`.

## Checks

- AutoCroissant full suite 665 passed (parser change); targeted suites after the later changes; esbuild bundle each time.
- lua_mock 0 failures on live data and on a stress tts.json (the six longest names); TTS suite 24 of 24 on the
  rebuilt save (09a590d5) at the wrap-up.
