# Website small fixes (#15)

**Status:** picked by the owner 2026-10-24 ~20:00 (four fixes); not built. Queue item 4. Source: "Website fixes" in
`~/Documents/Tools/wt/scratch/ideas-sweep-report.md` (chart renames and a phone page picker were not picked). All code
is TTSCardMaker `site/` (paths below from that repo's root); straight to its main, one commit per fix.

**Goal.** Four things players hit that are broken or slow today, fixed without new features.

## 1. Tooltips on phones

Today a tap fires `pointerover` then `pointerout` (`site/assets/app.js` ~1277, `site/assets/tip.js` `tipController`),
so a badge's meaning shows and hides at once.
- For `pointerType === "touch"`: show on tap (`pointerup` on a `TIP_SEL` element), ignore that pointer's `pointerout`,
  hide on the next tap outside the shown element or on scroll. Mouse and pen keep today's hover rules.
- A tapped card link still opens the card (the tip shows only for elements without a navigation, or on long press for
  `a.card-ref`).

## 2. Exact name listed first

Searching "Enfeebled" lists that card 6th. In the grid's order (`app.js` `listOf` -> `sortCards`), when the query
(folded by `site/assets/text.js` `norm`) equals a card's name or alias, that card goes first whatever
the sort; Enter in the search box with exactly one such card opens it.

## 3. Deck export with pinned links

`site/assets/deck.js` `ttsSave` builds `FaceURL` from `raw` = `.../main/` (from `build_site.py` `RAW`), so TTS
downloads every card again although the save holds the same image at a pinned link. The TTS save pins each link to
the commit that last changed the file (`commands/tts_save.py` `_pin_links` in AutoCroissant).
- `site/build/build_site.py` writes per card `pin`: that same commit (full SHA, as the save has it), from the history
  it already reads for Image versions.
- `deck.js`: `FaceURL = https://raw.githubusercontent.com/<repo>/<card.pin || "main">/<path>`; card backs keep the
  save's own pinned back links.

## 4. Rulebook pages get links

Today the rulebook's contents list uses `href="#/rules" data-scroll="page-<id>"` (`app.js` `renderRules` ~720), so a
page can't be linked or shared.
- Route `#/rules/<id>` scrolls to `page-<id>` after render; the contents links and each page heading's anchor use it.
- `#/rules` alone stays the top. Old links keep working.

## Verification (lean)

A look on a phone (Safari) for 1 and 4; one exported deck imported in TTS loads from cache (3). Tests: at most one
assert added to an existing site smoke test (the cut keeps about 15 in AutoCroissant), no new files.
