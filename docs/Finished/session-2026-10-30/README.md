# Session 2026-10-29/30: the rulebook on the website, the TTS save's panels

AutoCroissant ..d86fe30, TTSCardMaker ..3b32375e (plus Card sync's save commits), data f417aaf9. The jetson runs
17601aa: later bot commits touch only Card sync's modules (card_data, rulebook_pages, tts_save), tests and skills, so
no deploy was needed. Memory `rulebook-on-website` holds the details and rulings; this is the summary.

## The rulebook (TTSCardMaker site/rulebook, 24 chapters)

- Written on the website: chapters `site/rulebook/NN-slug.md` (wording as printed plus the owner's rulings of 10-30),
  pictures in `img/`, fonts in `fonts/`. `rulebook.js` draws them (syntax in its header), `rulebook-edit.js` edits them;
  History buttons open a git-style diff (GitHub API, anonymous). Chapter 23 Timing Tags is the badges' meanings site-wide.
- The bot reads `db/rulebook.json` (card_data writes it); `/query_rulebook` and `/export_rulebook` use chapters.
- The Photoshop pages went to `MDW/Old Rules/Rulebook` (the older ones in `Very Old Rulebook/`); card_data skips PSDs in
  any `Rulebook` folder and drops their versions, details and credits' warnings (the move had briefly counted 37 as
  archived cards).
- TTS gets a searchable PDF (`TTS/Rulebook/Rulebook.pdf`, printed by `commands.rulebook_pages` from the website's
  `#/rules/print`): reprinted only when a page's pixels change; `pages.json` keeps each page's hash and text. Contents
  page numbers; list numbers get room for two digits.

## The editor (rulebook-edit.js)

Undo/redo (own stack; buttons, Ctrl/Cmd+Z, Shift+Z, Y, the Edit menu); the page is morphed, not redrawn (about 3 ms);
drags once a frame with a drop label; a fixed 52 px tools bar above the page (never over a picture): the 3×3 anchor grid
(a row's picture in its column; a placed one snapped in its panel; one in the text: top/middle/end of its panel by row,
left/own line/right by column), size, delete, Make space; arrow keys (Make space: through the text; a row's picture
across columns and up/down its column); snapping with guides for placed pictures; a picked section, panel, box or row
takes the drops landing in it; blank space keeps a picture where dropped (Make space: between the sections, which make
room); Insert, Colour and Pictures menus (search, Add as Picture/Mascot says); panel pictures added as blocks; decimal
widths (the real cause of "pictures resize when dragged"); floats at most half wide; rows never squeezed beside floats.

## The TTS save

- Assets: everything the table uses is in `MDW/Assets` (card backs, Book_Cover, felts, holders, markers, and the once
  outside-hosted book model, table model and asset bundles, Maxwell); the save links nothing outside the repo. drift.py
  allows moved-file links; its baseline is 0ee01372 (the markers' scripts removed).
- Global script (`TTS/Global.lua`, written into the save by tts_save; never edit the save's script):
  - Right-click a card (in hand too) > Card details: up to three draggable panels per player, A-/A+ sizes, Rules
    buttons by chapter (types, tags, status effects, damage types, negation, reactions), "+ Name" spawns the cards its
    text names into the player's hand face down at their played size.
  - Right-click the table > Find a card (name, kind, stars, type, tag); right-click the rulebook or its bag > Contents
    and search (Book.setPage).
  - Panels are built on a player's first use and the data decoded then (1,100 elements at load stalled the game); no
    Lua patterns (MoonSharp's "pattern too complex").
  - Turn markers: one onObjectRotate replaces six per-frame coroutines (player flips only).
  - Hand zones: a side's zones follow its seated players (setValue): own seat if taken, else the first seated teammate.
    The 10-22 "simpler Lua" had dropped the old zone claiming assuming TTS teams share hands; they share only
    visibility.
- Data for the panels: the website's `data/tts.json` (build_site `tts_details`).

## Process

The website and TTS changes were checked in headless Chrome (scratch harness `~/Documents/Tools/wt/scratch/rbe-test`,
run with `rulebook/cdpeval.mjs`) and a fengari Lua harness (`scratch/tts/luacheck`); nothing ran inside TTS itself.
Card sync's 30-minute window was skipped four times at the owner's request (cancel the waiting run, dispatch with
`bake=pending`).

## After the wrap-up (owner: fix the TTS check suite and the stale docs)

- `TTS/checks` 24 of 24 again (was 18): `lua_mock.py` rewritten for today's Global.lua (on the save's own objects and
  zones; 124 checks; it caught all ten deliberate breakages tried), `scripts_global_refs.py` reads `TEAMS` as seat = team
  and checks `SIDE_ORDER`, the link checks leave the global script's live-data links unpinned (and fetch `.obj` files
  uncompressed), the updater checks copy the whole bot and its new inputs.
- A real updater bug it found: tts_save pinned the global script's `main/…/pages.json` link and the next step wrote
  Global.lua back, so every run reported "1 link pinned" and "Global script written" for an unchanged save; the global
  script and UI are now left out of the pinning.
- Stale docs: the checks README, the website README's editor paragraph and `tts.json`, `build_site.py`'s comment, the TC
  README (the in-game right-click menus), `TTS/decks.json`'s unread 10-20 `objects` inventory dropped, bugs doc D1
  re-worded and D4 retired.
