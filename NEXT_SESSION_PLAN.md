# Next session plan (2026-10-20)

Start in `~/Desktop/AutoCroissant`. Read memories `jetson-server` and `ttscardmaker-integration` first.

**State at the end of 2026-10-20**
- Bot on the jetson runs main `05c643e` (AI generation moved to branch `archive/ai-generation`); pickles `7d0e29f`;
  50 slash commands; jetson and GitHub in sync.
- TTSCardMaker main is pushed: website + editor live at https://michaeljsr.github.io/TTSCardMaker/.
- History-only clone on the jetson: `~/Documents/TTSCardMaker-history.git` (~1 MB). Git there is 2.25: never run a
  diff in it (`log --name-only`/`--raw` download blobs); walk `rev-list --parents` + `ls-tree -r` (~10 s, verified
  against GitHub on 4 cards).
- PSD research (report, Photopea harness, psd-tools patch, survey data): `~/Documents/psd-research/`. TTS survey
  scripts are in `/private/tmp/fontexplore/scripts/tts_survey*.py` (gone after a reboot; easy to redo).

## Decided (2026-10-20)

| # | Decision |
|---|---|
| D1 | TTSCardMaker pushed. |
| D2 | The TTS save lives in TTSCardMaker, kept up to date by the bot. The website can view and edit it (move cards between decks and other actions TTS allows). |
| D3 | "Latest commit per PNG" comes from the history clone on the jetson. |
| D4 | No Aspose. Photopea first; if it fails, build a psd-tools fork, with as much time and money as it takes to make it fully functional. |
| D5 | Cards in the save that no longer exist in the repo: report only for now. |
| D6 | AI is on `archive/ai-generation` (done); `/frankenstein` stays on main. |

## Task 1. Commit-sha image links (small; do first)

`/query`, ability-search results and embeds post `https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/<sha>/<path>`
with `<sha>` = the latest commit where that PNG's blob changed (Puzzled Cat -> `24742d07…`).
1. `commands/repo_history.py`: `git fetch` the history clone; walk new commits only (`rev-list --parents
   <last>..main`, `ls-tree -r`); a commit changed a path when its blob differs from every parent's; keep
   `{path: (blob, sha)}` and the last head in a small JSON next to the pickles.
2. Refresh on /update_stats (the tree ETag says when) and at startup; `get_card_url` uses it, falls back to `main`.
3. Done when: replies use sha links, no extra requests per /query, survives a restart.

## Task 2. TTS save auto-update (bot)

**Context:** new cards are added by hand to several decks in the save, and their Name typed on the card. Image
links point at `main`, so with mod caching on players keep the first image they loaded.

**Preliminary audit of `~/Downloads/TS_Save_15.json` (TTS v14.2.2, 5.5 MB):** 228 top-level objects, 57 decks,
2,388 card objects, 821 distinct images, all 9,109 links on `main` (one odd `refs/...`). Decks by kind × stars,
mostly 2-3 copies; N.M.E. ×2, Status Effects ×3, Minions ×3, Auxiliary Items ×3, six unnamed partial creature
decks, bags "Specific Item Decks", "Rulebooks", "Creature Types". One CustomDeck (1×1) per card; backs
`MDW/cardback.png`, `cardback_item.png`, `cardback_statuseffect.png`. Descriptions hold role tags: discard them.
Flagged so far: Minions decks contain a 2★ creature folder and `Auxiliary/Creatures`; Auxiliary Items decks contain
regular items; the global Lua `resizeTargets` (large cards) misses **Militron (No Armor)**; a loose "Militron" card
sits on the table.

1. **Move the save into TTSCardMaker** (e.g. `TTS/Impossibility Simulator.json`), committed once by hand.
2. **Audit report** (user reviews before step 4): per deck GUID: name, copies, repo folders feeding it, counts;
   repo cards missing from a copy; save cards with no repo PNG (report only, D5); cards in an unexpected deck;
   Nickname vs website name; wrong backs; duplicates; the large-card list vs `Auxiliary/Creatures`.
3. **Rules file** `TTS/decks.json`: deck GUID -> rule (kind + stars or folder globs) and its copies.
4. **Updater** `commands/tts_save.py` (`python -m commands.tts_save <save.json> [--dry-run]`):
   - pin FaceURL/BackURL to the PNG's latest commit sha only when its blob changed (compare the blob at the URL's
     sha with the current one via the history clone); the first run pins all;
   - add new cards to every copy of their deck (new CustomDeck id, `CardID = id*100`, unique GUID, back by kind,
     Nickname = website name, empty Description); rename/move updates path and Nickname;
   - regenerate the Lua `resizeTargets` from `Auxiliary/Creatures`; clear role tags;
   - backup, atomic write, short change report; the sha in each URL is the only state.
5. **Bot:** `/update_tts_save` (admin) and an automatic run when /update_stats sees PNG changes. The bot commits the
   save to TTSCardMaker through the GitHub API (its token has write access; same retry-on-conflict as the website
   editor) and posts the report.
6. Done when: a dry run shows only expected changes; a player with mod caching on sees an updated card refresh; a
   second run changes nothing.

## Task 2b. View and edit the TTS save on the website

A "Table" tab on the card browser: load the save from TTSCardMaker (on demand; 5.5 MB), show decks and bags with
their cards, and let editors (same sign-in and batch as card edits) move or copy cards between decks, remove or
add cards, rename decks and reorder. Publishing applies the operations to a fresh copy of the save and commits it,
so it merges with the bot's updates. Start after Task 2's audit and rules exist.

## Task 3. PSD edits from website edits (Photopea; psd-tools fork as fallback)

Research (full report, harness and data: `~/Documents/psd-research/REPORT.md`):
- **Photopea works.** Headless renders are 0.2-2.2% off the official PNGs (psd-tools 1.25: 0.2-16%), 1 s per card,
  73-85 MB PSDs fine. Stat edits change only the right pixels; PSDs saved by Photopea keep layers, effects, smart
  objects and text runs.
- **Text:** Photopea's own text API wipes style runs. What works: rewrite the text in EngineData with psd-tools
  (runs mapped per character), then open in Photopea and set `textItem.tracking = tracking` to force a re-render.
  Photopea draws text 10-34 px higher than Photoshop: calibrate per layer (text-box error 24-34% -> 6-8%).
- **psd-tools bug:** 90° gradient overlays render flat (`angle % 90`); a one-line fix takes Sonic 8.8 -> 3.3%.
- Runner: GitHub Actions (x64). Fonts must come from a public origin (raw.githubusercontent of `MDW/Fonts`), not
  localhost; a missing font raises a blocking Alert.

Plan:
1. **Pipeline** (`TTSCardMaker/site/psd/` or a bot module; decide where): for each card in `Card Edits.json`:
   psd-tools applies the edit to the PSD (stats/stars/types: toggle layers, per creature row; text/name: rewrite
   EngineData keeping runs; new status words coloured from the status chart) -> Photopea renders the original and
   the edited PSD, re-renders edited text with the calibrated offset, saves the PSD -> **delta merge**: the official
   PNG outside the changed region (dilated diff), Photopea's pixels inside it.
2. **Gate:** reject when the changed region is larger than expected, a font is missing, or re-parsing the new PSD
   doesn't give the edited values. On pass, one commit: PSD + PNG + the baked fields removed from Card Edits.json.
3. **Runner:** a TTSCardMaker workflow on pushes that change `MDW/Card Edits.json` (Playwright + Photopea); the
   website shows "PSD pending" until it lands. Try the jetson later.
4. **First milestone:** stats only (lowest risk), on 3 test cards, verified by opening one PSD in Photoshop.
   Then names, then ability text.
5. **Fallback / previews:** send the gradient fix upstream (needs the user's GitHub account for the PR); if Photopea
   breaks or blocks automation, build the psd-tools fork per the report (effects + adjustments ~2-3 weeks, text
   engine ~4-8 weeks), budget unlimited (D4).

## Task 4. Deep optimization and refinement pass

BUGS_AND_IMPROVEMENTS.md §2.5 (network, speed, robustness, code health, concurrency, tests, docs).

## Order

Task 1 -> Task 2 steps 1-3 (report to the user) -> Task 3 first milestone -> Task 2 steps 4-6 -> Task 2b -> rest
of Task 3 -> Task 4.
