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
| D4 | No Aspose. Photopea works, but the goal is a psd-tools fork good enough to replace it on the jetson with no browser (unlimited time/tokens). Edits run on GitHub Actions if possible, else on the jetson. |
| D5 | Cards in the save that no longer exist in the repo: report only for now. |
| D6 | AI is on `archive/ai-generation` (done); `/frankenstein` stays on main. |

## Task 1. Commit-sha image links (small)

`/query`, ability-search results and embeds post `https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/<sha>/<path>`
with `<sha>` = the latest commit where that PNG's blob changed (Puzzled Cat -> `24742d07…`).
1. `commands/repo_history.py`: `git fetch` the history clone; walk new commits only (`rev-list --parents
   <last>..main`, `ls-tree -r`); a commit changed a path when its blob differs from every parent's; keep
   `{path: (blob, sha)}` and the last head in a small JSON next to the pickles.
2. Refresh on /update_stats (the tree ETag says when) and at startup; `get_card_url` uses it, falls back to `main`.
3. Done when: replies use sha links, no extra requests per /query, survives a restart.

## Task 1b. Emoji sync inside /update_stats (small)

The startup sync only uploads missing emojis, and only at restart; the weekly reminder `f590a57e` (`/sync_emojis`,
Sun 3 AM) covers new icons between restarts, and neither refreshes an edited icon. Instead: during /update_stats,
compare `Types/` and `Types/Stars/` blob shas (already in the tree) with a small stored map and upload or replace
only those emojis. Then remove the reminder: `/remove_reminder reminder_id:f590a57e`.

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

## Task 3. psd-tools fork: PSD edits on the jetson, no Photoshop, no browser (MAIN EFFORT)

The user's call (2026-10-20): spend as many tokens and as much time as needed to make the psd-tools fork good enough
to replace Photopea. Fork: https://github.com/MichaelJSr/psd-tools, cloned at `~/Desktop/psd-tools` (main =
v1.25.0, `upstream` remote; read its `CLAUDE.md` first). Research, harness and data: `~/Documents/psd-research/`
(`REPORT.md`; `fork/patches.py` holds the gradient fix).

**Where things stand:** psd-tools renders card structure and stat bars right, but is 0.2-16% off the official PNGs
(median 1.68% on 50 cards). Photopea is 0.2-2.2% off and handles text, so it is the **reference renderer** while
we build (and the emergency fallback). Upstream: drop shadow PR #679 is open (updated 2026-10-14); outer glow #680
was closed; no text rendering work exists anywhere open source.

**What "good enough" means:** edits only replace the pixels they change (delta merge onto the official PNG), so the
bar is set per region: stat-bar cells and text boxes must match Photoshop. Full-card targets for previews: median
< 0.5% off, no card > 2%.

1. **Harness first:** score every card (~813) psd-tools vs the official PNG, per layer kind and effect; plus a text
   harness: render each text layer and compare with its own stored raster in the PSD (exact ground truth for ~3,500
   text layers). Run it after every change.
2. **Rendering fixes, by impact:** the gradient-length bug (Sonic 8.8 -> 3.3%); drop shadow (build on #679);
   outer glow; vibrance, gradient map, channel mixer, colour balance; inner shadow/glow, bevel, satin; perceptual
   gradients. Each with tests, each also a PR upstream.
3. **Text engine:** parse EngineData (runs, paragraph styles, point vs box text, transforms); fonts from
   TTSCardMaker `MDW/Fonts` (87 used; report missing ones); uharfbuzz shaping + freetype-py rasterizing; leading,
   tracking, kerning, alignment/justification, line breaking close to Adobe's composer, faux styles, caps,
   underline, baseline shift; text effects (stroke, shadow, glow). Calibrate on the text harness.
4. **Edit pipeline (Python only):** for each card in `Card Edits.json`: apply the edit to the PSD (layer toggles per
   stat row/creature; text and name via EngineData keeping runs; new status words coloured from the status chart)
   -> render the changed layers -> delta-merge onto the official PNG -> gate (re-parse gives the edited values,
   changed region as expected, no missing font) -> one commit: PSD + PNG + the baked fields removed from
   `Card Edits.json`. The website shows "PSD pending" until it lands.
5. **Runner:** a GitHub Actions workflow on pushes that change `MDW/Card Edits.json` (pure Python, fork pinned). If
   Actions can't do it (time, size or download limits), the jetson handles the card browser's edit requests: the bot
   notices the Card Edits.json change (its blob sha in the tree) and runs the pipeline.
6. **Done when:** the harness meets the targets, and 10 real edits (stats, a name, ability text with coloured
   status words) look identical to Photoshop when opened on the Mac. Then Photopea is dropped.
7. Pin the bot, website and pipeline to a fork tag only while a needed change isn't released upstream.

Photopea can't be forked (closed source); its paid self-hosted edition ($500-2,000/month) isn't needed.

## Task 4. Deep optimization and refinement pass

BUGS_AND_IMPROVEMENTS.md §2.5 (network, speed, robustness, code health, concurrency, tests, docs).

## Order

**Task 3 first and foremost** (the user wants the next session spent on it). Quick wins if useful in between:
Task 1, Task 1b. Then Task 2 steps 1-3 (audit report to the user) -> Task 2 steps 4-6 -> Task 2b -> Task 4.
