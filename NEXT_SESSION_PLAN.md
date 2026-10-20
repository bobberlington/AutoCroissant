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

## Orchestration (ultracode, Fable 5.1 as orchestrator)

The main session (Fable 5.1) plans, decides, reviews diffs and does every production action itself (pushes,
jetson restarts, pickle commits); agents never touch production in parallel. Match agent, model and effort to the job
(`agent({agentType, model, effort, isolation})` in workflows):

| Job | Agent | Model / effort |
|---|---|---|
| Read-only sweeps (code maps, TTS save structure, PSD feature surveys) | `Explore` | sonnet, low-medium |
| Designs with real trade-offs (text engine, effect compositor, TTS updater) | `Plan` judge panel: 3 independent designs, then judges | fable, high-xhigh |
| Hard implementation (text engine, layout/line breaking, compositor math) | general-purpose | fable, xhigh |
| Well-specified implementation (one effect or adjustment, emoji sync, sha links, TTS script pieces, website tab) | general-purpose, `isolation: 'worktree'` when several edit one repo | opus, high |
| Adversarial verify (refute each fix: harness scores, tests, a Photoshop/Photopea reference) | general-purpose, 2-3 skeptics per change | opus, high (fable for text) |
| Mechanical runs (score ~813 cards, regressions, builds) | plain scripts first; an agent only to read results | haiku/sonnet, low |
| Docs, memories, plan upkeep | general-purpose | sonnet, medium |

Rules: one agent owns a file at a time; worktrees only for parallel edits to the same repo; review gates (the TTS
audit report, anything outward-facing) stop the workflow for the user; log anything a workflow skipped or sampled.
The psd-tools fork works through feature branches and upstream PRs (normal for a fork; TTSCardMaker and
AutoCroissant stay main-only).

## Decided (2026-10-20)

| # | Decision |
|---|---|
| D1 | TTSCardMaker pushed. |
| D2 | The TTS save lives in TTSCardMaker, kept up to date by the bot. The website can view and edit it (move cards between decks and other actions TTS allows). |
| D3 | "Latest commit per PNG" comes from the history clone on the jetson. |
| D4 | No Aspose. Photopea works, but the goal is a psd-tools fork good enough to replace it on the jetson with no browser (unlimited time/tokens). Edits run on GitHub Actions if possible, else on the jetson. |
| D5 | Cards in the save that no longer exist in the repo: report only for now. |
| D6 | AI is on `archive/ai-generation` (done); `/frankenstein` stays on main. |
| D7 | The jetson moves to Python 3.14 for everything, then 3.10 is removed (Task 0). Build it from source with the 3.10 recipe; python-build-standalone is the fallback. |

## Task 0. Jetson: Python 3.14 for everything, then remove 3.10 (parallel track)

Why: the psd-tools fork (and psd-tools >= 1.20) needs Python >= 3.11; the jetson runs 3.10.19 with psd-tools 1.18.
Inventory (2026-10-20): `autocroissant.service` and `ojwikibot.service` run `/usr/local/bin/python3.10`; no cron
jobs; `~/Documents/homelab` to check; ~90 packages in 3.10's site and `~/.local/lib/python3.10`. 3.10 was built
from source: `--enable-optimizations --with-lto=full --enable-shared --with-openssl=/usr/local/openssl-3
--with-openssl-rpath=auto`, `CFLAGS=-O3 -march=native -fomit-frame-pointer -flto`, OpenSSL 3.6.0 in
`/usr/local/openssl-3`. 4 cores, 6 GB RAM.
1. Finish the inventory: both units' ExecStart, OJWikibot's and homelab's requirements, shebangs, anything else
   importing 3.10 paths.
2. Build 3.14 (latest patch) with the user's recipe (the same one 3.10 used; `ax_cv_c_float_words_bigendian=no` is
   required because `-flto` breaks configure's float byte-order check):
   ```
   ax_cv_c_float_words_bigendian=no ./configure \
     --enable-optimizations \
     --with-lto=full \
     --enable-shared \
     --with-openssl=/usr/local/openssl-3 \
     --with-openssl-rpath=auto \
     CFLAGS="-O3 -march=native -fomit-frame-pointer -flto" \
     CPPFLAGS="-I/usr/local/openssl-3/include" \
     LDFLAGS="-Wl,-O1 -Wl,--as-needed -flto -Wl,-rpath=/usr/local/openssl-3/lib" \
     PKG_CONFIG_PATH=/usr/local/openssl-3/lib/pkgconfig
   ```
   then `make -j4` under nohup (PGO+LTO: expect 1.5-3 h) and `sudo make altinstall` so 3.10 stays until the end.
   Fallback if the build fails: a python-build-standalone 3.14 (`uv python install 3.14`).
3. Per bot, a venv on 3.14 with its requirements. Watch: `distutils` is gone (AutoCroissant's setup.py imports
   `distutils.command.clean`; use setuptools'), `audioop` is gone (discord.py voice needs `audioop-lts`), aarch64
   wheels for numpy/pandas/opencv/davey/aggdraw (build from source if missing). Rebuild AutoCroissant's Cython
   modules with 3.14.
4. AutoCroissant moves from psd-tools 1.18 to 1.25+ (later the fork): run the full parser regression on all cards
   first (1.18 vs 1.25 output) and review differences before deploying.
5. Point each unit's ExecStart at its venv, restart one bot at a time, smoke-test (/query, /update_stats, music,
   OJWikibot's commands).
6. After a stable week: remove 3.10 (`/usr/local/bin/python3.10*`, `pip3.10`, `/usr/local/lib/python3.10`,
   `libpython3.10*`, `~/.local/lib/python3.10`); keep `/usr/local/openssl-3`. Update the jetson memory and the
   build-and-env / run-and-operate skills.

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
only those emojis, including their crossed-out `_no` variants; delete emojis whose icon is gone; download icons by
blob sha (raw `main` can lag). Then remove the reminder: `/remove_reminder reminder_id:f590a57e`.

## Task 2. TTS save auto-update (bot)

**Context:** new cards are added by hand to several decks in the save, and their Name typed on the card. Image
links point at `main`, so with mod caching on players keep the first image they loaded.

**Preliminary audit of `~/Downloads/TS_Save_15.json` (TTS v14.2.2, 5.5 MB):** 228 top-level objects, 57 decks,
2,388 card objects, 821 distinct images, all 9,109 links on `main` (two odd `refs/...` links; 153 more non-card
images: markers, tiles, tokens). Decks by kind × stars,
mostly 2-3 copies; N.M.E. ×2, Status Effects ×3, Minions ×3, Auxiliary Items ×3, six unnamed partial creature
decks, bags "Specific Item Decks", "Rulebooks" (loose rulebook cards, not a deck), "Creature Types". One
CustomDeck (1×1) per card, shared by its copies (773 of 799 deck ids are used by more than one object); backs
`MDW/cardback.png`, `cardback_item.png`, `cardback_statuseffect.png`, plus rulebook-page and stat-marker backs.
Descriptions hold role tags: discard them. Flagged so far: Minions decks contain one 2★ creature
(`Creatures/Other/2 Stars`) and `Auxiliary/Creatures`; Auxiliary Items decks contain regular items; the
**Militron (No Armor)** card is not in the save at all, and the global Lua `resizeTargets` (large cards, keyed by
Nickname) misses it; a loose "Militron" card sits on the table.

1. **Move the save into TTSCardMaker** (e.g. `TTS/Impossibility Simulator.json`), committed once by hand.
2. **Audit report** (user reviews before step 4): per deck GUID: name, copies, repo folders feeding it, counts;
   repo cards missing from a copy; save cards with no repo PNG (report only, D5); cards in an unexpected deck;
   Nickname vs website name; wrong backs; duplicates; the large-card list vs `Auxiliary/Creatures`.
3. **Rules file** `TTS/decks.json`: deck GUID -> rule (kind + stars or folder globs) and its copies; also the
   Rulebooks bag and the marker/tile objects.
4. **Updater** `commands/tts_save.py` (`python -m commands.tts_save <save.json>`; writes the updated save and a
   change report next to it, no flags):
   - pin every repo image link in the save (card FaceURL/BackURL, and the markers, tiles, tokens and backs) to the
     latest commit sha of its file, only when the file's blob changed (blob at the URL's sha vs now); the first run
     pins all;
   - add a new card to every copy of its deck: the card object (CustomDeck entry, `CardID = id*100`, unique GUID,
     back by kind, Nickname = website name, empty Description, Transform like its siblings) and the deck's
     `DeckIDs`, `CustomDeck` map and `ContainedObjects`; rename/move updates path and Nickname;
   - regenerate the Lua `resizeTargets` from `Auxiliary/Creatures`, keyed by the same Nicknames the updater writes;
     clear role tags;
   - backup, atomic write; the sha in each URL is the only state.
   - History source: the jetson's clone (`~/Documents/TTSCardMaker-history.git`); on another machine the script
     makes its own blobless clone next to the bot (path in global_config), else falls back to the GitHub API.
5. **Bot:** `/update_tts_save` (admin) and an automatic run when /update_stats sees PNG changes. The bot commits the
   save to TTSCardMaker through the GitHub API (its token has write access; same retry-on-conflict as the website
   editor) and posts the report with a download link.
6. **Getting it to players:** they download the save from TTSCardMaker (raw link posted by the bot, and a link on
   the website) into `Documents/My Games/Tabletop Simulator/Saves/`; a Workshop upload stays manual. Decide with the
   user whether the bot also attaches the file in Discord.
7. Done when: the first run's report shows only expected changes; a player with mod caching on sees an updated card
   refresh; a second run changes nothing.

## Task 2b. View and edit the TTS save on the website

A "Table" tab on the card browser: load the save from TTSCardMaker (on demand; 5.5 MB), show decks and bags with
their cards, and let editors (same sign-in and batch as card edits) move or copy cards between decks, remove or
add cards, rename decks and reorder. Publishing applies the operations to a fresh copy of the save and commits it,
so it merges with the bot's updates. Start after Task 2's audit and rules exist.

## Task 3. psd-tools fork: PSD edits on the jetson, no Photoshop, no browser (MAIN EFFORT)

The user's call (2026-10-20): spend as many tokens and as much time as needed to make the psd-tools fork good enough
to replace Photopea. Fork: https://github.com/MichaelJSr/psd-tools, cloned at `~/Documents/Tools/psd-tools` (main just past
v1.25.0, `upstream` remote; read its `CLAUDE.md` first). Needs Python >= 3.11: on the jetson, after Task 0. Research, harness and data: `~/Documents/psd-research/`
(`REPORT.md`, whose Photopea-first verdict is superseded by D4; `fork/patches.py` holds the gradient fix).

**Where things stand:** unpatched psd-tools 1.25 renders card structure right but stat bars flat (the
gradient-length bug) and is 0.2-45% off the official PNGs (median 1.68% on 50 cards; Jetpack 45%, Regenerative
Shield 43%: vibrance, gradient map, drop shadow). Photopea is 0.2-2.2% off and handles text, so it is the **reference renderer** while
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
7. Pin the bot and the pipeline to a fork tag only while a needed change isn't released upstream (the website
   doesn't use psd-tools).

Photopea can't be forked (closed source); its paid self-hosted edition ($500-2,000/month) isn't needed.

## Task 3c. Effect layers on pushes (after Task 3)

Port `MDW/Scripts/Effect Layers - All Cards.jsx` (Photoshop-only ExtendScript) to Python on the fork: find status
words with the text engine's character positions, cut them into `Ability FX: <words>` pixel layers, copy the layer
styles from the status chart PSD (the authority), render them, re-export the PNG; idempotent like "Update all".
Run it from the push pipeline for PSDs whose blob changed, only after a PSD has been quiet ~30 min (PSDs can't be
merged; don't race their author), skip its own commits, honour a skip list in `MDW`, and post what changed in
Discord. First prove it on all cards against the JSX batch's output.

## Task 4. Deep optimization and refinement pass

BUGS_AND_IMPROVEMENTS.md §2.5 (network, speed, robustness, code health, concurrency, tests, docs).

## Order

**Task 3 first and foremost** (the user wants the next session spent on it), with **Task 0 in parallel** (the long
Python build runs in the background); Task 3c follows Task 3. Quick wins if useful in between: Task 1, Task 1b. Then Task 2 steps 1-3 (audit report to the user) -> Task 2 steps 4-6 -> Task 2b -> Task 4.
