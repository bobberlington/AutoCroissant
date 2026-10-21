# Next session plan (2026-10-20)

Start in `~/Desktop/AutoCroissant`. Read memories `jetson-server` and `ttscardmaker-integration` first.

**State at the end of 2026-10-20**
- Bot on the jetson runs main `05c643e` (AI generation moved to branch `archive/ai-generation`); pickles `7d0e29f`;
  50 slash commands; jetson and GitHub in sync.
- TTSCardMaker main is pushed: website + editor live at https://michaeljsr.github.io/TTSCardMaker/.
- History-only clone on the jetson: `~/Documents/TTSCardMaker-history.git` (about 27 MB). Git there is 2.25: never run a
  diff in it (`log --name-only`/`--raw` download blobs); walk `rev-list --parents` + `ls-tree -r` (~10 s, verified
  against GitHub on 4 cards).
- PSD research (report, Photopea harness, psd-tools patch, survey data): `~/Documents/Tools/psd-research/`. TTS survey
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
| D2 | The TTS save lives in TTSCardMaker, kept up to date by a TTSCardMaker Action running the bot's updater (`tts-save.yml`; `/update_tts_save` by hand). The website can view and edit it (move cards between decks and other actions TTS allows). |
| D3 | "Latest commit per PNG" comes from the history clone on the jetson. |
| D4 | No Aspose. Photopea works, but the goal is a psd-tools fork good enough to replace it on the jetson with no browser (unlimited time/tokens). Edits run on GitHub Actions if possible, else on the jetson. |
| D5 | Cards in the save that no longer exist in the repo: report only for now. |
| D6 | AI is on `archive/ai-generation` (done); `/frankenstein` stays on main. |
| D7 | The jetson moves to Python 3.14 for everything, then 3.10 is removed (Task 0). Build it from source with the 3.10 recipe; python-build-standalone is the fallback. |

## Task 0. Jetson: Python 3.14 for everything, then remove 3.10 (parallel track)

**Status 2026-10-20:** steps 1-5 done. Python 3.14.8 built with the recipe below in `~/build/python/Python-3.14.8`
(1 h; `make altinstall` + `sudo ldconfig`; missing only _dbm, _tkinter, _zstd). Venvs `~/venvs/autocroissant`
(requirements.txt; psd-tools 1.25.0) and `~/venvs/ojwikibot` (requirements + requirements-dev); both setup.py builds
compile on 3.14; both units' ExecStart point at their venv (`/etc/systemd/system/*.service`; OJWikibot's copies in
`scripts/services/msrouji/` still say python3.10). Parser regression psd-tools 1.18 vs 1.25: 813 cards, 0 differences.
Step 6 done the same day (user's call): 3.10 removed (`/usr/local/bin/python3.10*`, `pip3.10`, `/usr/local/lib/python3.10`,
`libpython3.10*`, headers, pkgconfig, `~/.local/lib/python3.10` and the 24 `~/.local/bin` scripts with 3.10 shebangs);
`/usr/local/bin/python3`, `python3-config` and `pip3` now point at 3.14; `/usr/local/openssl-3` kept. Power-cycled with
`jetson-powercycle cycle` (Shelly plug, 20 s off): ssh back in 75 s, both bots logged in within 90 s of boot, no failed
units, docker (jellyfin, adguardhome) up; the only boot errors are the usual L4T noise (initramfs magic, lp/ppdev
modules, imx219 camera probes, nvphs). The build-and-env and run-and-operate skills were refreshed by the Task 4 docs step (2026-10-20).

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

**Status 2026-10-20:** done (e13d754).

`/query`, ability-search results and embeds post `https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/<sha>/<path>`
with `<sha>` = the latest commit where that PNG's blob changed (Puzzled Cat -> `24742d07…`).
1. `commands/repo_history.py`: `git fetch` the history clone; walk new commits only (`rev-list --parents
   <last>..main`, `ls-tree -r`); a commit changed a path when its blob differs from every parent's; keep
   `{path: (blob, sha)}` and the last head in a small JSON next to the pickles.
2. Refresh on /update_stats (the tree ETag says when) and at startup; `get_card_url` uses it, falls back to `main`.
3. Done when: replies use sha links, no extra requests per /query, survives a restart.

## Task 1b. Emoji sync inside /update_stats (small)

**Status 2026-10-20:** done (e13d754).

The startup sync only uploads missing emojis, and only at restart; the weekly reminder `f590a57e` (`/sync_emojis`,
Sun 3 AM) covers new icons between restarts, and neither refreshes an edited icon. Instead: during /update_stats,
compare `Types/` and `Types/Stars/` blob shas (already in the tree) with a small stored map and upload or replace
only those emojis, including their crossed-out `_no` variants; delete emojis whose icon is gone; download icons by
blob sha (raw `main` can lag). Then remove the reminder: `/remove_reminder reminder_id:f590a57e`.

## Task 2. TTS save auto-update (bot)

**Status 2026-10-20:** steps 1-7 done. Save committed as `TTS/Impossibility Simulator.json`, rules `TTS/decks.json`
(audit: `TTS/AUDIT.md` in TTSCardMaker, written by `TTS/audit.py`), updater `commands/tts_save.py` (`python -m commands.tts_save <save.json>`),
`/update_tts_save` (admin). The automatic path is TTSCardMaker's Action `.github/workflows/tts-save.yml` (pushes of
PNGs, `TTS/decks.json`, `site/Card Edits.json`; daily 11:41 UTC; dispatch), which runs `python -m commands.tts_save`
with the `AUTOCROISSANT_LOCAL_DIR` / `AUTOCROISSANT_HISTORY_CLONE` env overrides read by global_config.py (done
2026-10-20; the bot's run after /update_stats and its symlink step were removed). First production run: commit dceef8eb (867 links pinned, 12 cards added, ZOEY -> Z!!!!!
rename, 56 Nicknames, 575 role tags cleared, 10 CustomDeck keys fixed, resizeTargets regenerated); the second run
reported no changes. 29 website names were set to the printed card names first (Card Edits.json). The bot's
fine-grained token is read-only on the card repo (POST git/blobs -> 403), so commits go through git objects in the
history clone pushed over SSH with the AutoCroissant key; give the token Contents: write to use the API path.
Players download the save from the raw link the bot posts (not attached in Discord: 5.9 MB). Decisions taken for the
audit's open items: complete decks get missing cards at the rule's count; hand-picked and empty decks are left alone;
cards outside a complete deck's rule are moved out unless listed in the rule's include; gone images are reported only.

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

**Status 2026-10-20:** shipped (TTSCardMaker 0d49893) and reverted the same day at the user's request (75692fb;
`site/assets/table.js` and `tts.js` are gone). The tab loaded the save on demand, listed decks and bags with their
cards and rules, gave players the download link, and let editors move, copy, add, remove, reorder and rename;
publishing re-applied the staged operations to the fresh save and committed it byte-for-byte like the bot (a JS port
of the updater's serializer), keeping `TTS/decks.json` in step. Not rescheduled; the Play tab (item 16) must not reuse
the name "Table".

A "Table" tab on the card browser: load the save from TTSCardMaker (on demand; 5.9 MB), show decks and bags with
their cards, and let editors (same sign-in and batch as card edits) move or copy cards between decks, remove or
add cards, rename decks and reorder. Publishing applies the operations to a fresh copy of the save and commits it,
so it merges with the bot's updates. Start after Task 2's audit and rules exist.

## Task 3. psd-tools fork: PSD edits on the jetson, no Photoshop, no browser (MAIN EFFORT)

The user's call (2026-10-20): spend as many tokens and as much time as needed to make the psd-tools fork good enough
to replace Photopea. Fork: https://github.com/MichaelJSr/psd-tools, cloned at `~/Documents/Tools/psd-tools` (main just past
v1.25.0, `upstream` remote; read its `CLAUDE.md` first). Needs Python >= 3.11: on the jetson, after Task 0. Research, harness and data: `~/Documents/Tools/psd-research/`
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
5. **Runner:** a GitHub Actions workflow on pushes that change `site/Card Edits.json` (pure Python, fork pinned). If
   Actions can't do it (time, size or download limits), the jetson handles the card browser's edit requests: the bot
   notices the Card Edits.json change (its blob sha in the tree) and runs the pipeline.
6. **Done when:** the harness meets the targets, and 10 real edits (stats, a name, ability text with coloured
   status words) look identical to Photoshop when opened on the Mac. Then Photopea is dropped.
7. Pin the bot and the pipeline to a fork tag only while a needed change isn't released upstream (the website
   doesn't use psd-tools).

Photopea can't be forked (closed source); its paid self-hosted edition ($500-2,000/month) isn't needed.

## Queued (2026-10-20, user)

- **TTS save by Action — done (2026-10-20):** `tts-save.yml` in TTSCardMaker is the automatic path; the bot's
  post-/update_stats run (`save_affected`, `update_after_stats`) is deleted, `/update_tts_save` stays; global_config
  reads `AUTOCROISSANT_LOCAL_DIR` / `AUTOCROISSANT_HISTORY_CLONE` so the Action needs no symlink (its sparse
  checkout now lists every module the updater imports).
- **Move the website data files out of MDW/ — done (2026-10-20):** `Card Authors.txt`, `Card Edits.json`,
  `Website Admins.txt`, `Constructed Ban List.txt` live in TTSCardMaker `site/` (13ac680 on main, with build_site.py,
  edit.js, deck.js, site.yml / tts-save.yml and site/README). Bot: `AUTHORS_PATHS` / `EDITS_PATHS` in
  global_config.py point at `site/` only (read by repo_traverser, query_card and tts_save); the `MDW/` fallback,
  its tests and the doc notes are dropped.
- **Two kinds of card edits.** The pipeline (Task 3 step 4), `CardEdits` and the editor must distinguish edits that
  change the PSD and the database (name, stats, ability text) from database-only edits (note, aliases, hidden,
  problems_ok, allow_high_stats, dup_of, series, subseries, creatures); only the first kind is baked and removed.
- **Rules skill:** `.claude/skills/impossibility-simulator-rules` (the game itself, from the rulebook pages).
- **Rulebook data issues** found while writing the rules skill (details: `.claude/skills/impossibility-simulator-rules/reference/rulebook.md`,
  end sections): /query_rulebook and the website carry hidden or off-canvas text players never see (duplicate
  paragraphs on 09, a hidden PASSIVES block on 11, an older resolve-order draft on 12 Reactions2 that contradicts
  12 Reactions3, an older ITEMS block on 22); type-injection slips on pages 14, 14.2, 15, 16, 19.2, 20 (icons appended
  after page numbers, `[support]` for the machine icon, swallowed minion icon gaps, lost star glyphs on 08); rulebook
  contradictions (21 NORMAL damage formula vs 07/10; 12.3 Hardened stacks; 07.3 text vs picture; 19 vs 09 on when
  status effects proc); tags used on cards but defined nowhere: (XCD), (XGCD), (XU) and 1GCD/2CD/1CD/2U/2GCD/3U/4U.
  Fixes are card/rulebook content (TTSCardMaker) plus a parser rule to skip hidden text layers on rulebook pages.
  Campaign: `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` (item 19).
- **Subseries** (user, 2026-10-20): a card can have a subseries besides its series (Goombella: series Mario, subseries
  Paper Mario: The Thousand-Year Door). Database-only edit: `subseries` (nullable string) in `Card Edits.json`. Website
  side done (TTSCardMaker 15a16be: filters, search `subseries~`, card page, Stats breakdown, editor field in the
  database-only group). **Bot side done (2026-10-20, after Task 4):** `CardInfo.subseries` (+ `to_dict`),
  `CardEdits.TYPES/NULLABLE/ATTRS`, `prep_dataframes` column, query DSL `subseries==`/`subseries~`, `/query` embed
  "Series · Subseries" (shown for cards with a subseries or website data; /query never showed the series before),
  `/update_metadata` and `/mass_replace` fields, a PSD re-parse keeps it like author and series, `/export_cards`
  column, help text, skills; tests in `tests/test_subseries.py`.

## Ideas 5-23 (2026-10-20, user)

Items 5-15 were pasted by the user on 2026-10-20 (1-4 were not in the paste) and refined and ordered the same day (memory
`website-and-pipeline-ideas`); 16-19 and 21 have campaign docs (21 a stub), 20 is a smaller design. All of it is queued after Task 3 steps
4-7 (the edit pipeline). **Order:** 12, 14, 13 -> 15 -> 6, 7, 8 -> 9, 10, 11 -> 5; then 16-21. Record status here as each ships.

**Decisions (user).** Private decks live in the repo encrypted (WebCrypto AES-GCM, key from a deck passphrase via PBKDF2;
the whole payload, name and description included, is encrypted; salt and owner login in clear; the owner's browser may
remember the passphrase); public decks are plain JSON `site/decks/<login>/<slug>.json`. Profiles exist only for people who
create one: authorship is already listed four ways (the Author dropdown `#/cards?author=Name`, the "by Name" link on a
card page, search `author==`/`author~`, the Stats "Cards per author" chart). Generated cards are fully separate from real
cards and never in statistics (item 18).

- **12. Effects on toggled stat layers.** The layer effects must render on the stat-bar layers the edit pipeline toggles,
  or an edited stat sits flat beside its neighbours; Task 3 step 2's effects work, scored per stat cell by the harness.
- **14. Ability FX layers from the text.** The status-word `Ability FX: <words>` layers are derived: delete and regenerate
  them on every ability edit (Task 3c); a validator check flags orphaned FX layers whose words left the ability.
- **13. Stats above 10.** Survey the two cards (Sonic spd 20, Royal Eradicator Main Cannon atk 12), pick one canonical
  encoding (the template's digit layers run 0-9; a 10 is no visible digit), re-encode both, drop `EXCESSIVE_STAT_EXCLUSIONS`
  (five names in `commands/psd_parser.py`; Pix, Tainted Lazarus and Twin Emperors are all <= 10 today, so the whole list can go).
- **15. Text-first rulebook.** One text file per page in TTSCardMaker; pages generated by the fork's text engine; overflow
  paginates into continuation pages; fixes hidden layers, injection slips and undefined tags at the source (item 19).
- **6. Sign-in top right.** The existing `edit.js` GitHub-token sign-in, shown top right on every tab; nothing new to authenticate.
- **7. Profiles.** `site/profiles/<login>.json`, created and edited through the publish flow; page `u/<login>/`; a users page
  listing the people who made one.
- **8. Profile sections.** All optional: favourite and hated card and item, series, featured deck, motto, free label/value pairs.
- **9. Contribution data.** Authored cards (`site/Card Authors.txt`), recent edits from `site/Card Edits.json` `_by`/`_at`,
  commits from the history walk (the clone `commands/repo_history.py` keeps).
- **10. Contributions tab.** Beside the version viewer; one page per person over item 9's data.
- **11. Leaderboard.** By category, with titles; counts publish batches, not fields (a 20-field edit is one contribution).
- **5. Deck browser.** Public decks as repo files, private decks encrypted (decision above); the deck builder's `d/` share
  code stays the way to pass a hand around.
- **16. Play tab (simulator):** `docs/Campaigns/SIMULATOR_CAMPAIGN.md`.
- **17. Game reports; home of the shared report schema (its §4):** `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md`.
- **18. AI-generated cards:** `docs/Campaigns/AI_CARDS_CAMPAIGN.md`.
- **19. Rulebook consistency:** `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md`.
- **20. Inline editing** (smaller; agreed 2026-10-20). In edit mode the card browser is the editor. Three primitives only:
  1. Click a value to edit it in place: stat bar cells; the ability; double-click the name; fact chips (series, subseries,
     author, note) as inline comboboxes. Proposed, not yet agreed: stars, a token picker on `[` and tag completion on `(`,
     a type-row palette, rulebook title and text.
  2. Drag one or many cards onto an Author or Series facet to assign it (Subseries, Hidden and ban facets proposed).
  3. One bottom batch bar listing pending edits as sentences ("Goombella: spd 4 -> 5"), per-item revert, and the existing
     Review and publish.
  Each inline editor keeps the form's marker for edits that change the card image (name, stats, ability: PSD + database)
  vs database-only (Queued, "Two kinds of card edits"). The current form survives only as an "All fields" fallback reached
  from the batch bar. No new popups or panels; view mode unchanged.
- **21. Card PSD edits and standardization** (on hold: other developers are editing the PSDs; one batch when the user
  lifts it): the ledger of every intended PSD change, then an Action that reports on push;
  `docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md`.
  Status 2026-10-21: push guard: first rule, the font check, written against the uncommitted tree; live once both repos are committed and the `DISCORD_FONT_WEBHOOK` secret is set (TTSCardMaker `.github/workflows/font-check.yml` running `commands/font_check.py`; the bot's `/fonts`; the PSD ledger itself stays on hold).
- **22. Web editor** (later): an in-depth browser card editor once psd-tools is extended and the cards standardized; end
  state "card as data": one data file per card is the single truth, PSD and PNG are regenerated by the fork, the parser
  verifies the round trip, Card Edits.json dissolves, so PSD edits and database edits are one thing; migration in tiers,
  server-rendered previews first; grows from item 20; `docs/Campaigns/WEB_EDITOR_CAMPAIGN.md`.
- **23. Card details dropdown** (user, 2026-10-21): on the card browser, a per-card "Details" dropdown with every useful
  fact about the PSD: fonts used, each expandable to the layers using it (missing fonts flagged); text layers (text, font,
  size, colour, visibility); effects per layer; smart objects; canvas size, colour mode, ICC profile; type-icon style
  (image layers or font tokens); stat-row encoding; status-word FX and overlay layers; template match and ledger
  deviations; parser problems and PSD-vs-edit values; a collapsible raw layer tree as the catch-all. Data: the bot
  gathers it in /update_stats (it already opens every changed PSD) into a committed derived JSON keyed by PSD blob sha
  (no new pickle), pushed with /push; build_site.py writes one small file per card, fetched when the dropdown opens. The
  font check's per-blob cache, /fonts and the push guard read the same data. After the website foundation (6-8).

## Task 3c. Effect layers on pushes (after Task 3)

Port `MDW/Scripts/Effect Layers - All Cards.jsx` (Photoshop-only ExtendScript) to Python on the fork: find status
words with the text engine's character positions, cut them into `Ability FX: <words>` pixel layers, copy the layer
styles from the status chart PSD (the authority), render them, re-export the PNG; idempotent like "Update all".
Run it from the push pipeline for PSDs whose blob changed, only after a PSD has been quiet ~30 min (PSDs can't be
merged; don't race their author), skip its own commits, honour a skip list in `MDW`, and post what changed in
Discord. First prove it on all cards against the JSX batch's output.

## Task 4. Deep optimization and refinement pass

`docs/BUGS_AND_IMPROVEMENTS.md` §2.5 (network, speed, robustness, code health, concurrency, tests, docs).

**Status (2026-10-20): done** (one commit on main; numbers in BUGS §2.5, audits on `docs/Finished/optimization-pass-2026-10-20/`; scripts in `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/`).
- Tests: `tests/` (220 offline tests then, 238 today; 38 parser goldens, `requirements-dev.txt`); run with
  `~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests` before every push.
- Robustness: B4, B5, B7, C1-C3, C5, C6 fixed; `/list_guild_members` removed (50 commands); the queue loops survive a bad
  item; git network calls time out after 120 s; temp files always removed.
- Network and speed: 1 commit request per changed card, lookups beside downloads, one tree fetch shared by the card list and
  the emoji listing, fuzzy ability search x18, lazy cv2/yt_dlp/PyGithub; outputs identical (813 cards, 0 differences).
- Code health: `psd_analyzer.py` split into `card_models` / `card_edits` / `psd_parser` / `repo_traverser` behind the
  `psd_analyzer` facade (pickles unchanged), `github_api.py`, repo paths in `global_config.py`; 11 compiled modules, so a
  deploy MUST rebuild (`setup.py build_ext --inplace`).
- Docs: help.py and describe strings match all 50 commands; skills' line references and module map refreshed.
- Open (owner decisions or later): B8 (deletion alone doesn't prune), B9 (blob 5xx still records the tree ETag),
  admin gates on /purge etc. (B10, the 1 s queue tick and the one message per search result were done 2026-10-21).

## Order

**Task 3 first and foremost** (the user wants the next session spent on it); Task 3c follows Task 3. After Task 3: the
ideas list 5-15 in its agreed order, then items 16-23 (campaign docs, inline editing and the details dropdown; "Ideas 5-23" above; 21 waits for the user to lift the PSD hold). Tasks 0, 1,
1b, 2 and 4 are done and Task 2b was reverted, not rescheduled (status lines above); nothing else is scheduled.
