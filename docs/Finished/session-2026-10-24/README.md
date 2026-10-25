# Session 2026-10-23 evening to 2026-10-25 ~04:00

Worked from the plan of 10-23 17:30 (`PLAN_AS_OF_2026-10-23.md`, here). Rulings: memories `owner-decisions-2026-10-23-night`
and `owner-decisions-2026-10-24-night`. Commits since that plan: AutoCroissant 232 non-merge (main 0ca9f33), TTSCardMaker
167 (origin/main 86733bd2). Fork `cards` = `cards-20261025` (9edbc1fa). Also here: the 10-23 unbakeable list and the
10-21 standardization rehearsal, both finished.

## Landed

- **History.** Recovery run (PICKLE cf9fd20, 10-24 05:30). Backfill from every PSD version in git (PICKLE 23eecae, 17:30),
  after the two PSD fixes it waited on (The Thin Man ATK 7 59956ce4, Grimora DEF 4 0262b237). Text fix: the merge's added
  versions had borrowed the ability, types and stars of the version they split; 58 versions on 51 cards given their own
  blob's values (530e51d, PICKLE 1282743). History stays continuous across renames.
- **Cutover** (10-24 23:02, before the planned 10-26 shadow end): Card sync is the live data build, bake, TTS save and
  changelog; card-edits.yml and tts-save.yml retired; pickles frozen; `CARD_SYNC = True` (AC 89813ce, 0d4c9c1; TC
  8d884af4, 4eb5142d). Re-seed run 40166422140 from 1282743: 18 of 812 cards re-dated, data 85f23f66. The site reads
  `data` (TC c0bc3ae9). Ruleset "Protect main and data".
- **Clock in GitHub**: a card push starts a fixed 30-minute window (environment `card-sync-quiet`); later pushes ride
  along; a timer-ended run cancels itself; the cron is the fallback. `/update_stats` only syncs; admins' `force_update`
  and `cards:` still dispatch (AC 2c2b6d6, ca7194f; TC 1c17ab28, d92b3f06).
- **Schedules on :00/:30** (AC 429adb6, TC cacabfca).
- **Refresh on use**: card commands check the data branch at most every 2 minutes (ac65a00); the 30-minute
  `/update_stats` reminder is no longer needed.
- **Python 3.14 on every Action** and the guard decisions of 10-24 (AC 4adb2d8 and after): R7 left to font-check, G2
  rendered anyway, the quiet window as environment `card-guards-quiet`. Guards rounds 7-8 (G2 render fixes).
- **Actions**: uv installs, superseded caches pruned (TC a9d20286, 9e5ad58a; AC c387ca8).
- **Website**: speed pass (10-24 morning, ~20 TC commits); AVIF thumbnails and the no-change deploy gate (TC da24fc0e,
  7e41ad58); the service worker dropped for its first-paint cost (4504a736); open pages reload to a new deploy
  (719375bb); live by commit dates (8dc74408); a baked verdict's notes shown (0bb10ce9).
- **BUG 2** (the bake rebuilds `FX:` layers; BUGS D7) landed 10-24 evening after rounds 1-6 and five review rounds.
  Phase 6 (owner, 10-24 18:30, 48cfd94): content problems bake, flagged; census `set_text_layer` passes 2,401 -> 5,069
  of 5,642. Glow tokens bake (fork `cards-20261024b`, 1ed822c).
- **Abilities over several layers** (ff2a72d, 81a2a12) and **tab stops** (a829b47; fork `cards-20261025`, pin 52ed1fa).
- **Parser**: hidden titles are no alternate names (2c23d6e); rulebook pictures are no icons (90ef73c). Re-parses done:
  Shadow Duelist "", Papyrus "The Great Papyrus", Sans "snas.", Green Mage ATK 6, the seven Auxiliary stars; the two
  name edits dropped as redundant (TC 8c9c00e8).
- **Bot**: `/query`, `/query_ability`, `/query_rulebook` fixes (e46f0f0, d24e260, e1f83a3); `/reminders` panel and
  `reminder.json` (4d588d9); incremental Cython build, 4 jobs (083e37f, c76bf60); the deploy script moved into the
  `jetson-deploy` skill (0ca9f33).
- **Upkeep**: the jetson's OpenSSL 3.6.0 -> 3.6.5 in place (10-24; 3.5 LTS later if wanted); the `data` changelog's
  rewritten commit links remapped (data 4949abed, 10-24 20:38); commit attribution off in `~/.claude/settings.json` and
  `~/.git-hooks/pre-push` refusing Claude trailers (10-24).
- **Data and site, 10-23 night**: TTS ratings import (278094c), site round 3 stats and profiles, archive tags with
  Unfinished (TC 9b342437), 16 archived credits; TTS save 52 nicknames (TC 9a318fdd); rulebook page 19, status effects
  in phase 4 (TC d050875f).
- **psd-tools**: round 12 the last (`cards-20261024` = 6a616ca6: 5,160 -> 3,037 bad px on the frozen corpus; B11
  fixed); round 11 dropped; corpora, round branches and the r9/r10/r12/24b venvs deleted; `~/.venvs/ac-py314-fork` is the
  one fork venv (f46c8dd).
- **Census** (10-25 05:41 UTC, `docs/Future/PROBLEM_CARDS_2026-10-25.csv`): 6,712 dry bakes over 764 cards; 124 cards
  with an edit that can't bake (folder-set stars not counted), 4 data problems, 308 with a note. The 10-23 list read
  573 of 845 whole cards bakeable.

## Owner rulings (brief; do not ask again)

- Bakes don't refuse content: flag it. Layouts the bake can't edit get support built (split titles, non-template rows,
  tab stops, folder stars).
- Cards need no types and no ability. Adding one uses the card kind's template (N.M.E. and Field: `Item_Template`),
  flagged.
- A stars edit moves the PSD/PNG to the matching `N Stars` folder, creating it if missing.
- The website editor: one text box per PSD layer; a layer changed in Photoshop since the edit is skipped and flagged.
- Variant pictures: ` #N` layer tags; the 5 PSDs tagged by script, one commit.
- Overlay-safe badges: the outlined pill via the font's `salt` under any overlay, stickers too; the 31 badge FX layers
  retired in one commit after the owner's sheet.
- Only main writes wait 30 minutes; data, changelog, site and previews run at once.
- Retired: `/update_metadata`, `/mass_replace`, `/alias`, `/bake_cards`, `/update_tts_save` (code to
  `archive/discord-edits`); the website gains Rename everywhere.
- Retire the pickles, `psd_details.json` and `/push` now (step 10); the owner tells bobberlington.
- Next after the editor: the test cut, smaller PNG pushes, website small fixes; `game.txt` later.
- No developer notices; no guides for the owner (the census CSV only).
- History: God Tamer's and Blank Card's values are misreads; credits Firebomb MichaelJSr, Maxim Tomato Chestnut, The Abyss
  Watchers hdckighfkvhvgmk. Repo size: smaller PNG pushes, never rewrite existing PNGs.
- Push guards: a PNG pushed before its PSD stays current when it matches Photoshop's image.
- The guards fix Every-line Composer on card text layers silently (Single-line; never in posts or comments). The
  link preview fans the creators' cards. The badge batch is rebuilt on current PSDs, never over a newer push.

## Incidents and lessons

- **TTSCardMaker history rewrite (10-24 ~01:00)** to drop attribution trailers (84 commits, Git Data API). Developer
  clones are stranded: a Pull would merge the old commits back. Fix per clone: `git fetch; git reset --hard origin/main`.
  Lesson: never rewrite shared history; write messages without trailers.
- **A venv deleted under a running check** (`ac-py314-r12`, during cleanup): the check was re-run on `ac-py314-fork`.
  Lesson: look for running users (`ps`) before deleting a venv.
- **The backfill merge borrowed text** for its added versions (White Defender's 2024 version showed the 2025 ability).
  Caught after landing; the text fix ran once before the re-seed. Lesson: rehearse every field an added record carries.
- **GitHub's scheduler** ran hours late, so the clock moved to push triggers with an environment timer.
- Timings on the Mac swung ~2x: other projects' jobs kept the load at 60-100.

## Landed after 04:00 (10-25 morning)

- **Card sync split** (AC e9112e8, ab565f9, deployed e6a5c54; TC 44440a31, dad7aeae): a card push updates `data`, the
  changelog and the site at once; only main writes wait. First live main commit 59b97943.
- **Discord card-repo commands retired** (AC 6e8f8e3, e7e7879; code on `archive/discord-edits`) and the website's
  Rename everywhere (TC 706f750a, 02eef4ea).
- **Site**: the link preview fans the creators' cards (TC c8cf4643).
- **Guards**: a live fix commit (b2561b66: 4 PNGs re-exported, 6 recompressed) after hdckighfkvhvgmk's push 86733bd2,
  which also changed PoD06/PoD10/PoD11/PoD12 under the held badge batch (to re-run).
