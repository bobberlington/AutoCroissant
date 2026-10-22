# Next session plan (written 2026-10-22 06:00, Opus 5.5)

> **Resume note (2026-10-22 09:50):** a second session (05:51-) resumed this plan after the first hit its usage limit.
> Its running state, every landing and the owner's new answers are in
> `~/Documents/Tools/session-rescue-2026-10-22/resume/SESSION2_NOTES.md` (read it first; this plan is rewritten at its end).

Start in `~/Desktop/AutoCroissant`. Read memories `next-session-plan`, `jetson-server`, `ttscardmaker-integration`,
`psd-edits-hold`, `fork-no-upstream` and `agent-model-preference` first. Agents run on Opus or lower. Commit to main.
Report of the session that wrote this: `docs/Finished/session-2026-10-22/README.md`. The plan it replaced (with the
2026-10-21 inventory this one was checked against) is `docs/Finished/session-2026-10-22/PLAN_AS_OF_2026-10-22.md`.

Everything from the 2026-10-22 session is in git (pushed, or in local branches listed below) or in
`~/Documents/Tools/session-rescue-2026-10-22/` (`worktrees.tsv`, uncommitted patches, the workflow scripts and journals,
`r4/` notes and scripts, `std-batch/` copy). /tmp does not survive a reboot: work from the rescue copies if it is gone.

## State at the end of 2026-10-22

- **AutoCroissant** main e9154d2, deployed on the jetson 05:45 (52 commands). New today: deleted PSDs archived in
  old_stats (`archived`, restore on re-add); admin-only /purge, /update_metadata, /mass_replace, /alias, /set_repo,
  /delete_song, /delete_all_music; card_bake one commit per run with `--pending`; save(preview=); TTS save updater rules
  (item decks filled, `add_new_since` developer decks, names = website, team fields never touched); rules skill refreshed for
  the cooldown update; (Action) badge parity test. Jetson venv: fork tag `cards-20261022`. The bot is in guild 545138546021892096
  (Impossibility Simulator, Administrator). Reminders: 05:00 /update_stats, 05:30 /push.
- **TTSCardMaker** origin/main 75933ad5. Actions: site.yml (push + every 6 h at :17 UTC); tts-save.yml (daily 11:41 UTC, only
  when inputs changed); card-edits.yml (LIVE every 30 min behind the pending gate, plus dispatch; pinned `cards-20261022`);
  card-check.yml, font-check.yml. Website: series/subseries/authors creation from the filter bar (idea 25), Newest/Oldest
  created sort, card mentions by title or file name in any case, sticky headers, cooldown + (Action) badges. TTS save: teams
  Hearts (Red/Orange/Green) and Clubs (Blue/Purple/Pink), 48-line Lua, item decks filled, developer decks, Maxwell the cat,
  White Palace in the Morgue (file moved too), snap points refined, graveyard + Penalty Box mats enlarged; TTS/checks (24).
- **stoogedings**: 341 badges in the repo (a74786e1, (Action) added). `~/Library/Fonts/stoogedings.ttf` is still the
  340-badge font: install the new one only after the standardization batch stops using Photoshop, then restart Photoshop.
- **Fork** MichaelJSr/psd-tools: `cards` 0302b08b = tag `cards-20261022` (round 4 + review/cards-r3; >2% cards 5, max 6.42).
  The user's round-4 review fix `review/cards-r4` 3a63bc4f merges into `cards` after round 5. No upstream PRs, ever.
- **PSD hold LIFTED** (2026-10-22 02:40). Rules that stay: never race a teammate's PSD push (30 minutes), one commit per
  batch chunk, rehearsal/gates before landing.
- Expected lag on 10-22: five baked titles (White Clot, M.O.A.B., D.D.T., B.F.B., B.A.D.) reach the website after the
  05:30 PICKLE 077337a and its site build; until then TTS/checks fails 3 (contents_names, final_spot, updater_idempotent).

## In flight when the session ended (workflows do not survive the session; resume from their saved state)

1. **Standardization batch** (stoogedings tokens + badges on every card; workflow `standardize-stoogedings-batch`,
   script and journal in the rescue folder). At 05:42 the Batch phase was at card 249 of 349 (31 held in
   `manifest_held.txt`; "gate-red" overflow cards such as Spamton_Neo and Leeroy_Jenkins are held by the driver; the pilot
   check refuted Plague: its token runs off the card). Work dir `/private/tmp/std-batch` (copy:
   `session-rescue-2026-10-22/std-batch/`; `driver.log`, `manifest*.txt`, `cards/` outputs, `orig/`). Still to run: the
   driver to the end (resumable), **Gates** (parser equality, pixel region, contact sheets, stale check -> `GATES.md`), the
   adversarial gate check, **Land** (copy passed cards into `~/Desktop/TTSCardMaker`, commits <= ~900 MB by top folder,
   `LANDED.md`, not pushed). To resume: copy the script, paste the journal's Prep/Pilot/pilot-check results into it in place
   of those agents, run Batch (resumes), Gates, Land. Then push each commit yourself: `git pull --rebase` first (card-edits.yml
   may have baked), nobody else pushed a PSD in the last 30 minutes. Then install the 341-badge font, restart Photoshop,
   /update_stats + `diff_stats.py` gate, /push.
2. **Rendering round 5** (fork; script `/tmp/r4/round5.js`, copy in `session-rescue-2026-10-22/r4/`; args
   `{base: '0302b08b', ...}`). r5_base: 812 cards, 0 errors, >2% 5, max 6.42. Branches (local in `~/Documents/Tools/psd-tools`):
   fix/curves-cged 4acadeeb (verified twice, clean), fix/blend-channels c860023b (being fixed), fix/brightness-contrast-modern
   5e540e39 (verifying), fix/drop-shadow-knockout 756b0cae, fix/group-stroke-alpha 95dd7543, fix/gradient-multistop (worktree was
   dirty: patch in the rescue folder). Remaining: verify/fix each, integrate, harness vs r5_base (`score_cards.py <run>`,
   `compare_scores.py r5_base <run>`; never `--help`), merge `review/cards-r4` 3a63bc4f, tag `cards-2026102x`, re-pin
   card-edits.yml, install in the jetson venv (never 05:00-05:40) and `~/.venvs/ac-py314-fork`; goldens must pass. Then the
   owner decides the parity bar [user].
3. **Data build phase 1** (`data-build-phase-1`): the design `docs/Campaigns/DATA_BUILD_ACTION.md` (committed today, owner
   answers folded in §16). Branches: AutoCroissant `work/query-as-of` f78eb49 (`/query as_of:`, reviewed clean, ready to merge);
   TTSCardMaker `work/analytics` 2098d525 (balance analytics on Stats + per card; second review clean; NOTES §14 cites /tmp
   harness paths: move them into the repo or drop the cite); AutoCroissant `work/data-build` 6bfc5d6 (step 1 per-PSD failure
   isolation + step 4 card_data/card_history/card_db/changelog; second review refuted: `changelog.payload()` IndexError on a
   restore-only group, plus findings in the journal; `card_bake --local` left out). Finish the fixes, review once more, merge.
4. **Discord mining** (`discord-mining-run`): the miner runs as its own process (`~/Documents/Tools/discord-archive/mine.pid`,
   GET only, resumable cursors; 8.5 GB so far; raw data never in a repo). Then: image OCR name matching (`work/discord-match`
   b4f4bdd), `REPORT.md`, `ambiguous.html` for the owner's rulings + `apply_rulings.py`, creation dates and metadata; merge
   `work/discord-archive` 5da6bae (includes the user's review fix 5a82b31) and `work/discord-match`. Start the miner again with
   `--config ~/Desktop/AutoCroissant/config.py` if it stopped (bot-commands and github-updates channels can be skipped).

## Checklist (in order)

1. [ ] **Collect the in-flight work** above (1-4) and push it. Delete merged branches and their /tmp worktrees; delete the
   `wip/*` refs made for rescue (psd-tools: wip/rv4-bricon-*, wip/wt-r4-final, wip/wt-tag-r3) once nothing needs them.
2. [ ] **Checks after the first unattended day**: the website shows the five baked titles; TTS/checks 24/24 after the 11:41
   UTC tts-save run (`AUTOCROISSANT=~/Desktop/AutoCroissant LUA_PYTHON=/tmp/tts-checks-venv/bin/python`; rebuild that venv
   with lupa if /tmp was wiped); `gh run list -R MichaelJSr/TTSCardMaker --workflow card-edits.yml` green and quiet (the
   duplicate-PNG cards get one re-verdict); `/query White Clot` on Discord; the bake credit line.
3. [ ] **After the batch lands (Photoshop queue, owner's answers of 2026-10-22):**
   - install the 341-badge stoogedings into `~/Library/Fonts`, restart Photoshop;
   - (Action) badge on the 28 cards (36 uses) with the updated walk (CARD_STANDARDIZATION 2026-10-22 note:
     `std/pre-action-2026-10-22/`);
   - gradient-Ability cards (Star Cannon style): keep the gradient; research a way where badges may take the gradient too,
     labels readable (options in `psd-research/demo-2026-10-22/star_cannon_options/`);
   - the batch's held cards (31 + gate-red overflow + Plague; Baba stays held);
   - rulebook pages: fix (delete the 6 hidden layers; 20_Creature_Types `tornado` -> `wind`) and badge them; then the parser
     may skip hidden text layers on rule pages;
   - God Tamer = 5 stars, Bubble = 1 star (star grouping, L14);
   - Tainted Isaac bars = 6, Tainted Eden = 4: rename the layers to the drawn bars;
   - the six case-typo titles (Springtrap, Dark Matter, Metal General, Masked Dedede, Star Dream, Zero Override): name edits
     in `site/Card Edits.json`, baked by card-edits.yml;
   - Grey Prince Zote and Knight Lost A Lot: website name edits only, layers unchanged;
   - re-export the 5 stale PNGs (Restrained, Blood Bomb, Dark Mind Eye, Asteroid Field, Kel); this Mac's exports are accepted;
   - leave the 45 Name layers as they are (owner).
4. [ ] **Data build steps 5-10** (design §14): Discord writers (/update_metadata and /alias commit to Card Edits.json /
   Card Authors.txt; `del_entry` -> hidden), card-sync.yml dispatch-only dry run, shadow runs, cutover in one sitting (the
   30-min Card sync does bake -> build -> TTS save -> changelog, one Discord post per push to channel 1562752259199213680 via
   secret `DISCORD_CHANGELOG_WEBHOOK`; data on the orphan `data` branch), Changes page tabs (changelog, database edits), the
   bot reads the data branch. [user] creates `ACTIONS_TOKEN` (fine-grained, TTSCardMaker only, Actions read/write) and puts it
   in the jetson's config.py. Retire the pickles, psd_details.json and /push within a day of clean runs; tell bobberlington
   first. Never write the webhook URL anywhere.
5. [ ] **Docs and skills drift** from today's landings (run a drift audit): card-edits.yml cadence (every 30 min, `--pending`,
   one alert per failure streak) everywhere the docs say "dispatch only"; tts-save daily gate; admin-only commands in help and
   skills (done on the branch; re-check); the archive (`archived`, restore); (Action) badge counts (341); analytics and as_of
   once merged; companion skill sections for the save's teams/mats/snap points; `docs/BUGS_AND_IMPROVEMENTS.md` statuses.
6. [ ] **Memory**: `next-session-plan`, `jetson-server` (deploy e9154d2 05:45, fork tag), `ttscardmaker-integration`.

## Owner decisions of 2026-10-22 (do not ask again)

- Bot commands /purge, /update_metadata, /mass_replace, /alias, /set_repo, /delete_song, /delete_all_music: admin only (done).
- Fork: no upstream PRs; keep it tailored to our cards; PR_BODIES upkeep stopped. Parity bar: decided after round 5.
- TTS save: daily, and batched with the data build every 30 min once Card sync exists; the 7 Specific Item Decks are filled by
  their titles (count 1); Piko Piko Hammer in the full 3-star items decks and 3 Star Equips; every new card goes into the
  developer deck of its stars and the full decks ("new" = not in the save on 10-20, commit 318001e); names = website names;
  Description, GMNotes, Memo, Tags and scripts are the team's (never touched); the hand-picked Morgue deck keeps its typo'd
  names; graveyards and Penalty Box enlarged; no game-master view; teams give same-side zone use and visibility.
- Changelog: one post per push (not daily); Discord channel 1562752259199213680; a GitHub Action posts it.
- Data: orphan `data` branch in TTSCardMaker; Discord edits become card-repo edits (aliases too); 30-minute cadence;
  ACTIONS_TOKEN yes (owner creates); retire the pickles within a day of clean runs.
- PSD hold lifted; automatic baking on (30 min); the standardization batch runs on this Mac.
- Gradient Ability cards: keep the gradient (research badges). Split titles: website name, layers unchanged. Accept this Mac's
  exports. Fix all six case typos. God Tamer 5 stars; Bubble 1 star. Tainted Isaac 6 / Eden 4 (drawn bars). Rulebook pages:
  fix and badge. Add an (Action) badge (done: grey, exact case, tooltip "Costs the creature's action"). Leave the 45 Name layers.
- `archive/local-update-stats` pushed. AUDIT.md kept for now. Discord mining: all channels, OCR names, rulings by the owner.

## Waiting on the user (not blocking)

Parity bar after round 5; the rulings in `ambiguous.html`; creating ACTIONS_TOKEN; Photoshop checks of the batch's 5 sample
cards (`LANDED.md`); BabaUI font (Baba.psd); PSDs for the PNG-only Stubborn_Queen and Gust_of_Trees; Photoshop-authored
fixtures (a bevel texture, a smart object with a painted filter mask); the live TTS check (a teammate takes a card back out of
another colour's zone; a mod-cached player sees an updated card); inline-editing extras (stars, token picker, tag completion,
type palette, rulebook text) still proposals; campaigns' open questions (16 simulator Q1-7, 17 game reports Q1/2/5/6, 19
rulebook P0 frame and C/U/M/T/G/S items, 22 web editor Q2-9); the AI sandbox repo (campaign 18, D1-D3 to revise; Q3 budget,
Q4 how the bot runs it); S8's ten acceptance edits.

## Pending-task inventory (carried from the 2026-10-21 inventory; done items dropped, see the session report)

- **Fork/rendering**: round 5 (above); parked classes (Cannon name Satin, stat_labels PIN_LIGHT, stars residual, Jevil spd
  Perc, PoD02Black_Bloon, Hive_Knight glow, textured outer bevel, large shadows/glows, own-stroke type layers, gradient maps,
  class C/D/G); text engine limits (warp, platform AA, colour emoji); template-fill API (FontSet faces, layer move/rename, lfx2
  copy, smart-object content) for L2/L5; Task 3 done-when (no card > 2%, S8 identical in Photoshop) then retire Photopea to an
  archive branch; BrightnessContrast.automatic reads `auto` but the corpus stores `Auto` (round 5 note).
- **Bot**: D3 Shadow Duelist out of ABILITY_EXCLUSIONS after a golden-checked re-parse; D4 old_stats balloon cleanup
  (optional); extraction harness / gap-type validator / exclusion burn-down to a certified list after the batch; idea 14
  orphaned-FX validator; S9 star moves move PSD + PNG; push-guard deviation rules (report only); fix-on-push for 0-pixel
  classes later; Task 3c effect layers on pushes; validation skill baselines (Bubble, God Tamer NO STARS FOUND) after L14;
  V1 /sync_emojis refresh live, V2 a reminder firing watched, V4 live queries.
- **Standardization (beyond the batch)**: L2 stat rows (45 cards, 37 template rebuilds), L3 Sonic rows, L5 Ability layers
  (34 moves, 7 hidden, 20 blank), L6 retyped titles (Sonic, ABBI, Tyler, Minecraft_Panda, Pandanoko, Holy_Card) and hidden
  Names, L7 Effect Layers on 40 overlay cards, L11 rest (overflow pages, V1 16_The_Field box, icons after page numbers), L15
  class D renderer first, D1 Computer Virus stats, S5/S8 live acceptance, then ABILITY_EXCLUSIONS to the certified list.
- **Website**: build-speed hardlinks done; profile/People/Decks/Details items of NOTES §7-12 that are still "not done"
  (check the lists); Task 2b TTS "Table" tab stays reverted (the Play tab must not reuse "Table").
- **Rulebook (campaign 19)**: P0-P6 and the §3 items wait for the owner; idea 15 text-first rulebook after the page fixes;
  the rules skill's status-effect count.
- **Operations**: reminders edited only with the service stopped; jetson history clone: never diff in it (git 2.25);
  /tmp worktrees are cleanup candidates once landed.
