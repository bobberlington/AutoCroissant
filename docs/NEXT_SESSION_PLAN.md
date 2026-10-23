# Next session plan (written 2026-10-23 ~11:30, Opus 5.5)

Start in `~/Desktop/AutoCroissant`. Read memories `next-session-plan`, `jetson-server`, `owner-decisions-2026-10-22-evening`,
`persistent-work-paths`, `agent-model-preference` and `fork-no-upstream` first. Reports of the two sessions this plan
follows: `docs/Finished/session-2026-10-22b/README.md` and `docs/Finished/session-2026-10-23/README.md` (with the
archived `RESUME_2026-10-22.md`). The plan this one replaces is `docs/Finished/session-2026-10-22b/PLAN_AS_OF_2026-10-22_0600.md`.

**Standing rules:** work unattended, no questions unless the owner says "ask me questions" (then report the "Waiting on
the owner" list). Agents on Opus or lower. Commit to main (feature branches only while work is in flight). Persistent
paths only (`~/Documents/Tools/wt/`, `~/std-work-1022/`): reboots wipe /tmp. Never deploy between 05:00 and 05:40; push
and deploy in one sitting. One PSD rewrite per card per pass; never race a teammate's PSD push (30 minutes).

## State at the end (2026-10-23 ~11:30)

- **AutoCroissant** main 2ada357, deployed on the jetson 10:36 (52 commands, 764 cards + 48 rulebook, emojis ready).
  Pickles: PICKLE a7c310e (05:30): 818 cards, old_stats 446 versions / 354 names, 0 stored problems.
- **Fork** `cards` = tag `cards-20261023` (ecc4b773), in the jetson venv, `~/.venvs/ac-py314-fork` and the three Actions.
  Round 8 integrated (r8/integration 7011f518: bad px 270,512 -> 187,299, -30.8%), not pushed or tagged; round 9 building.
- **TTSCardMaker** origin/main f603db7c: release 96d61368, then the first live bake on the new tag (e1a491be, 11:29:
  PoD11ZOMG's name, 6 redundant entries dropped, 2 refused) and a TTS save (108 links). The Desktop checkout is 1 ahead
  (af841c87, Energy_Ritual, a follow-up's local commit). card-edits.yml is live: dry run 39996528699 stuck queued (still
  queued at 11:39: cancel it), dry run 40003158811 and the live run 40003504861 (e1a491be) passed; the 30-minute schedule
  takes over. e1a491be refused Papyrus and Sans (their title layer is named after the printed title).
- **Shadow started** (card-sync.yml schedule on, `MAIN_COMMIT`/`POSTS` off); at 11:39 no scheduled run yet, no `data` branch
  (GitHub's scheduler ran 3.4-5.6 h late for this repo on 2026-10-22, so a late first run is normal).
- **Push guards** landed, off (`CARD_GUARDS` unset).

## Checklist (in order)

### 1. The recovery run (next `/update_stats`: the 05:00 reminder on 2026-10-24, or the owner earlier)
- [ ] Expect "History: 698 versions recovered from the bot's PICKLE commits (553 cards), 171 starts moved earlier; 64
      cards gone before archiving archived ...; 42 type misreads merged ...; 12 impossible values set to unknown ...; 32
      brought to today's series ...; 299 carried starts given their commit." plus ~219-236 re-parsed PSDs (batch B2) and a
      second History line "103 carried starts given their commit" (noise). old_stats -> about 1,155 versions / 670 names.
- [ ] Before its `/push` (or after the 05:30 one), run the gate: `diff_stats.py <head stats> stats.pkl --history <head
      old_stats> old_stats.pkl` (autocroissant-change-control). Compare with the rehearsal; undo = revert that PICKLE.
- [ ] If it printed "History cleanup skipped" or "History clone ... not read" / "... is no full clone": check
      HISTORY_CLONE and the jetson checkout are full clones; the next run retries (debugging-playbook §1).
- [ ] Fix later: the second History line's 103 carried starts (noise) and the 14 Mario versions a second run moves
      (normalize after Card Edits; harmless).

### 2. The owner's re-parses (Discord, after step 1; tell him when)
- [ ] `/update_stats cards:Shadow Duelist` (only printed Ability text counts; it becomes "").
- [ ] `/update_stats cards:00 Table of Contents,Miracle Matter,Twin Emperors,Warcraftian Druid,Warcraftian Hunter`
      (the padding parser fix ed7c013 kept PARSER_VERSION 2; same blobs, no new version).
- [ ] `/update_stats cards:Azamoth the Possessed Vessel,Igor Roballtowski,Mettaton EX,Mettaton NEO,Militron (No Armor),Morpho Knight,Shadow Queen`
      (their printed stars 5/2/4/4/4/5/5; Auxiliary/Creatures stars ruling).
- [ ] Then `/push` (or the 05:30 reminder). Residual-gap cards should drop below 10 (extraction campaign scoreboard).

### 3. Shadow -> cutover (DATA_BUILD_ACTION.md §14 steps 7-10)
- [ ] **Seed check** (after step 1's `/push`): `gh api repos/MichaelJSr/TTSCardMaker/branches/data`. If the first scheduled
      run seeded `data` from a7c310e (before the recovery PICKLE), re-seed: dispatch card-sync.yml with
      `seed=<the recovery PICKLE>` and `dry_run=false`. Runner checkouts are shallow, so only the PICKLE carries the recovery.
- [ ] **Daily check** (diagnostics-and-tooling Workflow 4) after each jetson `/push`: diff_stats `stats.pkl` vs
      `origin/data`; only the predicted refinements. Also build_site with and without `CARD_DATA_DIR`, the changelog's
      events vs the card pushes, most scheduled runs ending at the gate.
- [ ] **Shadow ends** 2026-10-26 10:36 at the earliest, or after 10 card pushes, whichever is later.
- [ ] **Cutover sitting** (outside 05:00-05:40): reconcile the jetson (clean `git status`, its last `/push` on origin;
      else re-seed). The owner: `/remove_reminder reminder_id:054b5933` (05:30 /push), `/remove_reminder
      reminder_id:d2664752` (05:00 /update_stats), then `/set_reminder when:<:12 or :42> frequency:30m command:/update_stats`
      (keep 134e3b24). Then site.yml data checkout + `CARD_DATA_DIR` + daily cron; card-sync.yml `MAIN_COMMIT`/`POSTS` on,
      `dry_run` default false, `posted.json` at the last shadow event; delete tts-save.yml and card-edits.yml in the same
      TTSCardMaker commit; the bot commit `CARD_SYNC = True`; deploy; verify (site footer "Card data built ...",
      `/update_stats` "already current", `/query` the goldens, the next push on the site/save/Discord within one interval).
      `ACTIONS_TOKEN` is already in the jetson's config.py.
- [ ] **Step 9:** the Changes page's Changelog and Database edits tabs fed by `data/changelog.json`.
- [ ] **Step 10, within a day of clean runs:** retire the pickles, psd_details.json and `/push`. The owner tells
      bobberlington first.

### 4. Card guards rollout (owner: dry run -> report ~3 days -> on; 30-minute wait kept)
- [ ] Dry dispatches (Actions > Card guards, `dry_run` true) on a few recent card pushes (B2's, the Photoshop pass's).
- [ ] Set the repo variable `CARD_GUARDS=report`; watch the standards channel ~3 days (no ping; font-check keeps its ping).
- [ ] Then `CARD_GUARDS=on`. Before it, the owner items in PUSH_GUARDS.md §15 (below) and the Environment wait timer.

### 5. Follow-up workflows still running at the end (wf_4f5da896-004 and others; check their branches)
- [ ] **Glow halos / Nebulous Precipice**: AC `work/fu-glow-halo` (7d7c079: test at 0.3 em, companion §1.8 and the §4
      renderer row after the Photoshop test) and TTSCardMaker `work/glow-tokens` (c1a602ad: font at 0.3 em, fc5bd15c);
      land both together, then Nebulous_Precipice gets `[darkmatter:glow]`. Until TTSCardMaker's lands,
      `test_token_limit_fits_every_icon` fails against the live clone (site `text.js` reads 48 characters, not 52).
- [ ] **Effect Layers.jsx overlay colour** (the Photoshop pass's overlayColor port, Mr_Krabs' ruling): TTSCardMaker
      `work/l7-effect-layers` cdfef5fb, not pushed and not in release 2: review, push.
- [ ] **Energy_Ritual** (candidate A, +3.333 pt on line 4): local TTSCardMaker af841c87, unpushed: pre-push check, push.
- [ ] **PNG recompression** (lossless on push in the guard's bot commit, plus one batch pass over existing PNGs): AC and
      TTSCardMaker `work/png-optimize` (at 11:39 only TTSCardMaker 6345d293, pyoxipng in card-guards.yml). Note e1a491be's
      PNG went 8.7 MB -> 1.7 MB.
- [ ] **Stars/types bake** (the bake learns star layers in the Stars group and the type icon/token swap, same gates):
      AC `work/stars-types-bake` (nothing yet). Today a pending website stars/types edit blocks the whole card.
- [ ] **Guards' real-issue checks** (restore R3/R4's non-layout checks as reports: no or several visible Ability
      layers, a hidden or rasterized Name): AC `work/guards-real-issues` (00e1d08, with docs).
- [ ] **Test drift from B2** (Computer Virus, Fire Staff FX; old overlays pinned): AC `work/test-drift-b2` 3cd49da.
- [ ] **Pandora author**: the owner ruled it stays MichaelJSr (archive ruling recorded). Do NOT land any TTSCardMaker
      `work/pandora-author`; delete it.
- [ ] Check the live bake e1a491be: the 2 refusals (Papyrus, Sans: the title layer is named after the printed title;
      decide whether the refused-layers bake should take it), the new PNG against the PSD.

### 6. psd-tools rounds until dry (stop when a round removes < ~2% of what is left)
- [ ] Round 9 (4 branches: gradient scale proto, gradient stroke silhouette box, Photoshop-fixture one-carrier effects,
      type-layer gradient box; `psd-research/RENDER_STATUS.md` cards_r8 critic list). Integrate, harness vs r8_final.
- [ ] Release when worth it: push r8/r9 into `cards`, tag `cards-2026102x`, re-pin card-edits.yml, card-sync.yml and
      card-guards.yml, reinstall the jetson venv (gcc for `_distance`, `pip check`) and `~/.venvs/ac-py314-fork`; goldens.
      One AutoCroissant test failed against the live clone at r8 (`test_overlay_word_layer_without_a_chart_style_refuses`,
      from TTSCardMaker 49cf3b28): update it with the release. The fork's `psd_tools.__version__` still says 1.25.0 while
      its changelog heads 1.26.0 (unreleased): mind it when naming the tag. The harness corpus is the frozen copy
      `~/Documents/Tools/psd-research/corpus-6e50c3ee` (21 GB); keep it until the rounds end.
- [ ] Remaining classes: ~34% stale exports (class H: re-export in Photoshop), ~9.5% effect noise, Satin
      (Royal_Eradicator's Name), Oryx's inner shadow, soft outer-glow kernel per size. Fork leftovers to re-check:
      `BrightnessContrast.automatic` reads `auto` (corpus `Auto`); Star_Cannon's name box; type-layer gradients
      (The_Structure, Clot_White, Clot_Gold, The_Kingchin). Never merge `feat/layer-edit` 9233a172 as is.

### 7. Bot and bake follow-ups
- [ ] **BUG 2** (BUGS D7): rebuild `Ability FX: ...` badge layers in the bake (approved; text geometry and colour
      glyphs are in the fork now). Until then 18-20 gradient badge cards refuse ability bakes.
- [x] Re-check the 4 stale verdicts (Metal General, Z.O.M.G., Star Dream, Zero Override) without `--pending`: none
      left (work/bake-verdicts). The live dispatch e1a491be re-checked them (none was final: three PSDs had changed, and
      Z.O.M.G.'s and Zero Override's parity reasons sit under PARITY_FLOOR), dropped the three names as redundant (the
      10-23 Standardize title case prints them) and baked Z.O.M.G.; its `baked` verdict is current (psd 9b87525a) and
      stays beside series/aliases.
      A dry run at 5c88f230: 2 cards with PSD fields (Papyrus, Sans), verdicts unchanged, nothing pending. No code bug.
- [x] PoD08MOAB, PoD09DDT, PoD10BFB, PoD12BAD: stale Txt2 stories on origin? No: in sync since 5a9e0fcf; 0 of 904
      PSDs stale at 6a54b4f6 (2026-10-23).
- [x] Green Mage: stat rows below the canvas (an error, not a verdict). work/bake-verdicts: `set_stat` refuses a bar
      wholly off the canvas, or a single digit mask (`unsupported-card: hp: the bar 'Hp Dark' lies outside the card ...`),
      a final verdict; the only such PSD of 812 on origin/main. Its printed ATK is 6, the bot reads 10 (BUGS D8: owner).
- [ ] Badge the 11 EarthboundZero cards, Koro-Sensei, The Evil Idol and Papyrus with the fork (owner ruling; colour
      glyphs landed in round 6, so this is unblocked).
- [ ] The research waits (NEXT_IDEAS_2026-10-22.md): the open-once patch (`wt/feat-speed-scratch/ac-open-once.patch`),
      rcache wiring in psd_edit, text geometry in the data build and a website overlay, template fill stage 2, the
      Photoshop oracle loop (approved; Photoshop is free since B2), the Perpetua lead (Hollow Knight).
- [ ] Extraction campaign: the structural candidate (blank-but-present Ability = no problem: ABILITY_EXCLUSIONS 21 -> 0),
      the gap-type validator, idea 14 (orphaned-FX validator), S9 (star moves move PSD + PNG), fix-on-push for 0-pixel
      classes, Task 3c (effect layers on pushes), L11 rest, L15 class D renderer.
- [ ] Live checks never done: V1 `/sync_emojis` refresh, V2 a reminder firing watched, V4 live queries; the website's
      `created` dates after the release (772/772 expected).

### 8. Website, TTS and Discord follow-ups
- [ ] Fetch the 10 oversize Discord files (ruled 2026-10-22 ~23:20); re-run the matcher if they move a date.
- [ ] The cards to open in Photoshop: five in `~/Documents/Tools/psd-research/std-batch2/b1_final/LANDED.md` (batch 1)
      and five in `std-batch2/LANDED_B2.md` (02_Abbreviations, 22_Negation, Palace_of_Shadow, Jack-In-The-Box_Zombie,
      Dark_Mind_Eye).
- [ ] Website NOTES §7-12 items still "not done"; Task 2b's TTS "Table" tab stays reverted.
- [ ] Rebuild lost environments when needed: `tts-checks-venv` (lupa), `tc-parity` (use `wt/tc-full.git`),
      `/tmp/fontexplore` (BUGS §1.6 points there).

### 9. Cleanup
- [ ] Worktrees in `~/Documents/Tools/wt/` whose branches are merged (ac-*, idea-*, fu-* once landed, rel-*, soft-parity*,
      r6-*/r7-* once pushed, `scratch/*`, research-*-scratch), and `wt/docs-1023` after this lands.
- [ ] Merged `work/*` branches in AutoCroissant and TTSCardMaker; TTSCardMaker backups `std/batch-landed-unreviewed`,
      `std/landed-243`, `std/queueA-landed`, `backup/b2-pre-exclude`; psd-tools `wip/*` refs and merged round branches.
- [ ] The 23 GB rescue folder `~/Documents/Tools/session-rescue-2026-10-22/` (B2 has landed; keep the notes and
      bundles until the branches are pushed or deleted); in `discord-archive`: `*.pid`, `match/before-*`, `verify/`;
      the 21 GB frozen corpus `psd-research/corpus-6e50c3ee` once the psd-tools rounds end.
- [ ] The Mac's local TTSCardMaker checkout: shallow; `git pull --ff-only` after the Energy_Ritual push.

### 10. Docs and memory
- [x] Memory `next-session-plan` (points here) and `jetson-server` (2ada357, fork `cards-20261023`), 2026-10-23 11:40.
- [ ] Memory still stale: `ttscardmaker-integration` (since 10-21: font-check webhook, card-sync shadow, password
      sign-in, `.mailmap`, created dates), `website-and-pipeline-ideas` (parity bar, the L2/L5/L6 standardization items
      the owner turned down), `psd-edits-hold` (batch rules of 10-21), `commit-to-main` (work branches in flight),
      `owner-decisions-2026-10-22-evening` (card-edits.yml is live again).
- [ ] ttscardmaker-companion §0's Card guards row still says "not landed" (it landed off in 96d61368): fix it with or
      after `work/guards-real-issues`, which edits the same row.
- [ ] Skills still carrying 2026-07-11 line numbers outside today's drift fixes (impossibility-cards-reference's
      psd_parser.py citations, failure-archaeology); re-run `scripts/optimization/refs6.py` / `provcheck.py`.
- [ ] DATA_BUILD_ACTION.md §6.1, §6.3 and §14 as built (`_bot` described); update as steps 7-10 happen.

Secrets and repo switches, by name only: autocroissant-config-and-flags (TTSCardMaker `DISCORD_CHANGELOG_WEBHOOK`,
`DISCORD_STANDARDS_WEBHOOK`; `DISCORD_FONT_WEBHOOK` deleted 2026-10-23; `CARD_GUARDS`; card-sync's `MAIN_COMMIT`/`POSTS`;
the jetson's `ACTIONS_TOKEN`).

## Waiting on the owner (report these; do not ask unless told)

- **Cutover:** the reminder swap (step 3) and telling bobberlington before step 10.
- **Push guards** (PUSH_GUARDS.md §15): the GitHub Environment wait timer (a Settings item); R11's exact-name rename
  (fix or keep telling); whether the guard should leave R7 to font-check; the fork render's refusals (9 of 34 then; 5 of 34
  with soft parity); Photoshop's image for a PNG pushed before its PSD; G2 (no Maximize Compatibility refuses); the
  "outside R" parity method; the login fallback and the commit link in the post.
- **History decisions left by the audit** (DATA_BUILD_ACTION.md §7.4): high stats kept by an allowance (Sonic 23, Pix, Twin
  Emperors, Royal Eradicator Main Cannon, archived Bill Cipher and Oryx) or set unknown (God Tamer's 15, Blank Card's 55s,
  Tainted Lazarus' 110s, the 10+ stars); 26 restored versions that differ in types alone while the PNG changed; whether a
  history starts at its bytes' first commit at any path; the 4 renamer credits (Abyss Watchers and Pontiff Sulyvahn:
  hdckighfkvhvgmk; Firebomb and Maxim Tomato: MichaelJSr); the dangling side-file lines (White Palace in Card Authors.txt;
  Gust of Trees and Stubborn Queen in Card Creation Dates.json), printed by every run as "match no card".
- **To tell him:** BUGS D4's owner notes (a card back after a deletion gap keeps its first start); the `_bot` field
  (schema addition); ability bakes refused on the gradient badge cards (BUG 2); 4 cards keep their hidden-layer text
  (they print it as visible layers); the Website edits cap of 100; per-field credit on 14 entries; Computer Virus left
  flagged.
- **Older items:** BabaUI font (Baba.psd); Photoshop-authored fixtures (a bevel texture, a smart object with a painted
  filter mask); the live TTS check (a teammate takes a card back out of another colour's zone; a mod-cached player sees an
  updated card); inline-editing extras (stars, token picker, tag completion, type palette, rulebook text); campaign
  questions (16 simulator Q1-7, 17 game reports Q1/2/5/6, 19 rulebook P0 frame and C/U/M/T/G/S, 22 web editor Q2-9); the
  AI sandbox repo (campaign 18, D1-D3; Q3 budget, Q4 how the bot runs it); S8's ten acceptance edits; repo size (LFS or a
  TTSCardMaker2 only if GitHub complains).

## Not chosen (don't build unless asked)

Printed-text audit; Discord `/preview`; the website edit-support map; interactive website cards; template passes L2/L3/L5/L6
(stat-row templates, Sonic rows, Ability-layer moves, retyped titles: "give our developers freedom"); point badges (points
stay text); font-side hardening.
