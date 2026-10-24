# Next session plan (written 2026-10-23 ~17:30, Opus 5.5)

Start in `~/Desktop/AutoCroissant`. Read memories `next-session-plan`, `jetson-server`, `owner-decisions-2026-10-22-evening`,
`persistent-work-paths`, `agent-model-preference` and `fork-no-upstream` first. Reports of the sessions this plan
follows: `docs/Finished/session-2026-10-22b/README.md` and `docs/Finished/session-2026-10-23/README.md` (morning
release and the afternoon section). The 11:30 version of this plan is in git history (68bf5a3); raw running notes:
`~/Documents/Tools/session-rescue-2026-10-22/resume/SESSION3_NOTES.md`.

**Standing rules:** work unattended, no questions unless the owner says "ask me questions" (then report the "Waiting on
the owner" list). Agents on Opus or lower. Commit to main (feature branches only while work is in flight). Persistent
paths only (`~/Documents/Tools/wt/`, `~/std-work-1022/`): reboots wipe /tmp. Never deploy between 05:00 and 05:40; push
and deploy in one sitting. One PSD rewrite per card per pass; never race a teammate's PSD push (30 minutes).

## State at the end (2026-10-23 ~17:30)

- **AutoCroissant** main 5f5fe82; the jetson runs 1bf6892 (deployed 15:52; the two commits after it are tests only).
  52 commands. Pickles: PICKLE a7c310e (05:30): 818 cards, old_stats 446 versions / 354 names.
- **Fork** `cards` = tag `cards-20261023c` (fa031443, round 10), in the jetson venv (17:02), `~/.venvs/ac-py314-fork` and
  the three Actions (TTSCardMaker 82a8ba7c). Bad px on the frozen corpus: 605,755 (round 6 base) -> 138,578 (-77%).
- **TTSCardMaker** main 82a8ba7c. Landed this afternoon: glow font 0.3 em (0fcbcdfe), Energy_Ritual cand A (41e2ddea),
  Effect Layers.jsx overlay colour (08b1152a), PNG recompression (6a54b4f6) and the one batch pass (56011634: 686 PNGs,
  4.18 GB -> 1.14 GB, the same pixels, as AutoCroissant with `Recompressed` trailers), TTS save re-pinned (5c88f230),
  types in any order (6dd5399e), fork pins (2921ecbd, 82a8ba7c).
- **Push guards:** `CARD_GUARDS=on` since 2026-10-23 17:34 PDT (report from 13:04; dry run 40012619052: 812 cards, 16 told, 0 errors).
- **Shadow:** card-sync.yml schedule on, `MAIN_COMMIT`/`POSTS` off. The `data` branch exists since 2026-10-23 14:21
  (63bd6603, seeded from the 05:30 PICKLE a7c310e, before the recovery: re-seed in step 3).
- **Website:** 772 cards, every one with a created date (0 provisional; the 48 rulebook pages have none by design);
  232 cards wait for a re-parse (B2), which step 1 does.
- **In flight at the end:** BUG 2 (AC `work/fx-rebuild`, worktree `wt/bf-fx`, review loop wf_97b6ba0d-096) and fork
  round 11 (rounds workflow wf_502b9ebf-b59, cap 11). See steps 5 and 6.

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
- [ ] `/update_stats cards:Papyrus,Sans` (titles named after their own text, 1bf6892: "The Great Papyrus", "snas.";
      no PARSER_VERSION bump, BUGS V6).
- [ ] Then `/push` (or the 05:30 reminder). Then dispatch the bake once so the two name edits drop as redundant:
      `gh workflow run card-edits.yml -R MichaelJSr/TTSCardMaker -f dry_run=false` (their refusals are final for the
      schedule). Order matters: re-parse first, else the site shows "Papyrus"/"Sans" until the re-parse.

### 3. Shadow -> cutover (DATA_BUILD_ACTION.md §14 steps 7-10)
- [ ] **Re-seed** (after step 1's `/push`): `data` was seeded from a7c310e at 14:21 on 2026-10-23 ("Recovery ... skipped"),
      so dispatch card-sync.yml with `seed=<the recovery PICKLE>` and `dry_run=false`. Runner checkouts are shallow, so
      only the PICKLE carries the recovery. Then `gh workflow run card-sync.yml -R MichaelJSr/TTSCardMaker -f
      cards="Papyrus, Sans" -f dry_run=false` (the data branch re-reads a card only when its blob changed or it is named).
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
- [ ] **Owner, 2026-10-23 ~18:30 (DATA_BUILD_ACTION §16 item 7):** bake verdicts move from Card Edits.json to
      `db/verdicts.json` on `data`, and bake/guard events (`baked`, `bake-refused`, `guard-png`, `guard-fix`) join the
      changelog. Build after BUG 2 lands, with steps 7-9.
- [ ] **Step 10, within a day of clean runs:** retire the pickles, psd_details.json and `/push`. The owner tells
      bobberlington first.

### 4. Card guards rollout (owner: dry run -> report ~3 days -> on; 30-minute wait kept)
- [x] Dry run over every card (40012619052, 2026-10-23): 812 cards, 16 told (R1 badge-font tags on Springtrap, Terry
      Hintz, Joel Miller, Future Niiue and more; R6 Nasty_Patty, Shell_Shield; R8 PoD08/09/10/12 Name composer), 0 errors.
- [x] `CARD_GUARDS=report` since 2026-10-23 13:04 PDT. The batch commit 56011634 ran as "0 cards, 686 left out (bot
      commits)", nothing told.
- [x] `CARD_GUARDS=on` since 2026-10-23 17:34 PDT (owner: now). Watch the standards channel and the bot commits; R11
      renames stay told (owner). Left in PUSH_GUARDS.md §15: the Environment wait timer and the other items below.

### 5. Follow-ups still open at the end
- [ ] **BUG 2** (the bake rebuilds `Ability FX:` layers instead of refusing 18-20 gradient badge cards; BUGS D7):
      AC `work/fx-rebuild` (worktree `~/Documents/Tools/wt/bf-fx`). Two review rounds found silent mis-bakes (Goofy's 1#1,
      badges typed as text, status layers spreading), all fixed; q6 decided unattended (rebuilt layers keep their own
      style). Review loop wf_97b6ba0d-096 running at the end: land only after a clean review, merged with origin/main,
      full suite on the fork venv. psd_parser changed too (round 3's rest): pull, Cython rebuild, restart on the jetson.
- [ ] Nebulous_Precipice glow: kept `[darkmatter:sticker]` (unattended option C; options A-D in
      `~/std-work-1022/glow/held.json`). A stroked Text group swallows a `:glow` halo (652 of 765 cards stroke it).
- [ ] e1a491be's new PNG against its PSD (the first live bake).

### 6. psd-tools rounds until dry (stop when a round removes < ~2% of what is left)
- [ ] **Round 11** (resumed wf_502b9ebf-b59 with the cap raised to 11): stroke corners (`effects.py _signed_distance`
      `near = 0.5 - alpha` overstates a quarter-covered corner pixel by ~0.25 px; 3,103 px on 291 cards; needs Photoshop
      fixtures) and The_Spire's Name (vibrance). Bar ~2,770 px. If it integrates and audits clean: tag
      `cards-2026102x`, re-pin `requirements-cards.txt` (the one pin since 2026-10-23), reinstall the jetson venv with
      `--force-reinstall` (gcc builds
      `_distance`; check `composite/noise_field.npz` is installed; `pip check`) and `~/.venvs/ac-py314-fork`; AC suite on
      the installed tag first.
- [ ] After the rounds: delete the frozen corpus `~/Documents/Tools/psd-research/corpus-6e50c3ee` (21 GB), the fork's
      round worktrees and branches (r6-* .. r11-*, `wip/*`), `~/.venvs/ac-py314-r9` and `-r10`.
- [ ] What is left: 93% of the bad px are 18 stale exports (class H: PoD06, Jevil, Clot_White, Disarmed, Credit_Card,
      Forget_Me_Now, Inner_Eye, Empowered, Enraged, ...; `RENDER_STATUS.md` cards_r10). Re-exporting them in Photoshop is
      the owner's call (it changes the published PNG). The fork's share is ~9.1k px. Never merge `feat/layer-edit` 9233a172.

### 7. Bot and bake follow-ups
- [ ] **BUG 2** (BUGS D7): built on `work/fx-rebuild` (worktree `~/Documents/Tools/wt/bf-fx`, 2026-10-23): the bake
      runs Effect Layers.jsx's Update all on the new text (review findings fixed on the same branch, three rounds;
      origin/main merged in each round, last 7ca804e). Phase 2 (owner's answers of ~17:40): own-style FX layers
      rebuilt, typed badges and icons set in the icon font, the card's `:sticker`/`:tint` spellings kept (UNBAKEABLE BUG 2 block). Phase 2
      review fixes (same branch, origin/main 8983719, then 8cbb211, merged in): a badge edited by deleting letters keeps its badge
      layer, gate `bare_icons`, typed badges refused under any tint (Pin Light, a 69% gradient) without a badge layer,
      a typed "★" set in the icon font, missing glyphs refused, an edit inside an own-style FX phrase refused. Round 2
      (same branch, origin/main be280e6, then 73dfc3c, merged in): a typed copy joining an own-style layer takes its words' colour (Palace of Shadow's "+2 DEF"
      printed grey), gate `gate_fx_colours`; a moved own-style phrase and letters glued onto one refuse; the parser's
      `#0` change reverted. Round 3 (2026-10-24, same branch): a badge layer takes its colour glyphs' red-less pixels
      (a typed `[undead]`'s eyes printed in the overlay's colour), `gate_fx` and `bare_icons` let no icon pixel out (they
      let 1%), "[fire:]" and any token the website draws as an icon but the font prints as letters refuse `font`
      (`unlettered_icons`; it baked as the letters), gate `lettered_icons`, an edit inside a phrase of several colours
      refuses, no status word inside a `[token]`, the ability edited before the title (Fire Staff's name + ability
      refused `bytes`). Round 3's rest (2026-10-24 ~01:00, same branch, origin/main 2427a23 merged in): the parser
      reads "[fire:]" in the icon font as typed, an UNKNOWN ICON TOKEN (`psd_parser.token_variant`, the rule `icon_text`
      shares; no card on main has one), gate `unshaped_icons` (each `icon_text` span shapes to one icon-font cluster;
      0 of 381 icon-font Abilities fail it), a whole ability deleted on an FX card refuses as blank (`reparse`: ABILITY
      TEXT NOT FOUND, its layers deleted; it said `styled-word`), and a bake opens each PSD once (`parse_card` takes the
      open PSD, `gate_reparse` the reopened one, `apply_jobs` edits a copy; `test_a_bake_opens_each_psd_once`); full suite after it (4563194): 1550 passed, 12 skipped, 0 failed.
      Suite on the fork venv after round 3: 1544 passed, 12 skipped, 1 failed
      (`test_card_check.py::test_clone_readers_match_the_traversers`: a Finder `.DS_Store` in the clone's `Types/`;
      origin/main 7ca804e, merged in after, fixes it: `test_card_check.py` 24 passed). Land it after a clean review and deploy (psd_parser is a Cython module: pull, rebuild, restart). q6 (own
      style kept) answered yes; Clot Gold to confirm (its "2T1" made "2T2" now refuses). Before landing: open one bake
      that added the icon font to a plain card's FontSet (Ice Staff, Standstill, Mettaton NEO...) in Photoshop.
      Owner questions held: a typed plain token beside a card's `:sticker` icons (Palace of Shadow; a moved line's
      `[dark:sticker]` prints `[dark]` too: keep_tokens can't pair moved text), a token's `:sticker`/`:tint`/`:glow`
      variant typed on the website counted redundant (classify compares canonical_tokens, which fold the variants: the
      edit leaves Card Edits.json with no bake and no verdict, a `:glow` one too, which should refuse `font`; the token
      picker offers these edits), words typed in place of a badge or icon refused (Pizza's "(1CD)" -> "Once:", Gasoline's
      "[fire] type" -> "fire type": edit_styles keeps the icon font; near() could give the body style), the leading space of
      15 Abilities (Elmadan's) dropped by an ability edit (keep_ends keeps paragraphs, not the indent, by its test), a
      retyped token taking the card's commonest icon style (ruling C: Plague's skull 41.17 -> 52.03 pt when swapped),
      and whether a chart-style status layer joined by a typed word should be told too (only own-style ones are: Marx "Glued.
      Glued" gives no note; it looks the same as a new layer). Over-cautious refusals, accepted: 6 of the 91 icons typed
      on Turtle (flare pixels past their glyph boxes: light, the 6/7/8 stars, silksong_star), replacing an own-style
      status word ("2T Smoldering" -> "2T Burning" on Fire Staff). Website track (TTSCardMaker, not edited here): text.js
      `icon()` still previews "[fire:]" (a colon with an empty variant) as the fire icon, and `norm()` forgives "[Fire.]"
      or "[warp  star]"; the bake refuses them (`font`), so the editor should treat them as text like the font does.
      **Open owner items (BUG 2, not implemented):** (1) a badge or token typed in a title stays body text (ruling C says
      "wherever it is typed"); (2) a typed `:sticker`/`:tint` token takes the fill of the text around it, not the card's
      own token colour (Palace of Shadow's typed `[dark:sticker]` outline dark grey 0.2, its own white 1.0; the card is
      itself mixed: its `[light:sticker]` uses the body grey); (3) a badge typed next to an odd-size badge takes that
      badge's size (in front of Pizza's 18.66 pt first badge, not the card's usual 32 pt). Not this branch's: inserted text takes the nearest coloured word's fill (Binge Eater's ", 1T"
      olive after "SPD"); `normalize_ability`'s quote and space-before-punctuation stripping can drop characters on any
      ability edit.
      Not this branch's: appended text can run below the card's text panel artwork (the overflow gate checks the
      engine's box and the canvas: Autopsy Report with " Costs 5★ and [fire:tint]." adds a third line under the panel).
- [ ] Badge the 11 EarthboundZero cards, Koro-Sensei, The Evil Idol and Papyrus with the fork (owner ruling; unblocked).
- [x] FX layers in no chart status's style (Morpho Knight, The Watcher, Computer Virus, Mr Krabs, Palace of Shadow,
      Fire/Ice Staff, Flameblade) are rebuilt in their own style (owner 2026-10-23 ~17:40): built on `work/fx-rebuild`
      phase 2 (42 of 48 dry bakes). Also there: every typed stoogedings token set in the font (62 of 85).
- [x] Tests drifted from the clone: closed by origin/main 3bb3ede (Green Mage and Shell_Shock pinned) and 03cd785
      (BabaUI in the repo, R7 waiver gone). Full suite on `work/fx-rebuild` after merging 8cbb211 (2026-10-23 night,
      fork venv `ac-py314-fork`): 1481 passed, 12 skipped, 0 failed. After round 2 and merging be280e6: 1493 passed,
      12 skipped, 1 failed (`test_clone_readers_match_the_traversers`: Finder's untracked `Types/.DS_Store` in the
      local clone, 2026-10-23 21:26; passes without it). Then 73dfc3c merged (site tests only: 57 passed).
- [ ] The research waits (NEXT_IDEAS_2026-10-22.md): the open-once patch (`wt/feat-speed-scratch/ac-open-once.patch`),
      now only its bake half (the data-build half is on main since 4dcf33f, guarded by `tests/test_open_once.py`):
      built on `work/fx-rebuild` (2026-10-24: `parse_card` takes an open PSD, `gate_reparse` gets the bake's
      `reopened`, card_bake opens the old PSD once and `apply_jobs` edits a `deepcopy`); rcache wiring
      in psd_edit, text geometry in the data build and a website overlay, template fill stage 2, the Photoshop oracle
      loop (approved; Photoshop is free), the Perpetua lead (Hollow Knight).
- [ ] Extraction campaign: the structural candidate (blank-but-present Ability = no problem: ABILITY_EXCLUSIONS 21 -> 0),
      the gap-type validator, idea 14 (orphaned-FX validator), S9 (star moves move PSD + PNG), fix-on-push for 0-pixel
      classes, Task 3c (effect layers on pushes), L11 rest, L15 class D renderer.
- [ ] Live checks never done: V1 `/sync_emojis` refresh, V2 a reminder firing watched, V4 live queries, V6 (Papyrus and
      Sans after step 2).

### 8. Website, TTS and Discord follow-ups
- [ ] The cards to open in Photoshop: five in `~/Documents/Tools/psd-research/std-batch2/b1_final/LANDED.md` (batch 1)
      and five in `std-batch2/LANDED_B2.md` (02_Abbreviations, 22_Negation, Palace_of_Shadow, Jack-In-The-Box_Zombie,
      Dark_Mind_Eye).
- [ ] Website NOTES §7-12 items still "not done"; Task 2b's TTS "Table" tab stays reverted.
- [ ] Rebuild lost environments when needed: `tts-checks-venv` (lupa), `tc-parity` (use `wt/tc-full.git`),
      `/tmp/fontexplore` (BUGS §1.6 points there).

### 9. Cleanup
- [x] 2026-10-23: merged worktrees and branches in both repos, TTSCardMaker's backup branches (SHAs in the notes), the
      rescue folder's batch copies (45 GB), discord-archive leftovers, ~98 GB of large files in finished `wt/scratch/*`.
- [ ] Left: `work/land-st` (BUG 2's base, delete after it lands), the fork's round branches and the corpus (step 6),
      `~/std-work-1022` (38 GB of Photoshop copies; keep `glow/held.json` and the evidence the docs cite),
      `wt/scratch` (~20 GB), `wt/tc-full.git` (16 GB, the rehearsal harness reads it).

### 10. Docs and memory
- [ ] Skills still carrying 2026-07-11 line numbers outside the 10-23 drift fixes (impossibility-cards-reference's
      psd_parser.py citations, failure-archaeology); re-run `scripts/optimization/refs6.py` / `provcheck.py`.
- [ ] DATA_BUILD_ACTION.md §6.1, §6.3 and §14 as built (`_bot` described); update as steps 7-10 happen.

Secrets and repo switches, by name only: autocroissant-config-and-flags (TTSCardMaker `DISCORD_CHANGELOG_WEBHOOK`,
`DISCORD_STANDARDS_WEBHOOK`; `DISCORD_FONT_WEBHOOK` deleted 2026-10-23; `CARD_GUARDS` = report; card-sync's
`MAIN_COMMIT`/`POSTS`; the jetson's `ACTIONS_TOKEN`).

## Waiting on the owner (report these; do not ask unless told)

- **Answered 2026-10-23 ~17:40** (memory `owner-decisions-2026-10-22-evening`): Nebulous keeps the sticker; R3/R4
  reports stay; rebuilt FX layers keep their own style; off-chart status colours are recoloured to the chart; FX layers
  in no chart style get rebuilt in their own style; every typed stoogedings token (timing tags, types, any token the font
  draws) is set in the font by the bake; icon swaps carry the old icon's effects; Green Mage's PSD fixed (TTSCardMaker 78798eac: ATK 6, close BUGS D8 after step 1); all stale
  exports get re-exported; `CARD_GUARDS=on` now (17:34); R11 renames keep telling; the fork rounds stop after 11.
- **Cutover:** the reminder swap (step 3) and telling bobberlington before step 10.
- **Push guards** (PUSH_GUARDS.md §15): the GitHub Environment wait timer; whether the guard should leave R7 to font-check; the fork render's
  refusals; Photoshop's image for a PNG pushed before its PSD; G2 (no Maximize Compatibility refuses); the "outside R"
  parity method; the login fallback and the commit link in the post.
- **History decisions left by the audit** (DATA_BUILD_ACTION.md §7.4): high stats kept by an allowance (Sonic 23, Pix, Twin
  Emperors, Royal Eradicator Main Cannon, archived Bill Cipher and Oryx) or set unknown (God Tamer's 15, Blank Card's 55s,
  Tainted Lazarus' 110s, the 10+ stars); 26 restored versions that differ in types alone while the PNG changed; whether a
  history starts at its bytes' first commit at any path; the 4 renamer credits (Abyss Watchers and Pontiff Sulyvahn:
  hdckighfkvhvgmk; Firebomb and Maxim Tomato: MichaelJSr); the dangling side-file lines (White Palace in Card Authors.txt;
  Gust of Trees and Stubborn Queen in Card Creation Dates.json), printed by every run as "match no card".
- **To tell him:** BUGS D4's owner notes (a card back after a deletion gap keeps its first start); the `_bot` field
  (schema addition); 4 cards keep their hidden-layer text (they print it as visible layers); the Website edits cap of
  100; per-field credit on 14 entries; Computer Virus left flagged.
- **Older items:** Photoshop-authored fixtures (a bevel texture, a smart object with a painted
  filter mask); the live TTS check (a teammate takes a card back out of another colour's zone; a mod-cached player sees an
  updated card); inline-editing extras (stars, token picker, tag completion, type palette, rulebook text); campaign
  questions (16 simulator Q1-7, 17 game reports Q1/2/5/6, 19 rulebook P0 frame and C/U/M/T/G/S, 22 web editor Q2-9); the
  AI sandbox repo (campaign 18, D1-D3; Q3 budget, Q4 how the bot runs it); S8's ten acceptance edits; repo size (31 GB on
  GitHub; LFS or a TTSCardMaker2 only if GitHub complains).

## Not chosen (don't build unless asked)

Printed-text audit; Discord `/preview`; the website edit-support map; interactive website cards; template passes L2/L3/L5/L6
(stat-row templates, Sonic rows, Ability-layer moves, retyped titles: "give our developers freedom"); point badges (points
stay text); font-side hardening.
