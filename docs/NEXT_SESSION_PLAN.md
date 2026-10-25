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
  the three Actions (TTSCardMaker b7798bb1). Bad px on the frozen corpus: 605,755 (round 6 base) -> 138,578 (-77%).
- **TTSCardMaker** main b7798bb1. Landed this afternoon: glow font 0.3 em (b24c3519), Energy_Ritual cand A (e0775de0),
  Effect Layers.jsx overlay colour (8308103f), PNG recompression (b2168617) and the one batch pass (2330d7b0: 686 PNGs,
  4.18 GB -> 1.14 GB, the same pixels, as AutoCroissant with `Recompressed` trailers), TTS save re-pinned (85d23666),
  types in any order (1f9563e2), fork pins (5b68f6d1, b7798bb1).
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

### 1b. The stat history backfill (branch `work/backfill`; after step 1, before step 3's re-seed)
`history_backfill.json` (repo root) holds every stat-card PSD version in git read by today's parser, the misread rule
and 92 archive entries (26 records to add). Misread rule: a stat changes only where its own row on the PSD's stored
preview changed (more than 300 px past 24); a reading that changed elsewhere is a misread. A stat's row is placed by the
row labels (the "Lines" layers) where there are any, else by layer names, else (no row drawn at all: Grimora, Skeleton
Mate) by the band right of a text label named "Health", "Defense", ...; a pair with no stat region at all is "blind"
(never one print run; the file's `blind` list, 0 today). All-0 or 0/-1 lines (Clot Bone's hearts, Tainted Keeper's first
blob) and two creatures' rows summed (God Tamer) are unread; a hidden or out-of-range stat is unread on its own
(Michael's HP 17 keeps its ATK nerf). Renamed cards are read with the card they became. Three Ech blobs today's parser
misreads are read by hand (`BY_HAND`: SPD 4 not 1 on 2024-10-03, ATK 4 not 0 on 2025-03-22, from their previews). Made
by hand on the Mac (`python -m commands.history_backfill build`, ~6 min from scratch with 4 workers, seconds from the
cache, the fork venv, the full clone `~/Documents/Tools/wt/tc-full.git`, card head efcf17d2; it refuses to build when
Card Authors.txt, Card Edits.json or the font manifest is at head but not in the clone); design and runs in
`~/Documents/Tools/wt/scratch/backfill/` (DESIGN.md §12-14, fix3/). `/update_stats` merges it once, after `recover()` and
after the run's traversal (so renamed records get this run's Card Authors.txt), on a trial copy, and says so in one
`Backfill:` line. It merges nothing and prints `Backfill waits: ...` while a current card is unlike the file (not
re-parsed yet, or bytes the file never read and can't stand for), or while a `PRINTED` card's PSD still reads other
stats than it prints. A same-data re-save after the file's head takes its path's reading at the head. Left alone and
listed: versions newer than the file, unmatched versions (Slugcat's 3 Stars copy, Z!!!!!'s rename-era one), and stored
lines whose own bytes are unread (God Tamer's -1s). Every archive tag gets an `author`; a card archived after the merge
has only the card's own `author`, so the website reads the tag's, else the record's.
- [ ] **Fix two PSDs first** (`PRINTED`; the merge waits until both are fixed AND the file is rebuilt, so the fix is never
      counted as a 2026 change; never fix them after the merge): The Thin Man reads ATK 8 ('Atk Dark' 8 visible) and
      prints 7 since 2025-03-09 ("Thin Man: -1 ATK"); Grimora reads DEF 2 (zero-size marker '2' in 'bars/def') and prints
      4 since 2025-03-11 ("Grimora: +2 DEF"). Preview, layer render and PNG agree (scratch/backfill/fix3/sim/thin_head.png,
      grim_head.png). Thin Man: show 'Atk Dark' 7, hide 8; Grimora: rename the 'def' marker to '4' (the PNGs already
      print it). Any other `unread_edits` PSD fix goes the same way: before the build, never after the merge.
- [ ] Then rebuild the file at the new head (`build --git ~/Documents/Tools/wt/tc-full.git --head origin/main --cache
      ~/Documents/Tools/wt/scratch/backfill/fix3/readings.jsonl --out history_backfill.json`; fetch the new blobs first if
      it says so). Rehearsed on a local, never-pushed commit with both fixes (fix3/sim): the newest reading wins inside each
      print run, so Thin Man's nerf counts on 2025-03-09 and Grimora's buff on 2025-03-11; "PSDs to fix" says none.
- [ ] Owner reads the file's `changes` (79 on 71 cards today; 81 on 73 after the two fixes) and `unread_edits` (39 rows
      on 28 cards: a stat row printed a change no reading shows). The 10 rows the team's #finished-updates confirms carry
      `confirmed: <note>` (Chef Saltbaker x3, Clot Bone x3, Slugcat, God Tamer, The Thin Man, Grimora): real edits, not
      counted; the rest are restyles or noise (PoD04Zebra Bloon's 2025-03-01 ATK row is a stripe restyle: it prints the
      ATK 6 it reads). Also DESIGN.md §11: the qualified keys `Pizza (item)`, `Dogma (item)`, `Auto Croissant (minion)`;
      4 changes within a day of a card's first commit; 27 authors by first commit. **Amy Rose's Hammer is archived as
      renamed to Piko Piko Hammer** (git: a6c81145 `R098 Auxiliary/Items/Amy_Rose's_Hammer.psd -> Items/Equip/3
      Stars/Piko_Hammer.psd`, the same text and stars, then 0c8cdb36 to Piko_Piko_Hammer; Regenerative Shield, the card he
      ruled on, is indeed another card). To keep it Deleted, drop its `SIMILAR` line in history_backfill.py and rebuild.
- [ ] After step 1's PICKLE is on origin/main: copy its pickles to `scratch/backfill/rehearsal/prod/`, run `python -m
      commands.history_backfill rehearse --file history_backfill.json --git ~/Documents/Tools/wt/tc-full.git --pickles
      <prod> --out <out>` (it applies Card Authors.txt at --head as the run does), then `diff_stats.py prod/stats.pkl
      out/stats.pkl --history prod/old_stats.pkl out/old_stats.pkl`. Expected, from the jetson's 2026-10-24 05:07 copies
      with the two fixes emulated (fix3/sim; rehearse_fixed.txt, diff_stats_fixed.txt): stats.pkl 820/820, 0 modified;
      Backfill line: 30 versions corrected, 1 kept (God Tamer), 2 unmatched, 75 stat versions added, 6 folded, 60 starts
      earlier (10 current cards: Acro, Green Mage, Grimora, Markoth, PoD06Rainbow Bloon, Sabre, Tainted Eden, Tainted
      Isaac, The Thin Man, Wham Bam Jewel; diff_stats can't show `version`, rehearse lists them); old_stats 1,162 -> 1,257
      versions, which diff_stats prints as `versions added: 101` (75 stat versions + 26 archive records) and `dropped: 6`
      (the 6 folded; the line counts added and folded as diff_stats does: copies kept, stored versions gone), 671 -> 703
      names; 92 tags changed (26 new, every one given an author; Amy Rose's Hammer, Tainted Esau and Sacrifice retagged
      as renamed; Azamoth the Possessed King and Militron No Armor given the path git deleted); stat changes on current
      cards 68 (28 buffs, 36 nerfs, 4 moved) on 63 cards (the Stats page counts current cards only), 7 uncounted (God
      Tamer and the six 2025-03-22 partial rows). Second cleanup and second merge: nothing; a second `merge()` with the
      marker aside changes nothing and every touched version ends at its next start's commit (fix3/sim/again.py;
      `tests/test_history_backfill.py::test_the_committed_file_merges_once_on_the_real_pickles` with
      `AUTOCROISSANT_HISTORY_CLONE=~/Documents/Tools/wt/tc-full.git`, else it skips). Without the two fixes the merge waits
      and changes nothing.
- [ ] Land (one code commit; no pickles), deploy outside 05:00-05:40, run the diff_stats gate before the next `/push`.
      Undo: revert that PICKLE and delete `history_backfill.json` in the same push. A card repo stat-card commit after
      the file's head is left alone by the merge (a same-data re-save takes the head's reading); rebuild the file first if
      many land. A kill between the run's two pickle writes (old_stats.pkl saved, stats.pkl not: old_stats.pkl newer, no
      run summary) leaves the merge marked done with the current cards' starts unsaved: restore both pickles (`git
      checkout HEAD -- stats.pkl old_stats.pkl` on the jetson, before any `/push`) and run `/update_stats` again.
- [ ] **Dead code after its PICKLE is pushed:** delete `history_backfill.json`, `commands/history_backfill.py`, its call
      in `card_data.update()`, `tests/test_history_backfill.py` and the skill lines that name them.

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
- [ ] **Shadow ends** 2026-10-26 10:36 at the earliest, or after 10 card pushes, whichever is later. **Superseded
      2026-10-24:** the owner switches as soon as everything works; built on work/cutover (both repos, DATA_BUILD_ACTION
      §14 step 8 "As built"), runbook `~/Documents/Tools/wt/scratch/cutover-check-1024/RUNBOOK.md`.
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
- [x] `CARD_GUARDS=report` since 2026-10-23 13:04 PDT. The batch commit 2330d7b0 ran as "0 cards, 686 left out (bot
      commits)", nothing told.
- [x] `CARD_GUARDS=on` since 2026-10-23 17:34 PDT (owner: now). Watch the standards channel and the bot commits; R11
      renames stay told (owner). Left in PUSH_GUARDS.md §15: the other items below (the wait timer, R7 and G2 were decided
      on 2026-10-24: step 4b).

### 4b. Python 3.14 and the owner's guard decisions of 2026-10-24 (branches `work/py314`, both repos; not landed)
- Built: every Action on 3.14 (card-sync one venv; card-check and font-check on the fork; one wheel cache entry; the
  pyoxipng marker takes Linux x86_64 at any Python), R7 left to font-check, G2 rendered anyway ("no stored image to
  cross-check"), the quiet window as the environment `card-guards-quiet` (check / post / finish jobs). PUSH_GUARDS.md §20
  has the design, the Mac's cold-install numbers and the landing order.
- [ ] Owner to confirm (§20.4, defaults of 2026-10-24): a new PSD with no stored image beside a PNG at most 300 px
      past 8 levels of the render keeps the PNG and tells it (`png_export.KEPT`/`KEPT_PX`: the fork's noise; past it
      redrawn: the 34-commit replay's stale edits, 1,684 px and up, land within the bar of the author's export, and
      three current PNGs noisier than 300 px are rewritten needlessly; a stale edit of 300 px or less stays, told); the
      guard-1 line "the new PSD drawn against its PNG: bad N%" is dropped from the post (plan.json keeps the number).
- [ ] Undoing the environment: only with no push run waiting (`gh run list ... --status waiting`); a DELETE fails
      every post job waiting in it (§20).
- [ ] Land TTSCardMaker `work/py314`, then AutoCroissant `work/py314` in the same sitting (jetson deploy per the rule).
- [ ] Then the owner creates the environment (§20's `gh api` command; nothing ran it), once the push runs checked before
      AutoCroissant landed have finished (main's old Land behind the timer would miss a push made during the check).
- [ ] Watch the first runs: a cold wheel build on a runner (pyoxipng's Rust; card-preview's install step has 5 minutes),
      card-sync's one venv, a push's post job waiting in the environment, a dispatch and a re-run taking none.

### 5. Follow-ups still open at the end
- [ ] **BUG 2** (the bake rebuilds `Ability FX:` layers instead of refusing 18-20 gradient badge cards; BUGS D7):
      AC `work/fx-rebuild` (worktree `~/Documents/Tools/wt/bf-fx`). Two review rounds found silent mis-bakes (Goofy's 1#1,
      badges typed as text, status layers spreading), all fixed; q6 decided unattended (rebuilt layers keep their own
      style). Review loop wf_97b6ba0d-096 running at the end: land only after a clean review, merged with origin/main,
      full suite on the fork venv. Action code only (psd_edit, card_bake): pull on the jetson, no rebuild needed.
- [ ] Nebulous_Precipice glow: kept `[darkmatter:sticker]` (unattended option C; options A-D in
      `~/std-work-1022/glow/held.json`). A stroked Text group swallows a `:glow` halo (652 of 765 cards stroke it).
- [ ] b22f4cc4's new PNG against its PSD (the first live bake).

### 6. psd-tools rounds until dry (stop when a round removes < ~2% of what is left)
- [x] **Round 12, the last** (2026-10-24, tag `cards-20261024` = 6a616ca6, pinned in requirements-cards.txt): text
      diagonals, rulebook corners, inner glow, B11, fast open, render hotspots, wide strokes. Frozen corpus
      corpus-b89ce169: 5,160 -> 3,037 bad px (49 cards, 0 worse); all 2,963 bot boxes pass on the fork alone; renders
      ~1.7x and PSD opens ~3.2x faster. B11's trade: Toriel's Ability 0.18% -> 4.19% bad when re-drawn (net gain over
      3,923 layers). No round 13 (RENDER_STATUS.md cards_r12). Next fork work: the `:glow`/`:darkglow` halo (owner
      2026-10-24), then a tag.
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
- [ ] Badge the 11 EarthboundZero cards, Koro-Sensei, The Evil Idol and Papyrus with the fork (owner ruling; unblocked).
- [ ] With BUG 2 landed: FX layers in no chart status's style (Morpho Knight, The Watcher, Computer Virus, Mr Krabs,
      Palace of Shadow, Fire/Ice Staff, Flameblade) could be rebuilt in their own style too (the 2026-10-21 rule refuses
      them): owner.
- [ ] The research waits (NEXT_IDEAS_2026-10-22.md): the open-once patch (`wt/feat-speed-scratch/ac-open-once.patch`),
      now only its bake half (the data-build half is on main since 4dcf33f, guarded by `tests/test_open_once.py`):
      after `work/fx-rebuild` lands, `psd_edit.parse_card` takes an open PSD, `gate_reparse` gets bake's `reopened`
      and card_bake opens the old PSD once (twice per call now; check `classify` leaves the PSD alone); rcache wiring
      in psd_edit, text geometry in the data build and a website overlay, template fill stage 2, the Photoshop oracle
      loop (approved; Photoshop is free), the Perpetua lead (Hollow Knight).
- [ ] Extraction campaign: the structural candidate (blank-but-present Ability = no problem: ABILITY_EXCLUSIONS 21 -> 0),
      the gap-type validator, idea 14 (orphaned-FX validator), S9 (star moves move PSD + PNG), fix-on-push for 0-pixel
      classes, Task 3c (effect layers on pushes), L11 rest, L15 class D renderer.
- [ ] Live checks never done: V1 `/sync_emojis` refresh, V2 a reminder firing watched, V4 live queries, V6 (Papyrus and
      Sans after step 2).

### 8. Website, TTS and Discord follow-ups
- [ ] Refusal wording at the source: psd_edit `plan_stars` / `plan_types` refusals carry internal words ("the '5 Stars'
      folder sets them (moving the card is ledger S9)", layer boxes such as "at (12, 40, 60, 88)"). The website maps them
      to plain words (preview.js `PLAIN_WHY`, 2026-10-24); reword them in psd_edit after BUG 2 lands, and keep
      `PLAIN_WHY` in step. psd_parser also reads `[fire:]` as `[fire]`, though the font prints it as letters.
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
  draws) is set in the font by the bake; icon swaps carry the old icon's effects; Green Mage's PSD fixed (TTSCardMaker 45cf8746: ATK 6, close BUGS D8 after step 1); all stale
  exports get re-exported; `CARD_GUARDS=on` now (17:34); R11 renames keep telling; the fork rounds stop after 11.
- **Cutover:** the reminder swap (step 3) and telling bobberlington before step 10.
- **Push guards** (PUSH_GUARDS.md §15): the fork render's refusals; Photoshop's image for a PNG pushed before its PSD; the
  "outside R" parity method; the login fallback and the commit link in the post. (Decided 2026-10-24, §20: the wait timer,
  R7 left to font-check, G2 rendered anyway.)
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
