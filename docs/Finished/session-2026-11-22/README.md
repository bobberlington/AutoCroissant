# Session 2026-11-22: research rounds, the deferred five, two fork releases

From 11-21 23:40 (AC ce4bae3a = add5194d + docs, jetson add5194d; TC 5f22cef0, data ea50731e, psd_details VERSION
13; fork 977e4067, pin `cards-20261121b`) to the morning of 11-23. The owner answered questions in six bursts plus three
single rulings (below); the rest ran unattended.

Heads at the end:

| What | Head |
|---|---|
| AC main | 541feb9c (114 commits since ce4bae3a, all ours) + the wrap-up docs (`work/wrap-1122`) |
| TC main | 93e5892f (73 commits: 63 ours, 10 Card sync) |
| TC data | 9fc74401 (psd_details VERSION 15, 769 current records; card_checks 0 hand, 33 bars, 0 unredrawable); plates 01e56f95 |
| Fork | main bb8c62fc (47 commits; CI 2 jobs since 07:17); tag `cards-20261123` = baa9a901, pinned since AC 541feb9c; shared venv @baa9a901 |
| Jetson | 5553f8da (42 commands, 777 cards); nothing the bot loads changed after it |
| CardBrowser | main 4c1bd38, v0.1.2 (quiet pre-release) |
| AC suite | 1294 passed at the start, 1350 at AC 541feb9c + TC daca7927 |

## Landed

| What | When | Heads | Notes |
|---|---|---|---|
| Deferred: x31, r52-1, Papyrus mock-up, retire game reports (docs) | 11-22 04:40 | AC 6db56432, TC f65bd245 | Typed letters keep the card's font; titles shrink to clear the stars and type icons; pictures over a text layer stay over new words. VERSION 13 -> 14 re-read (data bf790043); Papyrus covers on 5 cards (plates 385745c2) |
| Deferred: x37 | 05:24 | TC 34b497f8, AC 37908037 | Preview tab hand-over: the tab in front checks; a starting press counts for it |
| Deferred: r44-2 | 06:40 | TC 2c72eff1, AC 3f6dfe89 | Stats: a chart's table drawn on its first Table press |
| cards s1 (r1-4, 12 items) | 13:24 | AC 8523b665, TC 1f230bea | Deployed |
| actbake s1 (r1-4, 8 items) | 15:26 | AC f279dbe1, TC cbd4bb1d | VERSION 14 -> 15 (Strokes) re-read, data 095061c3; plates 01e56f95. Guards: post and Follow up now in the check job |
| Owner: Beholster series, Smoldering | 15:56 | TC 4f642447, 2be56ea7 | Beholster in Enter the Gungeon (Card Edits); rulebook Status effects: 7 'Smouldering' -> Smoldering. Data 1f26262a, PDF reprinted |
| Fork release `cards-20261122` | 20:42 | fork e5915747, AC 5e489de4 | Owner G8 dead code + 8 linked-picture rounds. Linked pictures that can't be redrawn: 46 on 40 cards -> 1 (Burly Bomb). Fork Tests 19/19 (45110969929); Card checks 45111548362 (data 85d6f44a); Photoshop test 133 copies on 75 cards, 0 errors; shared venv re-pinned |
| actbake s2 (r5-8, 9 items) | 22:46 | AC e28a8944, TC 9f808c28 | Shipped by ship2-1122 (see Incidents) |
| cards s2 (r5-8, 11 items) | 23:50 | AC 43ebb2c0, TC 6903dba6 | Deployed. r6-3 part 2 backed out (owner G3 keeps PSD_DIRTY_VERIFY) |
| cards s3 (w0, w1, x0-x7, 10 items) | 11-23 00:36 | AC 5553f8da, TC f24c52e0 (save 7858e49d) | Deployed. x2's data refresh (dry run first): old_stats only, 8 values on Retro Kirby, Living Bread, Shell Shield (data ff441195) |
| actbake s3 (x2) | 01:12 | AC fdc718af, TC 2838c3e8 | Typed words at a paragraph's end take its line's colour (52 paragraphs, 42 cards) |
| Owner: fork docs (answer 21) | 01:13 | fork 64db7098 | README, usage, migration: `composite` is required. Tests 19/19 |
| Owner: 9 rulebook typos (answer 18) | 01:30 | TC 876f5b70 (PDF c01f201c, save 25077c28) | 7 chapters; headings WHEN ABILITIES CAN BE USED and BASIC ABILITIES. Data c522d262 |
| Owner: Reduce Noise, `cards-20261123` (answer 20) | 04:01 | fork baa9a901, AC 541feb9c | Burly Bomb's Reduce Noise redrawn: unredrawable 1 -> 0 (Card checks 45169866141, data b954baaf). Tests 19/19 |
| Owner: two more typos (answer 23) | 06:20 | TC daca7927 (PDF 5c6dc92f, save 93e5892f) | 'abbrviations' x2, 'reactivley'. Data 9fc74401 |
| Owner: fork docs 2 (answer 22) | 06:51 | fork 1b970405 | index, architecture, composite docstrings. Tests red: two macOS jobs never got a runner (Incidents) |
| Owner: fork CI, 2 jobs (answer 25) | 07:17 | fork bb8c62fc | test.yml: Linux + macOS on Python 3.14 (tests, linter, formatting, type checks); Windows, 3.11-3.13, 3.15/3.15t, the 3 no-composite jobs and the summary job dropped; CLAUDE.md, contributing.rst. Tests 45203571689 2/2 (macOS got a runner on the second try) |
| CardBrowser v0.1.2 | 11-22 01:05 | CardBrowser 4c1bd38 | A maker's Windows launch failed ('Python.Runtime.Loader.Initialize'). Fix: exe .config loadFromRemoteSources + an error box with the text; CI tests a downloaded-marked copy. README 120 -> 40 lines (owner: less is more). Quiet (owner 3) |

Other results:
- **Photoshop PNG exports (since 11-16): nothing was wrong.** JSX `saveAs(PNGSaveOptions)` always wrote the
  document's raw values, untagged. Cards whose PSD has the 'Display' profile have repo PNGs converted to sRGB, so a
  raw export is up to 55 levels off. On 11-16 the land agent compared without converting. Converted
  (`scratch/s1122/tasks/jsx-png/card_png.jsxinc` or `py/psconv.py`): 0 px. No Photoshop setting changed. This
  corrects the 11-16 incident in the last report.
- **11-08 commits without a Card-sweep trailer:** all five posted "New art" (39 cards, 5 messages); the four
  Single-line batches too (321 cards). Nothing to undo.
- **Desktop TC repack** (owner 6, 11-23 04:15-05:35, no agent running): 1189 packs + 9756 loose objects (29.3 GiB)
  -> 2 packs, 0 loose (29.1 GiB); 43,860 objects, 0 missing; promisor flag kept; fsck ok. `git repack -a -d -k
  --window=0` (the default delta search ran 40 min without writing and was stopped).
- **TTS Saves folder** (11-23 01:40): the old test save deleted; main's save (25077c28) copied in.
- **Correction:** the plan's "71 cards gain faces" was a sample size; after the VERSION 14 re-read 29 current records
  carry faces.

## Jetson deploys

| Head | When | Why |
|---|---|---|
| 8523b665 | 11-22 13:24 | cards s1 (query code) |
| 43ebb2c0 | 11-22 23:48 | cards s2 (main.py, help.py, query_card.py) |
| 5553f8da | 11-23 ~00:30 | cards s3 (query_card.py, help.py, card_history.py) |

No deploy for the deferred five, actbake s1-s3, the two fork re-pins or the docs (no file the bot loads; the jetson
has no fork). Always 42 commands, 777 cards.

## Owner answers (memory `owner-decisions-2026-11-22`)

| # | When | Answer |
|---|---|---|
| 1 | 11-21 23:58 | x31: land as is (NiseSegaSonic's curly quote prints big; accepted) |
| 2 | | Load: always low (pytest -n 3, low priority, game or not) |
| 3 | | CardBrowser v0.1.1 works; keep it quiet (no link, no announcement) |
| 4 | | Photoshop export: if a setting changed, change it back (none had) |
| 5 | 11-22 00:00 | /query keeps finding MDW/ and Types/ PNGs by exact name |
| 6 | | Desktop TC clone: careful promisor repack at the very end (done 11-23 05:35) |
| 7 | | D24 waits for a real font build |
| 8 | | Desktop TC clone's stray data/: delete (done 00:02) |
| 9 | 04:05 | While gaming: `taskpolicy -c utility nice -n 15`, never `-b` |
| 10 | 04:10 | Work unattended, no questions |
| 11 | 15:25 | Full plan, no cut (ETA ~11-23 15:00) |
| 12 | | Beholster: series Enter the Gungeon (its copied Ability stays for Croissant) |
| 13 | | Rulebook Status effects: Smoldering |
| 14 | 18:35 | status-panda.webp (old 'Smouldering' card picture): leave it |
| 15 | | Fork CI: keep the 19-job matrix (supersedes G8 fork-r1-9; superseded by 25) |
| 16 | | Fork release: land when ready |
| 17 | | Drop the release's per-round changelog entries |
| 18 | 11-23 00:40 | Fix all 9 rulebook typos (TC 876f5b70) |
| 19 | | Next session: research again, ~8 per lane |
| 20 | | Teach the fork Burly Bomb's Reduce Noise this session (`cards-20261123`: unredrawable 0) |
| 21 | | Fix the fork's README, usage and migration pages: composite is required (fork 64db7098) |
| 22 | 04:20 | Also index, architecture and the docstrings (fork 1b970405; docs only, no release) |
| 23 | | Also 'abbrviations' x2 and 'reactivley' (TC daca7927) |
| 24 | | Delete the s1109 `tc-unlanded-1110.bundle` (done; its heads in `wrap/branch_record.tsv.tc-unlanded-heads.txt`) |
| 25 | 07:05 | Fork CI: Linux + macOS on Python 3.14, 2 jobs (supersedes 15; fork bb8c62fc) |

Also 00:55: CardBrowser docs as short as possible.

## Research

Lanes: actbake (Actions, bake, previews, PNG export, mock-up), cards (parser, card data, TTS, website, bot, tests),
fork (psd_tools internals: linked pictures that can't be redrawn). Rules: a person must notice the gain (owner 11-16);
measure first; no record proposals. Full lists: accepted.md, rejected.md.

| Lane | Rounds | Finder runs | New ideas | Built | Kept | Rejected | Waiting built | Retries (judged / rebuilt / kept) | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| actbake | 8 (r2 found nothing) | 56 | 21 + 1 from the plan | 21 | 18 (17 merged: r5-2 repeated r5-0) | 3, + 1 already on main | 0 | 3 / 3 / 1 | 18 |
| cards | 8 | 48 | 31 + 3 from the plan | 32 | 23 | 9 | 2 of 2 | 9 / 8 / 8 (1 dropped: owner pick) | 33 |
| fork | 8 | 8 | 13 | 13 | 11 | 2 (both redone and kept) | 0 | - | 11 + dead code |

- Neither research lane went dry; both stopped at the 8-round cap. Everything kept landed (s3 on 11-23).
- Fork: owner answer 20 added Reduce Noise after the release (`cards-20261123`): no linked picture is left that Card
  sync can't redraw.
- Cardfix judges (2 passes): 0 card fixes. The rulebook typos they found were fixed on owner answers 18 and 23.

## Incidents and lessons

| What | Effect | Now |
|---|---|---|
| research5 bookkeeping bug (found 14:40 by the actbake s1 skeptic) | Its takeMerged matched merge agents' descriptive branch strings exactly, so merged ideas never counted. Its accepted lists were 1 item per lane; actbake s1's prep named only r4-3; s2 and s3 were skipped as "nothing to ship". Same bug left the fork lane's list empty | Not fixed in flight (a multi-lane resume reruns hours). ship2-1122 shipped r5-8 from snapshot branches; ship3s shipped the finish phase. Count from git (accepted.tsv, `git cherry`), never the script's list |
| `taskpolicy -b` while gaming | Pinned our work to busy E-cores: a full suite took 2:05 h with 3 false timeout reds (29 tests 711 s vs 22 s) | Owner 04:05: utility clamp |
| Finders ran up to 3.5 h | One ran 2 full suites and blocked ships behind suite-wait | 03:00: a finder returns in ~60 min, no full suite |
| before.json in the Desktop AC tree (~01:53) | A fork-lane script chdir'd into the AC path and wrote a relative output | Moved 05:10. Absolute output paths; never the Desktop tree as working folder |
| Orphaned tail | `tail -f` + grep shim from the fork r5 verify, parent gone 13.5 h | Killed by PID 23:55. Sweep for orphans after each wave |
| Fork release: three skeptic refusals | (1) no Card checks run after the re-pin, G8 CI cut not done, no release-level Photoshop test. (2) the push plan's revert could undo other lanes' AC work. (3) fix 1 had cut CI to one job, against the owner's 18:35 "keep the 19-job matrix" | The orchestrator's 18:35 question wrongly said the release kept 19 jobs; fix 1 had already cut them. fork-land-1122 restored and proved the 19 jobs, then landed (11-23: 2 jobs, answer 25). Read a branch's head before asking about it |
| cards s2 skeptic refusal | r6-3 part 2 quietly unset PSD_DIRTY_VERIFY in one test (owner G3 keep); the push plan moved the lane branch | Backed out as a revert on top (43ebb2c0) so s3 skips it; ships use snapshots |
| AC main ahead of the shared venv | 20:23-20:42, AC main needed the new fork before the venv was re-pinned | Lanes rebase onto main after a re-pin (BRIEF 17:20 rule) |
| Fork Tests on 1b970405 (11-23) | macOS 3.12 and 3.13 never got a GitHub runner in 4 attempts; every job that ran passed; Test Summary red | Docs only: nothing reverted, not rerun: the 2-job CI (answer 25) passed on bb8c62fc, which holds it (45203571689) |
| Shell slips (owner5, ship3-actbake) | A tag reached the fork a minute before its main (the shell ate the refspec; same SHA); relative paths made two empty worktrees inside tc-1122; a zsh `path` loop variable overwrote PATH | Nothing wrong left. Multi-step shell as a bash script with absolute paths |

## Conservative choices (unattended)

- actbake s1 shipped all 8 merged items, not the 1 the buggy list named (prep and skeptic agreed).
- research5 was left running with the bug; extra ship workflows shipped from snapshots, never moving `work/<lane>`.
- Photoshop: no setting changed (none had); shared Photoshop scripts not edited; the helper stays in scratch.
- Cardfix: nothing pushed. Beholster's copied Ability left for Croissant.
- cards r6-3 part 2 backed out (owner G3). cards x8 (PSD_DIRTY_VERIFY) dropped by both judges, not asked.
- cards x3 shipped only the parser half (the editor half disagreed with what the bot restores).
- cards x2 corrects only Retro Kirby, Living Bread and Shell Shield; Hero Cap, Slugcat and Massive Moss Charger keep
  their rows (their timelines' first record holds the same misread). Its data refresh: a dry Card sync first, then one
  live run with bake none.
- cards r7-3: main's ship copy 77582a93 kept; the original 1eab1edd dropped in s3.
- actbake r1-3 not built (already on main, TC 8ae326d1); r5-2 dropped at merge (same fix as r5-0).
- actbake s3's land skipped a second headless Chrome check after its rebase (main hadn't touched mockup.js or
  textcolor.js; the full suite with the site tests passed at the final heads).
- Typos: only the listed spellings. The two renamed headings' old links open the rulebook at its top (nothing linked
  them; keeping old links would be new code).
- Fork docs: only the pages each answer named; `PSDImage.save()`'s fallback when `composite` is missing left as is
  (code). Fork Tests red for runners only: no revert, no re-push.
- Fork: `use_limited_api` kept on 3.13+ (G8 fork-r9-4 deviation; build identical); lens branches r3-0, r4-0 bundled
  (`scratch/s1122/bundles/fork-1122-lens-{fork,ac}.bundle`); all 7 per-round changelog entries dropped.

## For the owner (nothing pushed)

- Beholster prints The Puppet's Ability word for word: tell Croissant.
- Fork: `PSDImage.save()` still keeps the old preview with a warning when `composite` is missing (answer 22 covered
  docs only). Make it an error? (plan, Open owner decisions)

## Watch items

- Card guards' new job layout (actbake r4-0, TC cbd4bb1d): proven only by a dry dispatch. On the first maker push
  with PSD/PNG changes run `research5/actbake-1122-s1-prep/land/first_push_check.sh`; watch the 60-min check-job limit.
- Flaky under load: `test_card_guards::test_the_checks_reports_come_once_that_job_ends` (reads checks.result before it
  is written; fix: wait for non-empty) and `test_the_bot_clone_and_records_run_here` (15 s clone wait).
- Rulebook search: 'types of' also finds 15. Minions (w0); a card's name typed there draws an unrelated "Did you
  mean" (Frog -> 'from'; x4); 'alot' suggests 'slot'; /query_rulebook doesn't defer, so a very long miss could pass 3 s.
- Text colours where the bake and the mock-up still differ at a boundary (on main before s3): Chris & Jordi ('DEF|
  lost'), Pix ('Marked For Life| until', also wrapped differently), Grimmchild ('+2 ATK| to parent').
- x3: a new card in a folder whose cards share one series takes it even from a related game (1 of 302 in a
  leave-one-out test: Godskin Noble would get Dark Souls III); the editor's reset label shows the folder name
  lowercased.
- The guards read requirements-cards.txt at ref main apart from the bot clone (a race only at a pin change).
- status-panda.webp still shows the old 'Smouldering' card (owner: leave it).

## Numbers

- Agents: research5-1122 299, tasks-1122 63, owner5-1122 11, ship2-1122 8, wrapA-1122 4, fork-land-1122 3,
  ship3-cards 3, ship3-actbake 3, one-offs 5 (CardBrowser fix + skeptic, owner3-1122, the two wrap-up drafts);
  wrapB-1122 still running at this writing. About 66.6 M tokens in the eight finished workflows (research5 49 M).
- 24 lane-rounds (16 research, 8 fork); 66 research builds and 13 fork builds, each checked by an independent
  verifier; 12 retries judged, 11 rebuilt, 9 kept.
- Accepted and landed: actbake 18, cards 33, fork 11 (+ dead code, + Reduce Noise).

## Files here

- `README.md`: this report. `accepted.md`: every accepted idea, by lane, with its ship. `rejected.md`: every rejected,
  dropped or not-kept idea with its reason, and the retries. `branches.md`: where every ref and bundle went.
- Sources (scratch `~/Documents/Tools/wt/scratch/s1122/`): `BRIEF.md` (owner words, LANDED lines),
  `orchestrator-notes.md`, `results/*.json`, `research5/<lane>-accepted.tsv`, `ship2|ship3/<lane>/ship_record.tsv`.

What is left: `docs/NEXT_SESSION_PLAN.md`.
