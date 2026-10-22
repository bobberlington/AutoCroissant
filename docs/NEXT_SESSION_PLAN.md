# Next session plan (2026-10-22, Opus 5.5)

Start in `~/Desktop/AutoCroissant`. Read memories `next-session-plan`, `jetson-server`, `ttscardmaker-integration`,
`agent-model-preference` and `psd-edits-hold` first. Agents run on Opus or lower (never above the session's model).
The 2026-10-21 session report is `docs/Finished/session-2026-10-21/README.md`; the full pending-task inventory it was
checked against is reproduced below.

**State at the end of 2026-10-21**
- Jetson runs main 1048324 (52 slash commands; parser version 2; psd_details.json; the fork is NOT installed there, stock psd-tools
  1.25). Its stats.pkl from the version-2 re-parse (819 entries, all stamped) is UNPUSHED until the user's /push or the 05:30 reminder;
  until then the website shows the old titles.
- TTSCardMaker main a3451c8: website with profiles, People, inline editing, Details dropdown, published decks; Actions site.yml,
  tts-save.yml, card-edits.yml (dispatch only, pinned to fork tag `cards-20261020`), font-check.yml and card-check.yml (push guard
  rules 1-2, report only).
- Fork: `cards` a6363a04 = tag `cards-20261020` (pushed). Local, unpushed: `review/round2-consolidated` 31e1a294 (/tmp/wt-review-r2),
  the nine round-3 branches (fix/clipped-overlay-group, fix/drop-shadow-spread, fix/inside-stroke-offcanvas, fix/outer-glow-coverage,
  fix/pattern-overlay-origin, fix/text-stroke-perc, fix/vibrance-fit, perf/save-with-preview, perf/viewport-effects; worktrees
  /tmp/wt-*), the round-4 scratch branches r4/*-scratch (/tmp/wt-r4-bevel, -gradshade, -blur). ROUND 3 WAS NOT INTEGRATED: its
  workflow was stopped at 17:50 before the integrate/critic stages (a model switch invalidated its build cache; do NOT resume
  wf_059e8902-7ff, it would rebuild the nine branches). State of the nine branches from the journal
  (`~/.claude/projects/-Users-michaelsrouji-Desktop-AutoCroissant/5c6ede3c-b4cb-4791-926e-2af4b5a959fa/subagents/workflows/wf_059e8902-7ff/journal.jsonl`):
  seven passed both reviewers after their fixes (fix/clipped-overlay-group 3e62a1d5, fix/drop-shadow-spread fbbaf6a6,
  fix/pattern-overlay-origin 027a7518, fix/inside-stroke-offcanvas b52a1428, fix/text-stroke-perc 17d471fe, fix/vibrance-fit 4e752142,
  perf/viewport-effects d3c36869); two carry a second fix that was NOT re-reviewed (the loop's cap): fix/outer-glow-coverage 464883df
  and perf/save-with-preview 6f91981e (its open findings were CMYK/Lab alpha on save(preview=) and has_composite on 3-channel
  documents, both addressed by the last commit, unverified). The bot-side half of save(preview=) that an agent drafted uncommitted in
  commands/psd_edit.py was saved as `~/Documents/Tools/psd-research/rescued-2026-10-21/psd_edit_save_preview_draft.patch` and the
  working tree restored; apply it only with the new pin tag installed and the tests green.
- PSD hold still on: no PSD/PNG write in TTSCardMaker; dry runs on /tmp copies only.
- /tmp holds things the tests and the fork work need (29 worktrees, /private/tmp/fontexplore JSX trial copies and std.jsx,
  /tmp/r4* diagnostics, the website node harnesses): a reboot loses them; item 4 below rescues what matters.

## Session checklist (2026-10-22, unattended unless a line says [user])

In order. A line is struck when it lands; the docs it closes move to `docs/Finished/`.

1. [ ] **Fork integration** (a fresh Opus workflow, not a resume). Re-review fix/outer-glow-coverage 464883df and
   perf/save-with-preview 6f91981e (two lenses each, as round 3 did); then merge the nine round-3 branches into a `cards_v4`
   worktree from a6363a04, run the 813-card harness vs cards_v3 and a critic with the stat-toggle check (what the stopped
   integrate/critic stages would have done). Verified 2026-10-21 evening with
   `git merge-base --is-ancestor`: the user's seven round-1 review commits (5d92a0e7, ddf64e52, fc75012a, 2c691bc2, 409aa349, 2db839da,
   c294d094) ARE in `cards` a6363a04; the six round-2 ones (1d363b1d, c2797ede, c48ee268, ad07098e, 3d8b4f22, d33e0bda) are NOT: they
   live on their fix/* branches and, hand-ported, in `review/round2-consolidated`. Merge `review/round2-consolidated`
   (31e1a294) and the round-3 branches into `cards` (conflicts expected in effects.py / paint.py / composite.py's import list);
   suite (ruff, mypy, pytest ~5300); full 813-card harness run vs `cards_v3` (`score_cards.py <name> ` then `compare_scores.py
   cards_v3 <name>`; NEVER `--help`); push `cards`, the nine branches and the consolidated branch; tag `cards-2026102x`; re-pin
   TTSCardMaker `.github/workflows/card-edits.yml` to the tag; install the tag in the jetson venv once `tests/test_parser_goldens.py`
   passes under it (`~/venvs/autocroissant/bin/pip install "psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@<tag>"`,
   then a restart off-hours, never during /update_stats); switch `card_bake` to `save(preview=)` (perf/save-with-preview). Refresh
   `~/Documents/Tools/psd-research/RENDER_STATUS.md` (cards_v4 section; drop the stale "Nothing pushed" lines; rewrite the class B
   text per RENDER_ROUND4_PLAN.md) and `PR_BODIES.md` (13 bodies: the user's review commits, the GradientInterpolation ->
   GradientMethod rename, round-3 bodies, the user's note that the round-2 branches stay as they are). Prune the /tmp worktrees of
   merged branches afterwards (keep wt-cards, wt-review-r2 until pushed, r4).
2. [ ] **Rendering round 4 implementation** (Opus agents; `RENDER_ROUND4_PLAN.md` is the brief): baseline `r4_base` on the integrated
   `cards`; `fix/smart-filter-mask`, `fix/composite-byte-rounding`, `fix/bevel-texture` in that order from the scratch worktrees;
   each reviewed and measured on its list; merge; `r4_integrated` vs `r4_base`; re-check the overlap table (bevel x
   perf/viewport-effects import line; fix/clipped-overlay-group's Difference satins on the Cannon's name). Bot follow-ups: rewrite
   `tests/test_psd_edit.py::test_cannon_rows_refuse_parity_at_the_baseline` (atk passes, hp refuses on the mean) and flip
   `test_moved_effect_paints_the_new_digit`; copy nothing more from /tmp (the bot-metric scripts are in psd-research/score/round4/).
   [user] Parity question: lit cells above mean 1.0 at 0% bad (Cannon 1.579, Glaive_Main_Cannon 1.649) and the 28 raw-export cards
   refused by ICC in `render_region`. [user] Photoshop-authored fixtures (a bevel texture, a smart object with a painted filter mask).
3. [ ] **Jetson and data.** After the user's /push: V4 live checks (`/query Raw Kawk`, `/query Black Clot`), the site shows the
   printed titles. Run `/fonts` once (fills the `supplied` flags of jetson-gathered psd_details records; B11). Design B11 (a fonts
   cache keyed by the MDW/Fonts tree sha for the bake and the details gather). Record every deploy in memory `jetson-server`.
4. [ ] **Rescue from /tmp and make the tests self-sufficient.** (Done 2026-10-21 evening: the scripts under /private/tmp/fontexplore
   (std/std.jsx, fx/*.jsx + *.py, scripts/*.py incl. tts_survey1-5, asl, spec, scale, regress) are copied to
   `~/Documents/Tools/psd-research/rescued-2026-10-21/fontexplore/` (108 files, 712 KB); the JSX trial-copy PSDs and the worktrees
   were not copied.) `tests/test_psd_edit.py` and `tests/test_card_bake.py` cite
   /tmp/wt-pipeline-dev and /tmp/wt-cards and the JSX trial copies under /private/tmp/fontexplore; make them find the fork through
   an env var (default `~/Documents/Tools/psd-tools`) and vendor or regenerate the small fixtures they need; copy `std.jsx` and the
   tts_survey scripts into `~/Documents/Tools/psd-research/` (MDW/Scripts/ only when the hold lifts); install pyyaml in the test venv
   so the two workflow-shape tests run; make `tests/test_card_bake.py` collect under the fork venv (requests).
5. [ ] **Allowed under the hold, not started:** the standardization rehearsal on /tmp copies of the ~110 PSDs with the three gates
   and the expected per-card diff list (CARD_STANDARDIZATION §4.1); the L2 template-row script and the L5/L6 rename/move scripts
   on the fork (layer-move API to confirm); D7 Twin Emperors second creature's DEF on the website (per-creature edit, then tick
   problems checked); D3 Shadow Duelist out of ABILITY_EXCLUSIONS after a golden-checked re-parse; idea 14's orphaned-FX validator
   check (does EFFECT LAYER OUT OF DATE cover it?); the companion skill §5 L1 questions (`[fire]/ [machine]` spacing, token order).
6. [ ] **Bot small items:** B8 (a deleted PSD stays in stats.pkl: prune when the tree lost it, not only when num_changed > 0);
   B9 (a blob 5xx must not record the tree ETag); GIT_TERMINAL_PROMPT=0 / SSH BatchMode on git calls; lazy GitPython in update_bot;
   a requests.Session per download thread; on_ready re-running inits after a failed RESUME; [user] admin gates on /purge,
   /update_metadata, /mass_replace, /alias, /set_repo, /delete_song, /delete_all_music; the parser skipping hidden text layers on
   rule pages (after the page fixes); OJWikibot unit copies still saying python3.10 (cosmetic).
7. [ ] **Website follow-ups** when asked (site/NOTES.md §7-12 "not done" lists): profile avatars and GitHub link, per-person chart,
   bake credit to the requesting editor, the deck-builder picker inside the profile editor, Details for rulebook pages and a
   "has details" filter, deck search from Cards and "decks using this card", the accent-colour cache (-2.9 s builds), the
   "PSD pending" hole for a baked name (EDIT_PIPELINE §9), the pickle-unpickling risk note (BUGS 1.6). Inline editing extras
   (stars, token picker, tag completion, type palette, rulebook text; more drag facets) stay proposed, not agreed [user].
8. [ ] **Docs, skills, memory upkeep:** quote the `word: word` frontmatter in 8 skills; refresh impossibility-cards-reference §4-5
   after parser changes; retire research-frontier Item 3 / extraction Option 1 residue or schedule them; keep this checklist and
   the memory current after every landing; finished docs to `docs/Finished/` with a dated README.

Waiting on the user (not blocking): lifting the PSD hold (then S5 live runs, S8's ten acceptance edits, the standardization batch
L1-L15 incl. the six reprints L8, Sonic L3, star moves S9, the rulebook pages L11 with `tornado` -> `wind`, God Tamer and Bubble's
star groups L14, Computer Virus's stat content D1; the six case-typo titles asked card by card first; the team re-exports the eight
stale PNGs (Skullord, Restrained, Blood_Bomb, Dark_Mind_Eye, Asteroid_Field, PoD04Zebra_Bloon, The_Collector, Holy_Blade); the two
PNG-only cards Stubborn_Queen and Gust_of_Trees need PSDs from their author); the user's review of the round-3 branches and of
`cards`; opening the 13 upstream PRs from `PR_BODIES.md`; BabaUI (Baba.psd); the team's timing-system update (then the stoogedings
badges and the rules skill, then idea 15 text-first rulebook); the AI sandbox repo (campaign 18, D1-D3 to revise); the campaigns'
open questions (16 simulator Q1-7, 17 game reports Q1/2/5/6, 19 rulebook P0 frame, 22 web editor Q2-9); pushing
`archive/local-update-stats`; a Contents: write token for the bot on the card repo; confirming a TTS player with mod caching sees a
refreshed card (Task 2 step 7).

### Pending-task inventory (2026-10-21 evening, verified against every doc and memory file)

The checklist above is the order; this is the complete list it was drawn from, grouped by area, each item with its source.
Items marked [user] need the user; [blocked: X] wait for X.

#### Pending tasks

##### B1. Fork and rendering
- Round 3 workflow wf_059e8902-7ff (task wki5k4vvc), nine branches, integrates as cards_v4 [in progress] (PLAN checklist 5; M:next-session-plan)
- Round 3 branches: fix/clipped-overlay-group, fix/drop-shadow-spread, fix/inside-stroke-offcanvas, fix/outer-glow-coverage, fix/pattern-overlay-origin, fix/text-stroke-perc, fix/vibrance-fit, perf/save-with-preview, perf/viewport-effects; none pushed [in progress] (GIT)
- Merge review/round2-consolidated 31e1a294 into cards with round 3 (effects.py conflicts likely) [blocked: round 3] (M:next-session-plan; R4 §5)
- Then harness vs cards_v3, push branches + cards, cut a new pin tag [blocked: round 3] (M:next-session-plan)
- Re-pin WF/card-edits.yml from cards-20261020 to the new tag [blocked: new tag] (M:next-session-plan; WF/card-edits.yml)
- Install the new tag in the jetson venv ~/venvs/autocroissant once tests/test_parser_goldens.py passes under it [blocked: new tag] (PIPE §8; M:website-and-pipeline-ideas)
- RS: add cards_v4 (round 3) section; fix stale "Nothing pushed" lines [blocked: round 3] (RS cards_v3; M:next-session-plan)
- PRB: refresh for the user's 13 review commits, GradientInterpolation -> GradientMethod rename, round 3 bodies; add the note the user asked for [blocked: round 3] (PRB; M:next-session-plan; M:website-and-pipeline-ideas)
- Round 4 step 1-2: after round 3 integrates, full baseline r4_base on the integrated cards [blocked: round 3] (R4 §3)
- Round 4 fix/smart-filter-mask (class B, 13 active painted-mask cards; >2% 21 -> 11 projected); prototype r4/blur-scratch f64f9cbe [blocked: r4_base] (R4 §2.1, §3)
- Round 4 fix/composite-byte-rounding (Mettaton EX pixel-bar shade; hp toggle mean 1.033 -> 0.556); prototype r4/gradshade-scratch 9109fc3c [blocked: r4_base] (R4 §2.2)
- Round 4 fix/bevel-texture (Cannon digits; atk/spd pass, hp/def still refuse); prototype r4/bevel-scratch 72716234; needs fix/pattern-overlay-origin merged first [blocked: r4_base] (R4 §2.3)
- Round 4 merge order 1-2-3, r4_integrated run, re-check the §5 overlap table (bevel vs perf/viewport-effects import, pattern origin, consolidated ff6f03cd) [blocked: round 4 branches] (R4 §3, §5)
- RS round 4 section must rewrite the wrong "class B unfixable" text [blocked: round 3 releasing RS] (R4 §3 step 5)
- Harness note: score_cards.py has no --help (a --help call starts a run named "--help") [open] (R4 §6)
- Round 4 parked: Cannon name region (Satin), stat_labels PIN_LIGHT, stars residual; Mettaton.psd co-located stops; Jevil spd Perc bias; PoD02Black_Bloon; Hive_Knight glow; textured outer bevel [open] (R4 §4)
- Photoshop-authored fixtures needed: bevel-texture.psd, smart object with blur + half-black mask [user] (R4 §4)
- RS follow-up fixture: group stroke over a child blurred drop shadow (Keelhaul Key, Dark_Sword) [open] (RS cards_v3 "Review follow-up")
- RS open: Pure_Vessel stroked type layer in a styled group (text gamma); Little_Guy stat_labels straight composite [open] (RS cards_v2 Regressions)
- RS open classes after round 2: large drop shadows/glows, type layers with own stroke (Skullord 3.0, Marx, Jevil...), adj:gradientmap, class C Perc fills, class D, class G (Shadow_Miner, Heavy_Mole) [open] (RS "What is still missing")
- Text engine limits: warp unwarped, platform AA as Strong, colour emoji silhouettes [open] (WEB §3.4)
- Fork template-fill API: add a face to FontSet, move/rename layers, lfx2 copy, replace smart-object content [open] (WEB §3.4; STD L1, L5)
- Task 3 done-when: no card > 2% (21 today) and S8's 10 edits identical in Photoshop; then retire Photopea to an archive branch [blocked: round 4 + PSD hold] (PLAN Task 3 step 6; PIPE §1, §13 S8)
- Upstream PRs: 13 bodies drafted (6 round-1, 7 round-2), none opened; round-2 bodies are based on `cards`, not upstream main; replace (#679) placeholders [user] (PRB; PLAN "Waiting on the user")
- Upstream context: drop shadow #679 open, outer glow #680 closed [user] (PLAN Task 3)
- User review of `cards` (round-2 branch reviews already done ~08:20); round 3 branches presumably next [user] (PLAN "Waiting on the user"; M:next-session-plan)
- Cleanup candidates (do not remove yet): 29 worktrees under /private/tmp/wt-* incl. /tmp/wt-r4-{bevel,blur,gradshade}, /tmp/wt-review-r2, /tmp/wt-pipeline-dev; local branches pipeline-dev 25fcf2e1 (not in cards) and pr-679 [open] (GIT)

##### B2. Bot (AutoCroissant)
- B8 a deleted PSD stays in stats.pkl (prune only when num_changed > 0) [open] (BUGS 1.1)
- B9 blob 5xx records the tree ETag; card stays stale [open] (BUGS 1.1, 2.5)
- B11 fonts on the jetson: font cache keyed by MDW/Fonts tree sha for bake + details gather [open] (BUGS 2.3; PLAN checklist 6)
- Repeat on_ready after a failed RESUME re-runs inits [open] (BUGS 2.5 "Not done")
- GIT_TERMINAL_PROMPT=0 / SSH BatchMode on git calls [open] (BUGS 2.5)
- Lazy GitPython in update_bot; requests.Session per download thread [open] (BUGS 2.5)
- No admin gate on /purge, /update_metadata, /mass_replace, /alias, /set_repo, /delete_song, /delete_all_music [user] (BUGS 2.5)
- card_bake: pass the merged PNG as the PSD preview once fork save(preview=) lands (card_bake still calls update_preview=True) [blocked: new tag] (PIPE §14 Q2; BUGS pass 2)
- tests/test_card_bake.py cannot collect under the fork venv (no requests); pyyaml not installed, two workflow-shape tests skip [open] (BUGS pass 2)
- /fonts cache from the local clone [open] (M:next-session-plan)
- Round 4 bot follow-ups: flip test_moved_effect_paints_the_new_digit and test_cannon_rows_refuse_parity_at_the_baseline [blocked: round 4 tag] (R4 §3)
- Parity decision for lit cells above mean 1.0 at 0% bad: Cannon 1.579, Glaive_Main_Cannon 1.649; 28 raw-export cards refused by ICC in render_region [user] (R4 §3)
- D3 remove stale Shadow Duelist from ABILITY_EXCLUSIONS after a golden-checked reparse [open] (BUGS 1.2; SK:autocroissant-psd-extraction-campaign)
- D4 old_stats balloon 637698b cleanup (optional, diff_stats-gated) [open] (BUGS 1.2)
- Extraction: scripted regression harness (old vs new parser; sandboxed local sweep) [open] (BUGS 2.1; SK:autocroissant-psd-extraction-campaign Option 1)
- Extraction: gap/type mismatch validator (high precision); bbox-anchored injection (research); exclusion burn-down 21 -> certified list [open] (BUGS 2.1; SK:autocroissant-research-frontier Item 3)
- Parser: skip hidden text layers on rule pages (4 rulebook entries change) [open, ordered after page fixes] (RULE P3; PLAN Queued)
- Idea 14 validator: orphaned FX layers whose words left the ability; check whether EFFECT LAYER OUT OF DATE covers it [open] (PLAN idea 14)
- Task 3c: effect layers on pushes (Action; PSD quiet ~30 min; skip own commits; MDW skip list; Discord) [blocked: PSD hold for live runs] (PLAN Task 3c)
- S9: star edits move PSD + PNG to the new N Stars folder (aliases, old_stats, authors, TTS save follow) [blocked: pipeline round 1] (PIPE §14 Q4; STD L14)
- Push guard later rules: deviations by class (check_cards.py or a bot module), report-only, by hand first [open] (STD §5)
- Fix-on-push for 0-pixel classes only (later; never during a hold) [blocked: PSD hold + proven batch] (STD §5; M:website-and-pipeline-ideas)
- Deploy state of df4a439 / 1048324 (font_check exit-0 used by /fonts) on the jetson: not recorded [open] (M:jetson-server last deploy 13:38)

##### B3. Website (TTSCardMaker)
- Printed titles go live only after the jetson's pickles are pushed and site.yml runs [blocked: /push] (M:ttscardmaker-integration; BUGS V5, renumbered from V3 on 2026-10-21 evening)
- card-edits.yml push trigger on site/Card Edits.json goes live when the hold lifts; header comment still cites S5/Thunderfury [blocked: PSD hold] (WF/card-edits.yml; PIPE §8)
- site/README.md has a Font check section but no Card check section [open] (SREADME; WF/card-check.yml)
- Profiles not done: avatar, GitHub link, per-profile share dialog, People tab counter [open] (NOTES §7)
- People not done: per-person chart, edit commits in recent, bake credit to the requesting editor, overall title [open] (NOTES §8)
- Profile editor not done: deck builder picker still its own dialog; Enter on a chosen tile; rerender() closes an open picker [open] (NOTES §9)
- Details not done: rulebook pages, Changes tab/table view, "has details" filter, per-font state memory [open] (NOTES §11)
- Decks not done: deck search from Cards, "decks using this card", deck count on People, OG montage; data/decks.json empty until a first publish [open] (NOTES §12; M:next-session-plan)
- Inline editing proposed, not agreed: stars, token picker on `[`, tag completion on `(`, type-row palette, rulebook title/text; Subseries/Hidden/ban drag facets [user] (PLAN idea 20)
- Build speed: accent colour cache by blob hash (-2.9 s), hardlink dist copies [open] (BUGS pass 2)
- Known hole: a `name` the pipeline will bake never shows "PSD pending" [open] (PIPE §9)
- Risk: build unpickles bot pickles with stub classes (code execution via AutoCroissant main; renames blank fields) [open] (BUGS 1.6)
- Task 2b TTS "Table" tab reverted, not rescheduled; the Play tab must not reuse "Table" [open] (PLAN Task 2b)

##### B4. Standardization and the PSD hold (all writes wait for the user's go)
- Lift the PSD hold [user] (M:psd-edits-hold; PLAN "Waiting on the user")
- Before the day (allowed now): rehearsal on /tmp/rehearsal copies of ~110 PSDs, three gates, expected per-card diff list [open] (STD §4.1)
- Before the day: new fork script for L2 template rows; L5/L6 rename/move scripts (fork layer-move API to confirm) [open] (STD L2, L5)
- Before the day: 15 chart styles loaded in Photoshop from MDW/Debuff And Attributes Layer Effects.asl [user] (STD §4.1, L7)
- L1 inline icons -> stoogedings tokens: 42 cards / 71 layers (12 cards / 23 layers within the 651); header badges stay images [blocked: hold] (STD L1; M:website-and-pipeline-ideas)
- L1 open before batch: `[fire]/ [machine]` spacing; token order vs gap order on every card [open] (SK:ttscardmaker-companion §5; STD L1)
- L1 tool std.jsx lives only in /private/tmp/fontexplore/std/ (copy into MDW/Scripts/ before a reboot) [open] (STD L1)
- L2 stat rows: 45 deviating cards; 37 pixel/drawn-bar cards rebuilt as template rows (7 Isaac tainted, Dark_Esau, Skeleton_Mate, Grimora, Puzzled_Cat, Baba, 5 Clots, False_Knight, Minecraft_Panda, Bulborb Larva, Tainted_Lazarus, Tainted_Eve, Empress Bulblax, Dirty_Bed, Blank_Card...); odd paths Uumuu, Whispy_Woods, Spinni, Little_Guy; 16 atk/def swaps, 26 odd bar names [blocked: hold] (STD L2; M:website-and-pipeline-ideas)
- L3 Sonic: two rows under Spd Dark with a marker in each; Cannon's Top 10 marker stays [blocked: hold] (STD L3; PIPE §7 idea 13)
- L4 digit effects: no PSD change needed (move on toggle); Mettaton lit-cell toggles wait for round 4 [blocked: round 4] (STD L4)
- L5 Ability layers: 34 out-of-Text moved under Text; 7 hidden (Abyss_Watcher, Bulborb Larva, Shadow_Duelist, Warrior_Dee, Butterfly, Empress Bulblax, Apex_Predator); 20 blank filled or certified; 5 several-Ability cards stay refused [blocked: hold] (STD L5; M:website-and-pipeline-ideas)
- L6 titles retyped as text: Sonic (Layer 0 copy), ABBI, Tyler, Minecraft_Panda, Pandanoko, Holy_Card; hidden Name: Baba, Man_Serpent_Shield, E._Gadd's_Garage [blocked: hold] (STD L6; M:website-and-pipeline-ideas)
- L6 six case-typo titles (SPringtrap, Dark MAtter, MEtal General, MAsked Dedede, StaR DREAM, Zero OVerride): ask card by card before any reprint [user] (STD L6)
- L7 Effect Layers - All Cards.jsx on the 40 overlay cards; 16 overlays on 10 cards stay refused [blocked: hold] (STD L7)
- L8 six reprints: Clot_White "White CLot" -> "White Clot"; PoD08-PoD12 M.O.A.B., D.D.T., B.F.B., Z.O.M.G., B.A.D. (pipeline live run) [blocked: hold] (STD L8; PIPE §12)
- L9 Text outline: not forced (decision); only per-card owner calls [user] (STD L9; M:website-and-pipeline-ideas)
- L11 rulebook pages: delete 6 hidden layers (09_TakingYourTurn x2, 11_Abilities x2, 12_Reactions2, 22_Negation); overflow/off-canvas (11_Abilities6, 12, 14, 20); V1 16_The_Field box; X2 icons after page numbers on 6 pages; 20_Creature_Types `tornado` -> `wind` (D2) [blocked: hold] (STD L11; RULE §3; BUGS D2)
- L12 two PNGs without PSDs (Stubborn_Queen, Gust_of_Trees): ask the author [user] (STD L12)
- L13 fonts: only BabaUI (Baba.psd) missing, deferred by the user [user] (M:ttscardmaker-integration; STD L13 stale)
- L14 stars: God Tamer, Bubble stars need grouping (NO STARS FOUND) [blocked: hold] (STD L14; STD §5)
- L15 class D (9 cards: Jevil, Massive_Moss_Charger, Moss_Charger, Marx, Wham_Bam_Jewel, Necrodeus, Sans, Morpho_Knight): renderer first [open] (STD L15)
- Class H stale exports to re-export: Skullord, Restrained, Blood_Bomb, Dark_Mind_Eye, Asteroid_Field, PoD04, The_Collector, Holy_Blade [blocked: hold] (RS cards_v2; R4 §4; M:next-session-plan)
- D1 Computer Virus (all stats missing) PSD content [blocked: hold] (BUGS 1.2; STD §5)
- S5 live runs and S8 the user's 10 acceptance edits (Mini_Doomer hp 5 -> 8 ... Mettaton_EX) [blocked: hold] (PIPE §12, §13)
- After the batch: /update_stats, diff_stats.py, /push, site.yml + tts-save.yml, then ABILITY_EXCLUSIONS to the certified list, refusals narrowed, ledger shas [blocked: batch] (STD §4.4)

##### B5. Rulebook and game data
- The team's timing-system update; then stoogedings badges + rules skill refresh; do not define tags before [user] (M:website-and-pipeline-ideas; PLAN "Waiting on the user")
- RULE P0 frame confirmation (classes, decision log at docs/Campaigns/rulebook-decisions.md, Photoshop fixes, checker by hand) [user] (RULE P0)
- RULE P1 checker site/build/check_rules.py (reproduce §3 numbers: 59 undefined tags on 44 cards) [blocked: P0] (RULE P1)
- RULE P2 decision log (not created) [blocked: P0] (RULE P2)
- RULE C1-C6 contradictions; U1 tags (1GCD, 2CD, 1CD, 2U, 2GCD, 3U, 4U, (1/2CD)); U2 undefined terms [user] (RULE §3)
- RULE M1 Smouldering vs Smoldering, Regeneration vs Regen; M2 pages define no status effect; M3 10 unused status effects [user] (RULE §3)
- RULE T1 Suffocation "Can't Effect"; T2 page typos; G1 (OFS) on Pandora's Gym, White Palace, (WOF) on Earth Totem [user] (RULE §3)
- RULE X1-X3 parser slips (14_Item_Types machine icon, icons after page numbers, 08 star glyphs, 15 swallowed gaps) [open] (RULE §3)
- RULE S1 silences (lane ties, combo rounding, hand limits, starting BRICKS...) [user] (RULE §3)
- RULE P4 guard step in site.yml (own job / continue-on-error); P5 one Claude API semantic pass; P6 via simulator [blocked: P1-P3] (RULE §4)
- Idea 15 text-first rulebook (text per page, fork-rendered, pagination) never started [blocked: hold + timing update] (PLAN idea 15)
- D7 Twin Emperors 2nd DEF reads 4, card shows 10: fix per creature on the website, tick "parser notes checked" [open] (BUGS 1.2)
- D6 0 of 813 PSDs on stoogedings (see L1) [blocked: hold] (BUGS 1.2)
- Rulebook gaps list in the rules skill: "~22" status count stale (19) [open] (RULE M2)

##### B6. Campaigns not started
- 16 Simulator (Play tab): designed, not scheduled; its precondition (deck browser) landed; questions 1-7 open [user] (SIM; PLAN idea 16)
- 17 Game reports: designed; P1 zones in the save (tts_save.py) need nothing blocked; questions 1, 2, 5, 6 open [user] (GR; M:website-and-pipeline-ideas)
- 18 AI cards: becomes a separate AI sandbox repo + own website and bot; first target 3-star Kirby; venues Claude Opus API or a custom model; doc D1/D2/D3 to revise [user] (M:website-and-pipeline-ideas; AI)
- AI Q3 monthly API budget, Q4 how the bot runs it [user] (AI §5)
- 19 Rulebook consistency: see B5 [user] (RULE)
- 22 Web editor (card as data): after S8, the standardization batch and parity; data files under site/cards/ (decided); questions 2-9 open [blocked: hold + parity] (WEB §3, §10)
- WEB preview option (c) needs a `values` input on card-edits.yml; (a) Pyodide spike [blocked: pipeline live] (WEB §5)
- 2.2 AI items (/ask Q&A, art LoRA, round-trip verification): parked with the AI sandbox [user] (BUGS 2.2)

##### B7. Jetson and operations
- Jetson stats.pkl from today's version-2 re-parse is unpushed: the user's /push or the 05:30 reminder 054b5933 [user] (given; M:jetson-server)
- Gulka and Po not stamped v2 (blob streams broke mid-file); next /update_stats re-fetches; tree ETag shortcut off until then [open] (M:next-session-plan)
- Fork not installed in the jetson venv (stock psd-tools 1.25.0) [blocked: new tag] (given; PIPE §8)
- /fonts never run on the jetson: jetson-gathered psd_details records carry supplied: null [open] (given; BUGS B11)
- S7 fallback only if ever needed: fork tag in venv, fonts dir, one live bake with Actions disabled, CPU/RAM peak [blocked: need] (PLAN checklist 6; PIPE §8)
- V1 /sync_emojis refresh:True not run live [open] (BUGS 1.5)
- V2 reminder rewrite: watch a live firing (56b43b9 at 05:30 suggests the /push reminder fired) [open] (BUGS 1.5)
- V4 live /query Raw Kawk and /query Black Clot after the push [open] (BUGS 1.5)
- OJWikibot unit copies in scripts/services/msrouji/ still say python3.10 [open] (PLAN Task 0)
- Bot's fine-grained token read-only on the card repo (commits via SSH); Contents: write would enable the API path [user] (PLAN Task 2)
- TTS: confirm a player with mod caching sees an updated card refresh (Task 2 step 7 never recorded) [user] (PLAN Task 2)
- Reminders: d2664752 /update_stats 5AM, 054b5933 /push 5:30AM, 134e3b24 weekly; edit reminder.pkl only with the service stopped [open] (M:jetson-server)
- History clone on jetson (git 2.25): never diff or log --name-only [open] (M:jetson-server)
- archive/local-update-stats is local and unpushed; ask before pushing [user] (M:archive-on-branches; GIT)
- /tmp at reboot risk: /tmp/wt-* worktrees (bot tests cite /tmp/wt-pipeline-dev and /tmp/wt-cards), /private/tmp/fontexplore (std.jsx, JSX trial copies tests use, tts_survey scripts), /tmp/r4* diagnostics, /tmp/i2_census.py, /tmp/scout, /tmp/review_tests_*.log, /tmp/deck_fixture, /tmp/verify_*, /tmp/heat, /tmp/*_cards.txt [open] (R4 §6; PIPE §7; PLAN header; M:next-session-plan)
- /tmp/card-check staged copy is now committed (a3451c8): cleanup candidate [open] (BUGS retired; GIT)
- Workflow journals for resumption: ~/.claude/projects/-Users-michaelsrouji-Desktop-AutoCroissant/5c6ede3c-.../subagents/workflows/ [open] (M:next-session-plan)

##### B8. Docs, skills and memory upkeep
- Session wrap-up: fresh dated checklist in PLAN; move today's checklist to docs/Finished/<dated>/README.md (not there yet); refresh memory; Opus completeness check [open] (M:agent-model-preference; PLAN checklist 8)
- PLAN: item 7 still [~] though every sub-idea landed; heading "Ideas 5-23" (24 exists); Order section and "Waiting on the user" stale [open] (PLAN)
- STD §6 questions 1-9 answered 2026-10-21: record the answers; §5 rule 1/2 status lines ("staged") stale; L13 fonts and L15 class B stale; header "companion skill does not yet" point here [open] (STD; M:website-and-pipeline-ideas; R4)
- WEB: Q1 answered (site/cards/); §3.3 class B and branch-merge state stale [open] (WEB; M:website-and-pipeline-ideas)
- AI: revise D1/D2/D3 for the sandbox repo; Phase 2 dependency (text engine now in cards) stale; docs/README row cites branch ai/cards [open] (AI; M:website-and-pipeline-ideas)
- GR: Q3, Q4, Q7 answered; D5 "folders do not exist" and "no private games until encrypted decks" stale [open] (GR; M:website-and-pipeline-ideas)
- SIM: status says profiles not created and starts after the deck browser (both done) [open] (SIM)
- RULE: duplicate id T1 (Suffocation; tornado); Q1 now waits for the timing update [open] (RULE §3, §8)
- BUGS: 1.6 "7 fonts missing" -> BabaUI only; retired card_check row says "staged in /tmp"; V3 row (re-parse ran 10:18-10:40); two "V3" ids; D1 now one card [open] (BUGS)
- PIPE: S9 cited in Q4 but missing from the §13 table; memory's doc-drift list (card_edits.py citations, creatures row, step 9 parity wording, §8 trigger) maybe fixed by f1379cb: verify [open] (PIPE; M:next-session-plan)
- WF/card-edits.yml header comment (S5/Thunderfury) stale [open] (WF/card-edits.yml)
- Memory: MEMORY.md next-session-plan line, next-session-plan.md description, ttscardmaker-integration description + "Pending" + "Printed titles" (fold reverted), jetson-server Python paragraph and last-deploy line all stale [open] (M:*)
- Skills: unquoted `word: word` frontmatter in 8 skills; impossibility-cards-reference §4-5 refresh rule after parser changes [open] (BUGS Maintenance; SK:impossibility-cards-reference)
- Skills: research-frontier Item 3 residue OPEN; extraction-campaign Option 1 harness open [open] (SK:autocroissant-research-frontier; SK:autocroissant-psd-extraction-campaign)

---

#### Contradictions found and fixed on 2026-10-21 evening (kept so the next reader knows what was reconciled)

- Gulka and Po not stamped parser v2; tree ETag shortcut off until re-fetched: only M:next-session-plan.
- review/round2-consolidated 31e1a294 (unpushed) must merge into cards: only M:next-session-plan and R4 §5; absent from PLAN and RS.
- Class H stale-export list (Skullord, Restrained, Blood_Bomb, Dark_Mind_Eye, Asteroid_Field, PoD04, The_Collector, Holy_Blade): RS cards_v2 + R4 §4 + M:next-session-plan; not in STD ledger.
- Class B: STD L15 and WEB §3.3 say "not reconstructible, rasterize in Photoshop"; R4 §2.1 says fixable from FEid/FMsk (13 cards). Contradiction.
- STD §6 lists 9 open questions; M:website-and-pipeline-ideas answers 1-9 (inline only, rebuild 37 bars, no forced outline, move 34 abilities, retype titles, deviations refused, fonts are the user's, report-first guard, old_stats archive accepted).
- WEB Q1 open (beside the PSD) vs M:website-and-pipeline-ideas "data files under site/cards/".
- AI D2/D3 (ai/cards branch, TTSCardMaker ai-drafts) vs M:website-and-pipeline-ideas "separate AI sandbox repo, out of TTSCardMaker".
- GR Q3/Q4/Q7 open and D5 "none of the three folders exists" vs M:website-and-pipeline-ideas answers and landed profiles/decks/People.
- GR §5 "no private games until encrypted decks exist" vs encrypted decks landed (TTSCardMaker 305812c).
- SIM status "starts after the deck browser; profiles not yet created" vs both landed (PLAN idea 5, 7).
- RULE §8 Q1 (define or rewrite U1 tags) vs M:website-and-pipeline-ideas "wait for the team's timing update".
- BUGS 1.6 "7 fonts missing" and STD L13 "6 faces missing" vs M:ttscardmaker-integration "only BabaUI" (a88b232).
- BUGS retired row and STD §5 say card check "staged in /tmp/card-check"; it landed (df4a439, a3451c8). STD §5 rule 1 "live once committed" vs PLAN idea 21 "LIVE".
- PLAN checklist 7 [~] vs every sub-item landed; M:next-session-plan still says item 7 '[ ]'.
- PLAN "Waiting on the user: round-2 branch reviews" vs M:next-session-plan (user committed all six review fixes ~08:20).
- Idea 15 (text-first rulebook) sat between 13 and 6 in the agreed order; skipped, not marked blocked anywhere (PLAN Order; M:website-and-pipeline-ideas).
- M:next-session-plan "Still open: B8/B9/B10, the 1 s queue tick" vs BUGS (B10 and tick retired 2026-10-21).
- M:ttscardmaker-integration "Printed titles" describes the case fold (squash/printed_title) the user reverted (ba500bd; M:website-and-pipeline-ideas "literal").
- M:jetson-server "next: sudo make altinstall ... remove 3.10" vs PLAN Task 0 done; its last deploy (13:38) vs e4e0a58 deployed ~14:52 (M:next-session-plan).
- RS cards_v3 "Nothing pushed" and PRB round 2 "Nothing pushed but feat/text-engine" vs GIT (origin/cards, origin/fix/*); PRB predates the 13 user review commits.
- PIPE Q4 cites step S9; the §13 step table stops at S8.
- WF/card-edits.yml header: push trigger "comes with S5 once Thunderfury is aligned" vs PIPE Q7 aligned; now gated only by the hold (M:psd-edits-hold).
- BUGS V3 "first live re-parse pending" vs M:next-session-plan (ran 10:18-10:40, 817 stamped).
- BUGS V2 "not known to have fired" vs GIT 56b43b9 PICKLE at 05:30 (the /push reminder).
- RULE §3 uses id T1 twice.
- D7 Twin Emperors fix is a website edit doable now: only BUGS 1.2.
- OJWikibot unit copies still python3.10: only PLAN Task 0.
- Stubborn_Queen / Gust_of_Trees PNGs without PSDs: only STD L12.
- Glaive_Main_Cannon parity 1.649 and the 28 raw-export ICC refusals: only R4 §3.
- Standardization rehearsal on /tmp copies (allowed now, not started): only STD §4.
- std.jsx and the JSX trial copies (tests/test_psd_edit.py skips without them) live only in /private/tmp/fontexplore: STD L1, PIPE §7.
- Bot fork tests cite /tmp/wt-pipeline-dev and /tmp/wt-cards (tests/test_card_bake.py, tests/test_psd_edit.py docstrings); pipeline-dev is not in cards: GIT only.
- card_bake still uses update_preview=True; PIPE Q2 says fork save(preview=) "not done" but round 3 perf/save-with-preview implements it.
- PSD-pending hole for baked names: only PIPE §9.
- Pickle-unpickling code-execution risk in TTSCardMaker Actions: only BUGS 1.6.
- Bot token read-only on the card repo (SSH commits): only PLAN Task 2.
- Task 2 step 7 (mod-caching refresh) never confirmed: only PLAN Task 2.
- Companion §5 open items (slash spacing, token order) before L1: SK:ttscardmaker-companion and STD L1.

---

---

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
   Actions can't do it (time, size or download limits), the jetson handles the card browser's edit requests by hand
   (`/bake_cards`; decided by the user 2026-10-21: no notice in the bot, the Action is the one live runner).
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

## Ideas 5-24 (2026-10-20, user; 24 added 2026-10-21)

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
  code stays the way to pass a hand around. LANDED 2026-10-21 (TTSCardMaker 305812c; `site/assets/decks.js`, `data/decks.json`,
  static `decks/<owner>/<slug>/` pages for public decks; private decks listed only on the owner's person page).
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
  Status 2026-10-21: push guard rule 1, the font check, LIVE (TTSCardMaker `.github/workflows/font-check.yml` running `commands/font_check.py`; the `DISCORD_FONT_WEBHOOK` secret set; the bot's `/fonts`); rule 2, the card check, LANDED (AutoCroissant df4a439 `commands/card_check.py`, TTSCardMaker a3451c8 `card-check.yml`; first whole-repo dispatch run 39690899334, job summary only); both checkers report one error line and exit 0 on an API failure since the same day. The PSD ledger itself stays on hold.
- **22. Web editor** (later): an in-depth browser card editor once psd-tools is extended and the cards standardized; end
  state "card as data": one data file per card is the single truth, PSD and PNG are regenerated by the fork, the parser
  verifies the round trip, Card Edits.json dissolves, so PSD edits and database edits are one thing; migration in tiers,
  server-rendered previews first; grows from item 20; `docs/Campaigns/WEB_EDITOR_CAMPAIGN.md`.
- **23. Card details dropdown** (user, 2026-10-21; bot side landed 2026-10-21: `commands/psd_details.py` gathers one record per
  card PSD by blob sha into `psd_details.json`, in `/update_stats` for every re-parsed card and by the backfill CLI
  `python -m commands.psd_details --clone <dir>` for all 813 (8.9 s on the Mac, 2.7 MB, records 0.9-16.7 KB, median 3.7 KB);
  `/push` commits the file with the pickles, `/pull` keeps it on a conflict, `/fonts` reads its fonts before downloading;
  site side built the same day: `site/build/build_site.py` fetches the file, joins records by blob sha and writes
  `data/details/<id>.json`; `site/assets/details.js` renders the dropdown; LANDED 2026-10-21 as AutoCroissant ad90c62 (file backfilled
  and committed, jetson deployed 13:38) and TTSCardMaker 9bdbc6f (site build green: 771 of 773 cards have a record, the two without are image-only)): on the card browser, a per-card "Details" dropdown with every useful
  fact about the PSD: fonts used, each expandable to the layers using it (missing fonts flagged); text layers (text, font,
  size, colour, visibility); effects per layer; smart objects; canvas size, colour mode, ICC profile; type-icon style
  (image layers or font tokens); stat-row encoding; status-word FX and overlay layers; template match and ledger
  deviations; parser problems and PSD-vs-edit values; a collapsible raw layer tree as the catch-all. Data: the bot
  gathers it in /update_stats (it already opens every changed PSD) into a committed derived JSON keyed by PSD blob sha
  (no new pickle), pushed with /push; build_site.py writes one small file per card, fetched when the dropdown opens. The
  font check's per-blob cache, /fonts and the push guard read the same data. After the website foundation (6-8).

- **24. Full-page profile editor with visual card picking** (user, 2026-10-21): editing a profile opens a full page (not the
  small dialog) where the favourite/hated card and item are picked from card tiles with previews (search + filters like the
  browser, click to pick), the featured hand is picked from the deck builder's saved hands, and the page previews the
  profile as it will look. Same publish flow and file format as today. Also (same request): a person's page shows their
  contributions AND a visual grid of the cards they made plus author facts (series, kinds, stars, first and latest card,
  most edited card...), not only counts; built right after ideas 9-11 land. The picker is the primitive idea 20 reuses.

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

The 2026-10-22 checklist at the top is the order: fork integration, round 4, jetson/data, the /tmp rescue, hold-safe
standardization work, bot small items, website follow-ups, upkeep. Of the ideas, 5-14 and 20-24 landed (12-14 as code only under
the hold); 15 waits for the hold and the timing update; 16-19 and 22 are campaign docs; 21 waits for the user to lift the hold.
Tasks 0, 1, 1b, 2 and 4 are done, Task 3's remaining steps are the checklist's items 1-2, Task 2b was reverted, not rescheduled.
