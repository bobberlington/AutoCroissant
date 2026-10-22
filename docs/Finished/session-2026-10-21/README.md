# Session 2026-10-21 (unattended, Fable 5.1 orchestrating, ultracode workflows)

The day's checklist as it stood when the session closed, kept for the record. Every item 1-7 landed; item 8 (upkeep) closed
with this report, the refreshed `docs/NEXT_SESSION_PLAN.md` and the memory files. Round 3's integration into `cards` and
everything after it moved to the 2026-10-22 checklist.

## What landed (both repos pushed; AutoCroissant tip 1048324 + the evening's docs commits, TTSCardMaker tip a3451c8)

AutoCroissant
- 00:09-02:55 web editor campaign doc (b40b95f); edit pipeline S1-S4 (6c376d2); ultrareview fixes; font aliases (f1379cb);
  0.1 s drains, atomic locked pickle saves, grouped /query_ability, font check + /fonts (ae1d2c5); plan item 23 (74a5c2c).
- 05:59 optimization pass 2 (a2670b9; `commands/git_commit.py`, font check 98 s -> 9.8 s; docs/Finished/optimization-pass-2-2026-10-21/).
- 07:42 PARSER_VERSION 2: printed titles into `display_name`, EXCESSIVE_STAT_EXCLUSIONS gone, lookup fix (5835139; jetson 07:49);
  10:02 titles shown literally as printed (ba500bd, user decision).
- 08:20 the user's review fixes cherry-picked (a7fb45e font check, ac3ae57 pipeline).
- 10:07 S6 (ideas 12, 13, 14) + S7 `/bake_cards` with `bake_lock` (6710a5b); 10:27 the edits-sha hook removed (62f8c1c, the Action
  is the one live runner); 11:14 a broken PSD download retried once (aa69799).
- 13:32 `psd_details.json` gathered in /update_stats, backfilled for 813 cards, pushed with the pickles, read by /fonts (ad90c62).
- 14:46 three blank-ability minions in ABILITY_EXCLUSIONS (21 names); tornado = wind recorded (e4e0a58; jetson 14:52).
- 16:03 push guard rule 2: `commands/card_check.py` (df4a439); 16:07 font check exit 0 on API failures; companion skill rows (1048324;
  jetson pull + restart 17:20).
- Bugs doc: B11 fonts on the jetson (ec0c988). Plan commits after every landing.

TTSCardMaker
- 00:51 PSD pending/refused badges; card-edits.yml dry-run dispatch (8393b3b); 01:11 the requested fonts (a88b232); 02:47 font-check.yml
  (f5a0e50); 06:00 Actions check out git_commit.py (8185aa6).
- 07:42 printed titles, allow_high_stats, Thunderfury aligned (1c13eaa).
- 08:50 ideas 6-8 sign-in top right, profiles, users page (4473dd1); 10:02 ideas 9-11 People tab with plain titles, `Michael = MichaelJSr`
  (41b929b); 11:23 idea 24 person cards panel + full-page profile editor + `pick.js` (812cb89); 12:28 idea 20 inline editing (d389416);
  13:39 idea 23 Details dropdown (9bdbc6f); 15:25 idea 5 published decks, private ones encrypted and shown only on the owner's page
  (305812c); 16:03 card-check.yml (a3451c8; whole-repo dispatch 39690899334 green, 26.7 min).

psd-tools fork and research
- cards_v2 (round 2 + text engine) then cards_v3 a6363a04 with the user's seven round-1 review commits; pushed; tag `cards-20261020`.
- The user's six round-2 review commits pushed on their branches; consolidated as `review/round2-consolidated` 31e1a294 (local).
- Round 3: nine branches built and reviewed (local; seven clean, two with a last fix not re-reviewed; integration moved to the
  2026-10-22 checklist after the workflow was stopped at 17:50); round 4 research: `RENDER_ROUND4_PLAN.md` (class B fixable from FEid/FMsk,
  byte rounding, bevel texture) with scratch worktrees /tmp/wt-r4-*.

Decisions the user took today are in memory `website-and-pipeline-ideas` (rounds 1-4) and `agent-model-preference`.

## The checklist as it stood

### Session checklist (2026-10-21, unattended; the user asks no questions be asked)

Every task still wanted this session, in order; a line is struck through here when it lands and the docs it closes move to
`docs/Finished/`. PSD and PNG writes stay on hold until the user lifts it.

1. [x] (landed 2026-10-21 ~06:30: round 2 integrated as cards_v2, the user's seven review branches merged as cards_v3, `cards` a6363a04 pushed, tag `cards-20261020` cut, RENDER_STATUS.md + PR_BODIES.md refreshed) Rendering round 2 (fork): wait for the workflow (6 branches on `cards`, harness ICC fix, integrate as cards_v2 + the
   text engine, critic incl. the stat-toggle check); review; then merge the user's seven reviewed branches
   (drop-shadow, gradient-overlay, outer-glow, text-engine, adjustments, bevel-satin, inner-shadow-glow) into `cards`,
   suite + harness, push `cards`, tag `cards-20261020` (S0 of the edit pipeline), refresh RENDER_STATUS.md / PR_BODIES.md.
2. [x] (done 2026-10-21: run 39605156043, dry run on Clot White: the Action installed the fork at tag cards-20261020, cached the fonts, baked the title 'White CLot' -> 'White Clot' in 33 s within the parity bar, uploaded before/after/diff crops and the new PSD/PNG as artifacts, committed nothing) S5: a dispatched DRY RUN of TTSCardMaker's card-edits Action against the tag; read its artifacts and report.
3. [x] Optimization and refinement pass 2 over tonight's new code (psd_edit, psd_regions, card_bake, font_check, the site
   badges and build): profile, dedupe the API/SSH commit code with tts_save, fix the single-result-no-image hole in
   /query_ability and the /fonts lazy-import shadowing, speed the font check, review tests; report to
   `docs/Finished/optimization-pass-2-2026-10-21/`. Landed 2026-10-21: `commands/git_commit.py` (one commit path),
   font check 98 s -> 9.8 s, both holes fixed, 350 tests; the fork-side bake speedups (preview write, viewport effects)
   wait for the worktrees.
4. [x] (done 2026-10-21, deployed to the jetson 07:49 as 5835139, site 1c13eaa; the next `/update_stats` re-parses every card once: `PARSER_VERSION` 2 reads the printed title into `display_name`, 157 cards, shown as printed, letter case included (user, 2026-10-21); `/query` also finds printed titles; `CardInfo.parser_version`
   makes the next /update_stats re-parse every card once; 8 goldens re-pinned; build_site shows the title with the stem as
   alias; `EXCESSIVE_STAT_EXCLUSIONS` deleted, Sonic and the Cannon `allow_high_stats` in Card Edits.json; Thunderfury aligned;
   20 name edits redundant now, 21 with Thunderfury, Sans and Papyrus have no `Name` layer) Decision Q1: the parser reads
   `Text/Name` into `display_name` (goldens refreshed; build_site shows it; the 22 redundant name edits drop by the
   redundancy rule); drop `EXCESSIVE_STAT_EXCLUSIONS` (Q3); align Thunderfury's entry.
5. [~] (running 2026-10-21 ~06:35: nine branches from cards a6363a04 incl. two bake speedups; integrates as cards_v4) Rendering round 3 if the critic leaves causes worth it (text gamma residue, group overlays, shadow shapes...).
6. [x] (landed 2026-10-21 as 6710a5b; deployed to the jetson the same hour; decided by the user 2026-10-21: the parity bar stays, the jetson hook is removed; the fonts cache is still open) S6 code only: the digit-effect move on toggle (idea 12), FX regeneration from text (idea 14), top-row-10 +
   bottom-row N-10 writing (idea 13); S7 jetson runner (/bake_cards by hand; the Action is the one live runner). No PSD lands.
   S7 code landed 2026-10-21: `commands/card_runner.py` (`bake_lock` shared with update_stats, `/bake_cards [card] [dry_run]`
   admin command, report posted like the TTS save's; 52 commands; 377 passed, 24 skipped). The `notice_edits` hook (one dry-run bake queued
   at the end of update_stats per edits-file change) was removed the same day: the user chose the Action as the ONE live runner,
   the jetson is a manual fallback. Only if the fallback is ever needed: the fork tag in `~/venvs/autocroissant`, a fonts dir on
   the jetson (no checkout there), one live bake with Actions disabled, the CPU/RAM peak.
   S6 code landed 2026-10-21 (design §7): idea 12 and 13 in `psd_edit.set_stat`/`template_rows`; idea 14 in `psd_edit.set_text_layer`
   (every ability edit regenerates the `Ability FX:` layers from the chart and hides the overlays it replaces; no fork change; proven
   against the JSX cuts on Gasoline and the Cannon, Mini_Doomer edits through every gate; bot suite 380 passed, 24 skipped under
   stock psd-tools). All of it lands nothing under the hold.
7. [~] (6-8 landed 2026-10-21 as TTSCardMaker 4473dd1; 9-11 as 41b929b with the People tab and plain titles; the person-page cards panel + idea 24 full-page editor with the visual picker site/assets/pick.js as 812cb89; 20 inline editing landed as the next TTSCardMaker commit (site/assets/inline.js: in-place stat cells, ability, name, chips; Ctrl/Shift-select + drag onto Series/Author; one batch bar; All fields fallback); 23 landed 2026-10-21 as AutoCroissant ad90c62 + TTSCardMaker 9bdbc6f (771 of 773 cards have a record; jetson 13:38); 5 (decks) landed 2026-10-21 as TTSCardMaker 305812c: public decks `site/decks/<owner>/<slug>.json`, private decks encrypted in the browser (PBKDF2 300000 + AES-GCM, nothing in clear but owner, slug and date), #/decks browser under the Deck builder tab (public only), deck pages, publishing through the edit batch, a Decks panel on person pages; private decks show only on the owner's page and by link (user, 14:30)) Website foundation (ideas 6-8): sign-in top right, opt-in profiles, users page; then 9-11, the person-page cards grid + author facts and 24 (full-page profile editor with visual
   card picking), 20 (inline editing), 23 (details dropdown), 5 (decks). Each as its own verified workflow against ~/Desktop/TTSCardMaker.
8. [ ] Memory and plan status after every landing; docs that complete move to `docs/Finished/` with a dated README.

Waiting on the user (not blocking): lifting the PSD hold (then S8, the standardization batch, the six title reprints,
Sonic, star moves); the remaining fork reviews (round-2 branches, then `cards`); opening the upstream PRs from
`~/Documents/Tools/psd-research/PR_BODIES.md`; BabaUI; the timing-system update (then stoogedings + the rules skill).

