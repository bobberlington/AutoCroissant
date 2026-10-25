# Next session plan (2026-10-25 ~06:00)

Start in `~/Desktop/AutoCroissant`. Read memories `owner-decisions-2026-10-24-night`, `lean-tests-and-verification`,
`no-guides-csv-only`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`. What landed and the rulings
in brief: `docs/Finished/session-2026-10-24/README.md`. Designs for the queue: `docs/Future/`.

**Standing rules:** work unattended; questions only when the owner says "ask me questions" (then report "Waiting on the
owner"). Agents on Opus or lower. Commit to main; `work/*` only while in flight. Persistent paths only
(`~/Documents/Tools/wt/`, `~/std-work-1022/`). Push and deploy in one sitting (`jetson-deploy` skill). Tests:
`~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests`; few new tests, one light review. One PSD rewrite per card
per pass; never over a teammate's newer PSD. No guides for the owner (the census CSV only), no developer notices.

## State

AutoCroissant main e6a5c54 (deployed); TTSCardMaker origin/main 577ea1e3; fork pin `cards-20261025` (9edbc1fa).
Card sync (card-sync.yml) is the data build, bake, TTS save and changelog, in two lanes: a card push updates `data`,
the changelog and the site at once; only main writes (bake, save, guard fixes) wait out a fixed 30-minute window
(first live main commit 59b97943). The bot refreshes its card data on use. The five Discord card-repo commands are
retired (6e8f8e3, e7e7879; code on `archive/discord-edits`); card edits are the website's, with Rename everywhere
(TC 706f750a, 02eef4ea). Pickles frozen at PICKLE 1282743. `CARD_GUARDS=on` (live fix b2561b66 after push 86733bd2).
Census: `docs/Future/PROBLEM_CARDS_2026-10-25.csv`.

## In progress (not landed)

- [ ] **Overlay-safe badges** (`work/badgealt`: AC 1059664; TC b3249078, bff181b5): outlined pill and one-colour
      sticker `salt` alternates, chosen by the bake under any overlay; the badge FX-layer workaround removed.
      Status: the 31-card batch (b511a638, rebased as 1cb63d11) must be **re-run on the current PSDs**, not merged:
      hdckighfkvhvgmk's push 86733bd2 changed PoD06, PoD10, PoD11 and PoD12. The owner approved the sheet (10-25 ~04:05);
      the re-run lands as one commit.
- [ ] **Data build step 10** (`work/pickles`: AC 8f433f4, 25d7e70, cc48201, 3a1adba; TC b0da3cce): delete the
      pickles, `psd_details.json`, `/push`, the legacy `/update_stats` path (B9 with it), `commands/history_backfill.py`
      and its files and test (DATA_BUILD_ACTION §14). Status: merged 2026-10-25 (0e6c7a0); the rest of the pickle
      era (the seed, the unpickler, the recovery) went on work/clean.
- [ ] **Guards' silent Composer fix** (`work/composer`, worktree `wt/ac-composer`, uncommitted): card text layers set
      to Every-line Composer become Single-line, the PNG re-exported if the text moved, committed with the other guard
      fixes, never mentioned in posts or comments. Status: merged (0fbe028); since work/clean it always switches (no
      0.9 bar): box text that rewraps is redrawn and its PNG re-exported through the guards' export.

## Queue (owner's order)

1. [ ] **Parts editor** (`docs/Future/PARTS_EDITOR.md`): one box per PSD layer, three tracks. Brings stars folder moves
       (S9), a first ability or type from the template, dropping ABILITY TEXT NOT FOUND and `ABILITY_EXCLUSIONS` (21;
       BUGS D1), the never-overwrite-Photoshop check, and the 5 variant PSDs' ` #N` tags (one scripted commit, plan
       shown first). Status: designed, build next.
2. [ ] **Test suite cut** (`docs/Future/TEST_SUITE_CUT.md`): xdist, 1,923 -> about 635 with the 15 site smoke tests;
       `discord_archive` and `discord_match` retired with their tests (done 2026-10-25, work/clean). Status: planned.
3. [ ] **Smaller PNG pushes** (`docs/Future/SMALLER_PNG_PUSHES.md`): oxipng CLI, XMP strip, opaque as RGB for every PNG
       the bot writes; no rewrite of existing PNGs. Status: designed.
4. [ ] **Website small fixes** (`docs/Future/WEBSITE_SMALL_FIXES.md`): phone tooltips, exact name first, deck export
       with pinned links, rulebook page links. Status: designed.
5. [ ] **Layout support** (`docs/Future/LAYOUT_SUPPORT.md`): 31 non-template stat bars, 21 unusual star layouts.
       Status: built on branch work/layout (2026-10-25), not landed; per-card results in the doc's "Built".
6. [ ] **Card sync verdicts** (`docs/Future/CARD_SYNC_VERDICTS.md`): `db/verdicts.json`, bake and guard events in the
       changelog. Status: designed.
7. [ ] **`game.txt`** (`docs/Future/GAME_TXT.md`). Status: designed, later.

## Waiting on the owner

- [ ] His Photoshop work: one Name and one Ability layer per card, the rulebook's Header and Page Number layers,
      retyping the picture titles, re-exporting the stale exports (Jevil, Credit_Card, Forget_Me_Now and the rest of
      the 18 class-H cards), and the 10 cards to open: `~/Documents/Tools/psd-research/std-batch2/b1_final/LANDED.md`
      and `~/Documents/Tools/psd-research/std-batch2/LANDED_B2.md`.
- [ ] Delete the 30-minute `/update_stats` reminder be1b0bcd (refresh on use made it unneeded); tell bobberlington
      about step 10.
- [ ] BUG 2 points: Clot Gold "2T1" -> "2T2"; a typed badge in a title (plain words, flagged); `:sticker` beside plain
      tokens (Palace of Shadow); words typed in place of a badge; Elmadan's leading space; Plague's icon size.
- [ ] Push guards: §20.4 (KEPT up to 300 px; drop the render line from the post); §15: the fork render refusing 9 of 34
      replays, the outside-R parity method, login fallback and commit link, per-rule waivers, R11 auto-rename, G8
      numbered PNGs.
- [ ] History (DATA_BUILD_ACTION §7.4): values the re-read confirms stay as printed and God Tamer's and Blank Card's are
      misreads (owner, 10-24). Left: the 26 restored versions that differ in types only; Pontiff Sulyvahn's credit;
      side-file lines matching no card (White Palace, Gust of Trees, Stubborn Queen).
- [ ] To tell him: BUGS D4's deletion-gap note; the `_bot` field; the edits cap of 100; per-field credit; Computer Virus
      flagged.
- [ ] S8: the 10 acceptance edits (EDIT_PIPELINE_DESIGN §12), or call it superseded by live use.
- [ ] The rulings draft (`~/Documents/Tools/wt/scratch/rulings/RULINGS_DRAFT.md`, 4 + 17 + 48 rows; ideas sweep item 9),
      for the team to confirm before anything is published.
- [ ] Live TTS check: a teammate takes a card out of another colour's zone; a cached player sees an updated card.

## Waiting on developers

- [ ] Reset TTSCardMaker clones after the 10-24 rewrite: `git fetch; git reset --hard origin/main` (never Pull-merge the
      old 84 commits). hdckighfkvhvgmk's is already on the new history. No notice drafted (owner).
- [ ] Reinstall the stoogedings font once the new badges land.

## Smaller open items

- [ ] B12: kerning in `psd_edit.story_style_drift` (BUGS §1.1). Status: open.
- [ ] D6: icon-font adoption, measured by `psd_details` `icons` (BUGS §1.2). Status: open.
- [ ] V1 `/sync_emojis refresh:True` live; V4 `/query Raw Kawk`, `/query Black Clot` live (BUGS §1.5). Status: open.
- [ ] Badge the 11 EarthboundZero cards, Koro-Sensei, The Evil Idol and Papyrus with the fork (owner ruling). Status:
      unblocked, after the badge batch.
- [ ] Reword the star and type refusals in `commands/psd_edit.py` (~4024, "ledger S9") and keep the site's `PLAIN_WHY`
      in step. Status: open; the parts editor removes most of them.
- [ ] Bake: inserted text takes the nearest word's colour (Binge Eater); `normalize_ability` drops quotes and spaces;
      appended text runs below the panel (Autopsy Report). Status: open.
- [ ] Skills: 2026-07-11 line numbers in impossibility-cards-reference and failure-archaeology. Status: open.
- [ ] TTS/AUDIT.md's two owner points ("new" = not in the save at 318001ea; White Palace in no deck); Stubborn Queen
      not in `resizeTargets` (not re-checked). Status: open.
- [ ] TTSCardMaker docs with stale lines: `site/README.md` and `site/NOTES.md` are fixed in `work/pickles` (TC b0da3cce);
      `TTS/checks/README.md` still names `~/.venvs/tts-checks`. Status: with step 10.
- [ ] Jetson: move from OpenSSL 3.6.5 (installed 10-24) to the 3.5 LTS, if wanted. Status: optional.
- [ ] Disk cleanup: `wt/scratch` (133 GB), `wt/tc-full.git` (22 GB; after step 10), ~40 dead `wt/*-scratch` dirs,
      `~/std-work-1022` (15 GB; keep `glow/held.json`), ~70 GB of pytest temp folders. In `wt/scratch` keep what the
      docs cite: `rulings/RULINGS_DRAFT.md`, `tests-plan/`, `ideas-sweep-report.md`, `png-research/REPORT.md`,
      `repo-size/workflow-result.json`, `cutover-check-1024/RUNBOOK.md`, `parts-editor/`. Status: open.

## Ideas parked

- Research: rcache wiring in psd_edit, text geometry and a website overlay (partly in the editor's details v2),
  template fill stage 2, the Photoshop oracle loop, the Perpetua lead (Hollow Knight).
- Extraction: idea 14 (orphaned-FX validator), fix-on-push for 0-pixel classes, L11's rest, L15 class D. The gap-type
  validator is likely moot (gap injection now fills one card's ability and 8 rulebook pages).
- Folding the push guards into Card sync (PUSH_GUARDS §11).
- Campaigns, designed, not scheduled (`docs/Campaigns/`): Simulator, Game reports, Rulebook consistency (U1 waits for
  the team's timing pages), Web editor (the parts editor is its next step), AI cards (parked).
- Accepted limits: Safari's 7-day storage signs a browser out; wide icons fitted to 1.35 em; no static page for private
  decks.

## Not chosen (don't build unless asked)

Printed-text audit; Discord `/preview`; the website edit-support map; interactive website cards; template passes
L2/L3/L5/L6; point badges; font-side hardening; a website service worker; JPG copies for TTS; rewriting existing PNGs.
