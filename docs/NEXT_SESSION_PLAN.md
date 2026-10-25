# Next session plan (2026-10-25 ~04:00)

Start in `~/Desktop/AutoCroissant`. Read memories `owner-decisions-2026-10-24-night`, `lean-tests-and-verification`,
`jetson-server`, `persistent-work-paths`, `agent-model-preference`. What landed up to now and the rulings in brief:
`docs/Finished/session-2026-10-24/README.md`.

**Standing rules:** work unattended; questions only when the owner says "ask me questions" (then report "Waiting on the
owner"). Agents on Opus or lower. Commit to main; `work/*` only while in flight. Persistent paths only
(`~/Documents/Tools/wt/`, `~/std-work-1022/`). Push and deploy in one sitting (`jetson-deploy` skill). Tests:
`~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests`; few new tests, one light review. One PSD rewrite per card
per pass; never race a teammate's PSD push (30 minutes). No guides for the owner, no developer notices.

## State

AutoCroissant main 0ca9f33; TTSCardMaker origin/main 86733bd2; fork pin `cards-20261025` (9edbc1fa). Cut over since
10-24 23:02: Card sync (card-sync.yml) is the data build, bake, TTS save and changelog; a card push starts a fixed
30-minute window; the bot refreshes its card data on use. Pickles frozen at PICKLE 1282743. `CARD_GUARDS=on`. Census:
`docs/Future/PROBLEM_CARDS_2026-10-25.md`.

## In flight (not landed)

- [ ] **Card sync split**: data branch, changelog and site at once on a card push; only main writes wait 30 minutes.
      `work/split` (AC e9112e8, TC 44440a31). Status: built, not landed.
- [ ] **Overlay-safe badges**: outlined pill and one-colour sticker `salt` alternates, chosen by the bake under any
      overlay; the badge FX-layer workaround removed. `work/badgealt` (AC f3f061b; TC d0e6c61d, e9700ddd, b511a638
      "retire 31 badge FX layers"). Status: the batch commit waits for the owner's before/after sheet.
- [ ] **Retire Discord card-repo commands**: `/update_metadata`, `/mass_replace`, `/alias`, `/bake_cards`,
      `/update_tts_save` (`work/retire` e7e7879; code kept on `archive/discord-edits`, push it with the landing), and the
      website's Rename everywhere (`work/rename`, AC 70c9ed9, TC 706f750a). Status: built, not landed.
- [ ] **Data build step 10**: delete the pickles, `psd_details.json`, `/push`, the legacy `/update_stats` path (B9 with
      it), `commands/history_backfill.py` and its files and test (DATA_BUILD_ACTION §14). `work/pickles` (both repos,
      no commits). Status: owner said now; he tells bobberlington.

## Queue (owner's order)

1. [ ] **Parts editor, one box per PSD layer** (`docs/PARTS_EDITOR_DESIGN.md`, §14 answered): three tracks (data and
       parser, bake, site). It also brings stars moves to the `N Stars` folder (S9), adding an ability or a first type
       from the template, dropping ABILITY TEXT NOT FOUND and `ABILITY_EXCLUSIONS` (21; BUGS D1), the never-overwrite-
       Photoshop check, and ` #N` variant tags for the 5 multi-picture PSDs (one scripted commit, plan shown first).
       Status: designed, build next.
2. [ ] **Test suite cut** (`~/Documents/Tools/wt/scratch/tests-plan/PLAN.md`: 1,690 -> 619 tests, xdist). Status: planned.
3. [ ] **Smaller PNG pushes**: oxipng CLI, XMP strip, opaque as RGB; never rewrite existing PNGs. Status: not started.
4. [ ] **Website small fixes** (#15; "Website fixes" in `~/Documents/Tools/wt/scratch/ideas-sweep-report.md`). Status:
       not started.
5. [ ] **Layout support**: non-template stat bars, unusual star layouts, Sonic's one-row SPD bar (census). Status: approved,
       not started.
6. [ ] **Verdicts to `db/verdicts.json`**, bake and guard events in the changelog (DATA_BUILD_ACTION §16 item 7).
       Status: not started.
7. [ ] **`game.txt`**. Status: later.

## Waiting on the owner

- [ ] Before/after sheet for the 31 badge FX layers (`work/badgealt`).
- [ ] Photoshop work, his own: one Name and one Ability layer per card, the rulebook's Header and Page Number layers,
      retyping the picture titles (the census lists the cards).
- [ ] Delete the 30-minute `/update_stats` reminder be1b0bcd (refresh on use made it unneeded).
- [ ] Tell bobberlington about step 10.
- [ ] Stale exports to re-export in Photoshop: Jevil, Credit_Card, Forget_Me_Now and the rest of the 18 class-H cards.
- [ ] Ten cards to open in Photoshop: `~/Documents/Tools/psd-research/std-batch2/b1_final/LANDED.md` and `LANDED_B2.md`.
- [ ] BUG 2 points: Clot Gold "2T1" -> "2T2"; a typed badge in a title (plain words, flagged); `:sticker` beside plain
      tokens (Palace of Shadow); words typed in place of a badge; Elmadan's leading space; Plague's icon size.
- [ ] Push guards: §20.4 (KEPT up to 300 px; drop the render line from the post); §15: the fork render refusing 9 of 34
      replays, the outside-R parity method, login fallback and commit link, per-rule waivers, R11 auto-rename, G8
      numbered PNGs.
- [ ] History (DATA_BUILD_ACTION §7.4): allowances for high stats (Sonic 23, Pix, Twin Emperors, the Cannon, Bill Cipher,
      Oryx, Tainted Lazarus' 110s, 10+ stars); 26 restored versions differing in types only; Pontiff Sulyvahn's credit;
      side-file lines matching no card (White Palace, Gust of Trees, Stubborn Queen).
- [ ] To tell him: BUGS D4's deletion-gap note; the `_bot` field; the edits cap of 100; per-field credit; Computer Virus
      flagged.
- [ ] Live TTS check: a teammate takes a card out of another colour's zone; a cached player sees an updated card.
- [ ] S8: the 10 acceptance edits (EDIT_PIPELINE_DESIGN §12), or call it superseded by live use.

## Waiting on developers

- [ ] Reset TTSCardMaker clones after the 10-24 rewrite: `git fetch; git reset --hard origin/main` (never Pull-merge the
      old 84 commits). No notice drafted (owner).
- [ ] Reinstall the stoogedings font once the new badges land.

## Smaller open items

- [ ] B12: kerning in `psd_edit.story_style_drift` (BUGS §1.1).
- [ ] D6: icon-font adoption, measured by `psd_details` `icons` (BUGS §1.2).
- [ ] V1 `/sync_emojis refresh:True` live; V4 `/query Raw Kawk`, `/query Black Clot` live (BUGS §1.5).
- [ ] Badge the 11 EarthboundZero cards, Koro-Sensei, The Evil Idol and Papyrus with the fork (owner ruling; unblocked).
- [ ] Reword psd_edit's star and type refusals (`psd_edit.py` ~4024 "ledger S9") and keep the site's `PLAIN_WHY` in step.
- [ ] Bake: inserted text takes the nearest word's colour (Binge Eater); `normalize_ability` drops quotes and spaces;
      appended text runs below the panel (Autopsy Report).
- [ ] Changelog: 247 of `data`'s events link commits the 10-24 rewrite removed (ideas sweep); remap or accept.
- [ ] Skills: 2026-07-11 line numbers in impossibility-cards-reference and failure-archaeology.
- [ ] TTS/AUDIT.md's two owner points ("new" = not in the save at 318001ea; White Palace in no deck); Stubborn Queen
      not in `resizeTargets` (not re-checked).
- [ ] TTSCardMaker docs with stale lines: `site/README.md` (details from the pickles), `site/NOTES.md` §5,
      `TTS/checks/README.md` (pickle push, `card_sync_steps.py`, `~/.venvs/tts-checks`).
- [ ] Cleanup: `wt/scratch` (128 GB), `wt/tc-full.git` (22 GB; after step 10), ~40 dead `wt/*-scratch` dirs,
      `~/std-work-1022` (15 GB; keep `glow/held.json`), ~87 GB of pytest temp folders. The jetson's OpenSSL 3.6 reaches
      end of life Nov 1.

## Ideas parked

- Research: rcache wiring in psd_edit, text geometry and a website overlay (partly in the editor's details v2),
  template fill stage 2, the Photoshop oracle loop, the Perpetua lead (Hollow Knight).
- Extraction: idea 14 (orphaned-FX validator), fix-on-push for 0-pixel classes, L11's rest, L15 class D. The gap-type
  validator is likely moot (gap injection now fills one card's ability and 8 rulebook pages).
- Folding the push guards into Card sync (PUSH_GUARDS §11).
- Campaigns, designed, not scheduled: 16 Simulator, 17 Game reports, 19 Rulebook consistency (U1 waits for the team's
  timing pages), 22 Web editor, 18 AI cards (parked).
- Accepted limits: Safari's 7-day storage signs a browser out; wide icons fitted to 1.35 em; no static page for private
  decks.

## Not chosen (don't build unless asked)

Printed-text audit; Discord `/preview`; the website edit-support map; interactive website cards; template passes
L2/L3/L5/L6; point badges; font-side hardening; a website service worker.
