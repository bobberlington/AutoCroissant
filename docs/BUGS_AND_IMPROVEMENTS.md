# AutoCroissant — Known Bugs and Future Improvements

Compiled 2026-07-11 from a full repo audit; revised 2026-10-19, 2026-10-20, 2026-10-21, 2026-10-23 and 2026-10-25 (twice). Detail, evidence and procedures
live in the skill library under `.claude/skills/`; each entry points at its owner (`ttscardmaker-companion`
covers TTSCardMaker: icon font, website, fonts, Photoshop scripting). Fixes route through
`autocroissant-change-control`. When an entry is fixed or retired, move its story to
`autocroissant-failure-archaeology` and delete the row here. The next session's plan is `docs/NEXT_SESSION_PLAN.md`.

**Retired** (stories in `autocroissant-failure-archaeology`):
- 2026-10-19: B1, B2, C7, C8, C9, C11; management.py Cython 3.3 failure; dead `.gitignore` patterns (Entries 1, 10).
- 2026-10-20: C13 token spellings (10.11); B6 by archiving local mode (10.10, 10.12); C10 and other dead helpers
  (10.7, 10.13); D5 stale stats.pkl (re-parsed and pushed: 795fd23, 7d0e29f).
- 2026-10-20 parser fixes: stars counted per star; stat bars per row (Sonic 20, Sans 1, heart bars counted);
  multi-creature cards get per-creature stats (impossibility-cards-reference §2, §3.4); duplicates (`<Name>N.png`)
  recorded with `duplicate_of` (ttscardmaker-companion §2.1, §2.5).
- 2026-10-20: AI generation moved to branch `archive/ai-generation` (B3 went with it); S1 fixed with `config.browser_cookies`; C4 fixed (10.16).
- 2026-10-20 operations: `/update_stats` re-parses only PSDs whose blob changed and saves after edits-only runs
  (10.15); old_stats history is appended only when a PSD's blob changed or the card moved (forced re-parses no longer
  archive); `/push` rebases onto new code first; history rewritten to drop an attribution trailer (10.14).
- 2026-10-20 (later): C14 sha-pinned image links (10.17); emoji sync inside /update_stats, weekly /sync_emojis reminder
  removed (10.18); TTS save updater and /update_tts_save (10.19); C12 Pillow and PyNaCl declared (10.26); V3 (Cython
  built on the jetson; a verification row, so no story: autocroissant-build-and-env §5).
- 2026-10-20 optimization pass: C1, C2, C3 (10.20); B4 (10.21); B5 (10.22); C6, by removing `/list_guild_members`
  (10.23); B7 (10.24); C5 (10.25); psd_analyzer split and the compiled shallow-path crash (10.27); network and speed
  (10.28).
- 2026-10-21: B10, atomic and locked card pickle saves (10.29); the 0.1 s queue tick and /query_ability's grouped
  output (10.30; the grouping reverted 2026-10-23: Discord previews only a few images per message).
- 2026-10-21 optimization pass 2: /query_ability's single-result hole (one card without a PNG answered `content=None`,
  which Discord dropped; it posts the name now) and the /fonts lazy-import shadowing; one commit path for the bake and
  the TTS save (`commands/git_commit.py`); the font check parses in a process pool (813 PSDs 98 s -> 9.8 s). Report:
  `docs/Finished/optimization-pass-2-2026-10-21/`.
- 2026-10-21 (plan item 23, bot side): `psd_details.json` (commands/psd_details.py, uncompiled) holds one record per card PSD by
  blob sha (fonts, text layers, effects, smart objects, canvas, icon style, stat-row encoding, FX/overlay layers, template
  deviations, a compact layer tree) for the website's Details dropdown; gathered in /update_stats, pushed with the pickles,
  backfilled by `python -m commands.psd_details --clone`; /fonts reads it first. Owner: autocroissant-run-and-operate §7.
- 2026-10-21: D7 multi-creature stats, by database-only website edits in TTSCardMaker's `site/Card Edits.json`: Twin Emperors'
  2nd creature DEF 10 (98e85f1); Pix and Tainted Lazarus checked against their PNGs, parser notes checked (fa2006d). stats.pkl
  takes them at the next /update_stats.
- 2026-10-21 (push guard, rule 2): `commands/card_check.py` (uncompiled, CLI only) reports a pushed card's parser and validator notes
  (what /update_stats prints; the parser's own notes among them are also the website's Parser notes) from TTSCardMaker's `card-check.yml` (landed 2026-10-21: AutoCroissant df4a439,
  TTSCardMaker a3451c8; the first whole-repo dispatch, run 39690899334, succeeded, job summary only; the font check's webhook secret). Owner: autocroissant-run-and-operate §7.
- 2026-10-22: B8, a deleted PSD stays listed: every full run archives it in old_stats.pkl (tagged `archived`, restorable;
  never deleted, the owner's call) (10.31).
- 2026-10-22/23: D2 (20_Creature_Types' `tornado` layer renamed `wind`, TTSCardMaker 100511d3; 0 pickled problem cards at
  a7c310e); B9's blob half (0166948, 2026-10-21: a failed download or PSD drops the tree ETag, so the next run retries);
  stale `tmp*.psd` left by a cut-off run are deleted at startup (`card_runner.sweep_temp_psds`, c0fc117).
- 2026-10-24/25 (report `docs/Finished/session-2026-10-24/`): B11 fixed in the fork (74a65e2b, round 12); B9 moot since the
  cutover (the legacy tree read goes with data build step 10); D3, D8 and V6 re-parsed (the `data` branch has them);
  D7 (BUG 2) landed 10-24 (phase 6 48cfd94; `work/badgealt` replaces the badge FX layers); V2 and V5 verified.
- 2026-10-25 (work/bakebugs, archaeology 10.34): B12, the bake follows a layer's Txt2 kerning; text typed beside a
  coloured word takes the body colour (Binge Eater); a whole-ability edit and a part edit keep the characters
  `normalize_ability` strips (Deepnest, Blasphemous Blade); an Ability past its panel is set smaller to fit (Autopsy Report).

---

## 1. Bugs

### 1.1 OPEN — high priority

None open (2026-10-25: B12 and the plan's three bake bugs fixed, archaeology 10.34).

### 1.2 OPEN — data debt (as of 2026-10-19)

| # | Issue | Evidence | Suggested action | Detail owner |
|---|-------|----------|------------------|--------------|
| D1 | **4 latent problem cards invisible in the pickled count.** stats.pkl stores only parse-time problems (count: 1); full re-validation reports 5: Computer Virus (all stats missing), Anubisath Guardian, Qiraji Soldier, Silithid (blank ability layers, June 2026 WoW batch). 2026-10-21: one card now: the user put Anubisath Guardian, Qiraji Soldier and Silithid into `ABILITY_EXCLUSIONS` (e4e0a58, 21 names); Computer Virus (a creature with no stat rows) remains. | Re-validation sweep, `autocroissant-validation-and-qa` baselines | Owner 2026-10-25: a card with no ability is no problem (the parts editor drops the check); left: Computer Virus's stats, in its PSD or certified | `autocroissant-validation-and-qa` §baselines |
| D4 | **A second, never-reverted old_stats balloon** (`637698b`, 2025-11-25): ~26 unchanged cards breadth-archived in one sweep still pollute old_stats.pkl history. | Byte trace in archaeology; snapshot forensics | 2026-10-22: the first `/update_stats` after the deploy removes the duplicates itself (`card_history.dedupe`, since 2026-10-23 inside `card_data.clean_history`; 475 -> 460 versions on cb8fe54's pickles, starts carried in `version`). Check that PICKLE with `diff_stats.py --history` (2026-10-23: it diffs old_stats.pkl too: versions dropped or added, starts moved, archive tags). The website dates versions by that record (`analytics.when`, TTSCardMaker work/version-dates, landed 2026-10-22 with c0b6f2d7). Owner notes: a card back after a deletion gap with the same data keeps its first start (as restore() does), so as_of and the site's over-time count it alive through the gap; a run folded into the current card can leave a later archived record before it (no row shows; nothing new) | `autocroissant-failure-archaeology` Entry 3 |
| D6 | **Icon-font adoption (re-measured 2026-10-23).** `psd_details.json` at the 05:00 run (813 PSD records, before batch B2's ~219 re-parses): 275 PSDs have text in stoogedings; the `icons` field reads tokens 45, mixed 230, images 470, none 68. The 470 `images` cards (and the image half of `mixed`) still depend on gap injection. (2026-10-19: 0 of 813.) | `psd_details.json` `icons` per record | Keep migrating in TTSCardMaker as cards are touched (the standardization batches: 242 cards on 2026-10-22, queue A, B2 7fd2aa30 on 2026-10-23); `icons` is the measure | `autocroissant-psd-extraction-campaign` |

### 1.3 OPEN — docs and cosmetics

None open (2026-10-20). C1-C3, C5, C6, C12 and C14 are retired above.

### 1.5 Pending live verification (as of 2026-10-19)

| # | Item | What was verified | What is still missing | Detail owner |
|---|------|-------------------|-----------------------|--------------|
| V1 | Startup emoji sync | Live on the jetson 2026-10-20: "126 icons ..., 0 uploaded, 126 ready" | `/sync_emojis refresh:True` not run live yet | `autocroissant-run-and-operate` §11 |
| V4 | `/query` name lookup: Card Edits.json names and aliases never joined the registry (the lookup used a mixed-case key against lowercase keys, since f2013eb); fixed 2026-10-21 in `query_card._add_lookup`, which now also adds printed titles (`_populate_titles`) | Tests `test_website_names_join_the_name_lookup`, `test_titles_join_the_name_lookup` | `/query Raw Kawk`, `/query Black Clot` (after the re-parse) on the live bot | `autocroissant-run-and-operate` §7 |

### 1.6 Companion repo (TTSCardMaker) items that affect the bot

Owned by `ttscardmaker-companion`:
- The website and editor are live (pushed 2026-10-20); the bot reads `site/Card Authors.txt` and `site/Card Edits.json` (moved there from `MDW/` the same day, TTSCardMaker 13ac680; the `MDW/` fallback is dropped).
- The website reads the `data` branch since the 2026-10-24 cutover; it no longer unpickles the bot's pickles.
- No font is missing from `MDW/Fonts` (was 7): the user added the Windows/Office faces on 2026-10-21 (TTSCardMaker a88b232) and
  BabaUI (`Baba.psd`) on 2026-10-23 (92b5ae6b, the owner's file `Baba UI Regular.ttf`). `font_check --all` over the clone:
  812 card PSDs, `MDW/Fonts` 274 files / 282 faces, nothing missing.

---

## 2. Future improvements

Owner's stated ambitions (2026-07-11): **perfect extraction**, **AI that knows the game**, and
"it's a hobby — keep it fun." Everything below is CANDIDATE — measured milestones, no promises.

### 2.1 Perfect extraction (the active campaign)

Executable plan with gates and expected numbers: `.claude/skills/autocroissant-psd-extraction-campaign/SKILL.md`.
Ranked menu:
1. **Turn the 2026-10-19 regression checks into a reusable harness.** The `relpath` fix itself is done (Phase 0,
   2026-10-19). What's left is scripting the two checks that proved it: old parser vs new parser on every PSD, and
   a sandboxed forced local sweep (`update_stats_local()`, on branch `archive/local-update-stats`) diffed against the
   stored data. Procedure: `autocroissant-validation-and-qa` §4.1.
2. **Migrate inline-icon cards to stoogedings tokens (D6).** A card with tokens in the font needs no gap
   guessing at all. This is the most direct route to perfect inline types, and it is card-content work in
   TTSCardMaker (`ttscardmaker-companion`).
3. **Restore gap/type mismatch detection as a high-precision validator check** — it existed (`fed8a83`) and thrashed (`081b1fd`); the precision conditions are derived in the campaign.
4. **Bbox-anchored type injection** — place `[type]` by icon coordinates instead of gap ordinals; the line-position estimation sub-problem is genuinely open. Matters only for cards not migrated by item 2.
5. **Exclusion-list burn-down** — 21 cards (ability; the stat list went on 2026-10-21: `allow_high_stats` per card). Owner 2026-10-25: cards need neither types nor an ability, so the parts editor drops ABILITY TEXT NOT FOUND (`psd_parser.py` ~1130) and the list goes to 0 (plan, queue 1).

### 2.2 AI that knows the game (on branch `archive/ai-generation`)

Main has no AI code since 2026-10-20; image generation lives on the branch. Any AI feature below would start there.

Steps and milestones: `.claude/skills/autocroissant-research-frontier/SKILL.md`.
1. **Game-grounded Q&A (`/ask`)** — corpus already exportable (48 rulebook pages + 813 structured cards); build the 20-question eval set FIRST, then a zero-dependency keyword baseline, only then heavier retrieval. Result when: ≥15/20 eval answers correct with citations. No LLM/embedding dep may enter core `requirements.txt`.
2. **In-style card art LoRA** — 1003 PNGs organized by series (Kirby 58 is the best corpus); train offline, drop into `models/loras/`, blind A/B with the friend group. Result when: ≥40% of generated cards mistaken for real in a shuffled test.
3. **Round-trip extraction verification** — re-render extracted text and image-diff against the card; single-card feasibility prototype first.

### 2.3 Operational hardening (small, high-value)

Done 2026-10-20 (B4, B5, B7, C1-C3, C5, C6, C12: failure-archaeology 10.20-10.26); B10 2026-10-21 (10.29), B11 2026-10-21, B8 2026-10-22 (10.31). B9 moot since the cutover.
- **Fonts on the jetson (done 2026-10-21; numbered B11 then, an id the fork bug reused; retired 2026-10-25).** The
  `fonts_cache/` fallback (`font_check.refresh_cache`) served the jetson's bake and details gather; both moved to the
  Actions (`PSD_TOOLS_FONT_DIRS`), so the cache was removed on 2026-10-25 (work/clean). `/fonts` reads the card data.

### 2.4 Planned next (docs/NEXT_SESSION_PLAN.md)

The ordered list is `docs/NEXT_SESSION_PLAN.md` (2026-10-25). PSD edits from website edits are live: since the
2026-10-24 cutover Card sync (card-sync.yml) bakes them after a card push's 30-minute window (card-edits.yml retired).

Done 2026-10-20: commit-sha image links (Task 1, e13d754), emoji sync inside /update_stats with the weekly
reminder removed (Task 1b, e13d754), TTS save automation and /update_tts_save (Task 2: bad0496, ce94585, c4caaf9; the automatic run moved to TTSCardMaker's tts-save.yml the same day),
the website Table tab (Task 2b, TTSCardMaker 0d49893; reverted the same day at the user's request, 75692fb).

### 2.5 Deep optimization and refinement pass (all code)

One pass over every module, measured before and after, with no behavior change unless listed:
- **Network and I/O:** done 2026-10-20 (10.28). One commit page per changed card (2 requests -> 1; a forced sweep
  2,714 -> 1,810 over all 904 tracked PSDs, about 2,450 -> 1,630 over the 813 it parses), lookups beside the
  downloads (6 PSDs 16-18 s -> 6.5-8.7 s), the tree ETag shared by /update_stats, the card list and the emoji
  listing (startup 200 + 200 -> 200 + 304), TTS save objects cached by sha (repeat run 9 requests / 6.4 MB -> 3 / 0 MB).
- **Speed:** done 2026-10-20 (10.28). Fuzzy step x18 on short ability queries (1138 -> 63 ms for 60), exact names
  skip difflib, lazy cv2/yt_dlp/PyGithub (`import main` ~1.1 s / 185 MB -> ~0.7 s / 123 MB). Measured and left alone:
  pandas copies (prep_dataframes never runs per query), idle cost (0.33 ms CPU/s), Cythonising the parser (its own
  logic is 1% of a parse).
- **Robustness:** done 2026-10-20: B4, B5, B7, C6; every `tasks.loop` survives exceptions (per-item try + log);
  every requests call had a timeout already, git fetch/pull/push now get 120 s; temp files removed in `finally`
  (partial PSD downloads, the emoji/history/TTS-save `.tmp` files). Tests: `tests/test_robustness.py`.
- **Code health:** done 2026-10-20. Dead code deleted (`analytics.parse_time_str`, `get_old_stats_as_dict`,
  `CardInfo.from_dict`, two unread FakeInteraction attributes); GitHub access in `commands/github_api.py` (token, headers,
  tree/blob fetches, retries); repo paths in `global_config.py`; `utils.say`/`queue_long_message`, `card_name`,
  `repo_history.pinned_ref`; `perms_check` -> `lacks_admin`, `dirty_files` -> `seen_cards`, `num_new` -> `num_changed`;
  `psd_analyzer.py` (2,347 lines) split into `card_models` / `card_edits` / `psd_parser` / `repo_traverser` behind the
  `psd_analyzer` facade (CardInfo/CardStats `__module__` pinned, so the pickles are unchanged; 11 compiled modules);
  `CardClassifier.classify` no longer reads out of bounds on a shallow path (compiled: SIGSEGV). Gates: 813 cards,
  0 differences on .py and compiled; pickle re-dump byte-identical; `tests/test_split_contract.py`.
- **Concurrency:** done 2026-10-21 (10.29): one lock per pickle and atomic saves (B10); GitPython calls on the event
  loop time out after 120 s.
- **Tests:** done 2026-10-20: `tests/` (350 passed, 18 skipped on 2026-10-21 after pass 2, about 31 s, offline; 38 parser
  goldens), run before every push (autocroissant-change-control).
- **Docs:** done 2026-10-20: help.py and the describe strings match all 50 commands (10.20); skill line references
  re-mapped after the split (`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/optimization/refs6.py`).
- **Owner decisions, done 2026-10-21 (10.30):** the queue tick is 0.1 s (reply latency up to ~100 ms per hop instead
  of ~1 s; reminders keep their 1 s heartbeat; idle 1.3 ms CPU/s), and /query_ability listed name and image link 10 per
  message instead of one message per result until 2026-10-23, when the owner went back to one message per card (Discord
  previews only a few images per message). Since 2026-10-25 two or more cards come as grid images, 20 per message
  (commands/card_grid.py; 'damage' 188 sends → 10). Tests: `tests/test_query_output.py`.
- **B-1, done 2026-10-25 (branch work/instant):** the drains wake on enqueue (1 s fallback poll); reply hop about 1 ms
  instead of ~50 ms (archaeology 10.30, autocroissant-architecture-contract).

- **Pass 2 (the edit pipeline, font check, site badges), done 2026-10-21:** audits and numbers in
  `docs/Finished/optimization-pass-2-2026-10-21/`. One commit path (`commands/git_commit.py`: API blob/tree/commit, SSH
  fallback, `cant_write`/`limited`) shared by `card_bake` and `tts_save` (card_bake 966 -> 777 lines, tts_save
  1022 -> 891); `github_api.token_from_env`/`blob_sha`/`fetch_verified`; `font_check --all` in a process pool
  (98 s -> 9.8 s, outputs byte-identical); dead code out of `psd_regions`/`psd_edit` (`region_masks`, `_union`, KIND
  fallback), `fx_pad`/`union_box` public, Args/Returns on the gates, truthful `__all__`; the website's `valid_verdict`
  accepts only `baked`/`refused`. 13 new tests (SSH fallback, 409, verdict skipped when the edit changed, Royal
  Eradicator's two-row atk, the `font` refusal on every fork run, import allowlist, workflow shapes).

Left over from that pass, all done since (audits on `docs/Finished/optimization-pass-2026-10-20/`):
- Done since (2026-10-21, 778c2ce/b6ddf41/171c84b/0166948): a repeat `on_ready` runs no init and revives only failed loops;
  `GIT_TERMINAL_PROMPT=0` / SSH BatchMode for the git calls; a failed blob no longer records the tree ETag (B9's blob half)
  (network-robustness.md items 8, 4, 5).
- Done 2026-10-21: lazy GitPython in update_bot; a `requests.Session` per download thread (github_api `_local`).
- Done 2026-10-22: `/purge`, `/update_metadata`, `/mass_replace`, `/alias`, `/set_repo`, `/delete_song` and
  `/delete_all_music` are admin only (tests/test_admin_gates.py; docs.md §1.6). The three card commands were
  retired 2026-10-25 (6e8f8e3).

Not done in pass 2 (2026-10-21; `docs/Finished/optimization-pass-2-2026-10-21/README.md` lists the rest):
- Done since: `save(preview=)` brought a bake from 23 s to 1.8 s (2026-10-22); the fork venv has `requests` and
  `pyyaml`, so the bake and workflow-shape tests run there.
- Website build: cache the accent colour by blob hash (-2.9 s of a 5.3 s warm build) and hardlink the dist copies
  (perf.md 5).

### 2.6 Explicitly parked

- **Replacing pickles-in-git** — done otherwise: the `data` branch replaced them at the 2026-10-24 cutover; data build
  step 10 deletes them.

---

## Maintenance

- Re-verify the OPEN bugs before working on them (one-liners in each owning skill's
  "Provenance and maintenance" section); line numbers drift.
- Library note: eight skill frontmatter descriptions contain unquoted `word: word` constructs (2026-10-20: analysis-toolkit,
  change-control, debugging-playbook, diagnostics-and-tooling, failure-archaeology, research-frontier, validation-and-qa,
  ttscardmaker-companion) — fine for the current lenient loader, need quoting if a strict YAML parser is ever adopted.
  Descriptions over about 1,500 characters are cut off in the skill listing; four were trimmed on 2026-10-20.
- This file is an index, not a second home for the stories. Keep entries to a few lines; link the owner skill.
