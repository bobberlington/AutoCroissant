# AutoCroissant — Known Bugs and Future Improvements

Compiled 2026-07-11 from a full repo audit; revised 2026-10-19, 2026-10-20 and 2026-10-21. Detail, evidence and procedures
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
  output (10.30).
- 2026-10-21 optimization pass 2: /query_ability's single-result hole (one card without a PNG answered `content=None`,
  which Discord dropped; it posts the name now) and the /fonts lazy-import shadowing; one commit path for the bake and
  the TTS save (`commands/git_commit.py`); the font check parses in a process pool (813 PSDs 98 s -> 9.8 s). Report:
  `docs/Finished/optimization-pass-2-2026-10-21/`.
- 2026-10-21 (plan item 23, bot side): `psd_details.json` (commands/psd_details.py, uncompiled) holds one record per card PSD by
  blob sha (fonts, text layers, effects, smart objects, canvas, icon style, stat-row encoding, FX/overlay layers, template
  deviations, a compact layer tree) for the website's Details dropdown; gathered in /update_stats, pushed with the pickles,
  backfilled by `python -m commands.psd_details --clone`; /fonts reads it first. Owner: autocroissant-run-and-operate §7.
- 2026-10-21 (push guard, rule 2): `commands/card_check.py` (uncompiled, CLI only) reports a pushed card's parser and validator notes
  (what /update_stats prints; the parser's own notes among them are also the website's Parser notes) from TTSCardMaker's `card-check.yml` (landed 2026-10-21: AutoCroissant df4a439,
  TTSCardMaker a3451c8; the first whole-repo dispatch, run 39690899334, succeeded, job summary only; the font check's webhook secret). Owner: autocroissant-run-and-operate §7.

---

## 1. Bugs

### 1.1 OPEN — high priority

| # | Bug | Where | Impact | Workaround / candidate fix | Detail owner |
|---|-----|-------|--------|---------------------------|--------------|
| B8 | **A deleted PSD stays in stats.pkl.** `update_stats` calls `prune_clean_cards()` only when `num_changed > 0`, so a deletion alone leaves the card listed until some other card changes. Pinned by `tests/test_update_stats.py::test_deleted_card_alone_is_kept` (update it with the fix). | `commands/repo_traverser.py` `update_stats` | Removed cards still answer /query | Also prune when the tree lost a path | `autocroissant-architecture-contract` |
| B9 | **A GitHub 5xx during /update_stats.** A tree 5xx raises `HTTPError` out of `update_stats`; a blob 5xx skips the card ("download failed") but the tree ETag is still recorded, so the next runs get a 304 and the card keeps its old parse until the card repo changes again. Pinned by `test_tree_5xx_raises_and_saves_nothing` and `test_blob_5xx_skips_the_card`. | `commands/repo_traverser.py` `traverse_remote` | A card silently stays stale | Don't record the ETag when a blob failed; retry with backoff (with B7) | `autocroissant-architecture-contract` |

### 1.2 OPEN — data debt (as of 2026-10-19)

| # | Issue | Evidence | Suggested action | Detail owner |
|---|-------|----------|------------------|--------------|
| D1 | **4 latent problem cards invisible in the pickled count.** stats.pkl stores only parse-time problems (count: 1); full re-validation reports 5: Computer Virus (all stats missing), Anubisath Guardian, Qiraji Soldier, Silithid (blank ability layers, June 2026 WoW batch). 2026-10-21: one card now: the user put Anubisath Guardian, Qiraji Soldier and Silithid into `ABILITY_EXCLUSIONS` (e4e0a58, 21 names); Computer Virus (a creature with no stat rows) remains. | Re-validation sweep, `autocroissant-validation-and-qa` baselines | Fix the four PSDs in TTSCardMaker (card content, not bot code) or certify as intentional | `autocroissant-validation-and-qa` §baselines |
| D2 | **"20 Creature Types" rulebook page has a `tornado` misspelling** (layer name in `MISSPELT_CARD_TYPES`; the user, 2026-10-21: it is supposed to be `wind`). The single pickled problem card; waits for the PSD hold. | `inspect_pickle.py --problems` | Fix the PSD layer name in TTSCardMaker | `impossibility-cards-reference` §validation |
| D3 | **Stale exclusion: Shadow Duelist** sits in `ABILITY_EXCLUSIONS` but now has ability text. | Campaign burn-down census | Remove from the exclusion set after a golden-checked reparse | `autocroissant-psd-extraction-campaign` Phase 3.4 |
| D4 | **A second, never-reverted old_stats balloon** (`637698b`, 2025-11-25): ~26 unchanged cards breadth-archived in one sweep still pollute old_stats.pkl history. | Byte trace in archaeology; snapshot forensics | Optional cleanup; low value, only with a diff_stats-gated PICKLE commit | `autocroissant-failure-archaeology` Entry 3 |
| D7 | **Multi-creature stats are a parser guess.** Pix, Twin Emperors and Tainted Lazarus carry `SEVERAL CREATURES`; Twin Emperors' 2nd DEF reads 4 but the card shows 10 (an empty row next to a shared one is ambiguous). | `StatTrackers.creatures` | Fix per creature on the website, then tick "parser notes checked". 2026-10-21: doable now under the PSD hold: `creatures` is a database-only website edit (`site/Card Edits.json`), not a PSD write | `impossibility-cards-reference` §3.4 |
| D6 | **No card uses the stoogedings font yet (0 of 813 PSDs).** About 45 PSDs (81 smart-object icon layers below the midline) still use image-layer icons, so they still depend on gap injection. Counting any layer kind gives 61 PSDs / 148 layers, some of them decoration rather than inline icons. | TTSCardMaker migration audit (2026-10-19) | Migrate them in TTSCardMaker (card content). The JSX prototype and its overlay-layer caveat are in the companion skill | `ttscardmaker-companion` |

### 1.3 OPEN — docs and cosmetics

None open (2026-10-20). C1-C3, C5, C6, C12 and C14 are retired above.

### 1.5 Pending live verification (as of 2026-10-19)

| # | Item | What was verified | What is still missing | Detail owner |
|---|------|-------------------|-----------------------|--------------|
| V1 | Startup emoji sync | Live on the jetson 2026-10-20: "126 icons ..., 0 uploaded, 126 ready" | `/sync_emojis refresh:True` not run live yet | `autocroissant-run-and-operate` §11 |
| V2 | Reminder rewrite (run-as-creator, name:value commands, migration of reminder.pkl) | Code read 2026-10-19; live 2026-10-21: the /push reminder fired at 05:30 (PICKLE 56b43b9) | Nothing: the rewrite is verified live (was: not known to have fired against live Discord yet; watch the first startup's `migrated command` log lines and the first firing) | `autocroissant-run-and-operate` §9 |
| V4 | `/query` name lookup: Card Edits.json names and aliases never joined the registry (the lookup used a mixed-case key against lowercase keys, since f2013eb); fixed 2026-10-21 in `query_card._add_lookup`, which now also adds printed titles (`_populate_titles`) | Tests `test_website_names_join_the_name_lookup`, `test_titles_join_the_name_lookup` | `/query Raw Kawk`, `/query Black Clot` (after the re-parse) on the live bot | `autocroissant-run-and-operate` §7 |
| V5 | `PARSER_VERSION` 2: the printed title read into `display_name`; `EXCESSIVE_STAT_EXCLUSIONS` gone (2026-10-21; numbered V3 until the same day, an id the retired Cython row already used) | Parser gate: 813 cards, 159 intended differences (157 titles, 2 STATS TOO HIGH allowed by `allow_high_stats`); suite 356; site built locally from patched pickles; live 2026-10-21: the first re-parse ran 10:18-10:40 and again at 12:16, all 819 entries stamped v2 | Re-parse done 2026-10-21 (see What was verified). Was: the first live `/update_stats` after the deploy must re-parse all ~813 cards once ("813 cards changed or new", minutes, no old_stats growth), then go quiet; the website and the `/query` embed title then show printed titles (94 differ from the file name only by letter case or spacing, CABBAGE, A P E X: shown as printed, the user's call of 2026-10-21; `/query` finds a card by its printed title) | `autocroissant-run-and-operate` §7 |

### 1.6 Companion repo (TTSCardMaker) items that affect the bot

Owned by `ttscardmaker-companion`:
- The website and editor are live (pushed 2026-10-20); the bot reads `site/Card Authors.txt` and `site/Card Edits.json` (moved there from `MDW/` the same day, TTSCardMaker 13ac680; the `MDW/` fallback is dropped).
- The website unpickles the bot's pickles with stub classes for `commands.*`: anyone who can push AutoCroissant
  `main` can run code in the TTSCardMaker Actions job, and renaming CardInfo attributes blanks website fields.
- Only BabaUI (`Baba.psd`) is still missing from `MDW/Fonts` (2026-10-21: the user added the Windows/Office faces in TTSCardMaker a88b232
  and defers BabaUI; was 7).
- Scratch scripts from 2026-10-19/20 live in `/private/tmp/fontexplore/`, which macOS clears on reboot.

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
5. **Exclusion-list burn-down** — 18 cards (ability; the 5-card stat list went on 2026-10-21: `allow_high_stats` in Card Edits.json per card); per-card recipe in the campaign; D3 above is the first easy win. Result when: the list reaches zero with problem count stable.

### 2.2 AI that knows the game (on branch `archive/ai-generation`)

Main has no AI code since 2026-10-20; image generation lives on the branch. Any AI feature below would start there.

Steps and milestones: `.claude/skills/autocroissant-research-frontier/SKILL.md`.
1. **Game-grounded Q&A (`/ask`)** — corpus already exportable (48 rulebook pages + 813 structured cards); build the 20-question eval set FIRST, then a zero-dependency keyword baseline, only then heavier retrieval. Result when: ≥15/20 eval answers correct with citations. No LLM/embedding dep may enter core `requirements.txt`.
2. **In-style card art LoRA** — 1003 PNGs organized by series (Kirby 58 is the best corpus); train offline, drop into `models/loras/`, blind A/B with the friend group. Result when: ≥40% of generated cards mistaken for real in a shuffled test.
3. **Round-trip extraction verification** — re-render extracted text and image-diff against the card; single-card feasibility prototype first.

### 2.3 Operational hardening (small, high-value)

Done 2026-10-20 (B4, B5, B7, C1-C3, C5, C6, C12: failure-archaeology 10.20-10.26); B10 2026-10-21 (10.29). Open: B8, B9.
- **B11 (open, 2026-10-21): fonts on the jetson.** The jetson has no TTSCardMaker checkout, so `card_bake._fonts_dir` falls back
  to `LOCAL_DIR_LOC/MDW/Fonts` (absent) and `/bake_cards` refuses every text edit there with `font`; new `psd_details.json` records
  gathered on the jetson carry `supplied: null` until `/fonts` has filled `font_names.json`. Fix: a font cache keyed by the
  MDW/Fonts tree sha (downloaded once by `/fonts`, reused by the bake and the details gather). Not urgent: the GitHub Action is
  the live bake runner and the Mac backfill carries the supplied flags for all 813 current blobs.

### 2.4 Planned next (docs/NEXT_SESSION_PLAN.md; Task 3 first)

1. PSD edits from website edits with the psd-tools fork, no Photoshop or browser (Task 3, the main effort; plan
   decision D4). Photopea is the reference renderer and emergency fallback until the fork meets its targets. Runs
   on GitHub Actions, else on the jetson (Python 3.14 venv since 2026-10-20; 3.10 removed the same day, Task 0).
   The profile measured `PSDImage.open` at 99% of a parse (4.84 of 4.88 s for 50 PSDs); lazy Txt2 and layer
   effects would cut about 40% (profile.md item 4).

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
  of ~1 s; reminders keep their 1 s heartbeat; idle 1.3 ms CPU/s), and /query_ability lists name and image link 10 per
  message instead of one message per result (one match keeps its image preview). Tests: `tests/test_query_output.py`.

- **Pass 2 (the edit pipeline, font check, site badges), done 2026-10-21:** audits and numbers in
  `docs/Finished/optimization-pass-2-2026-10-21/`. One commit path (`commands/git_commit.py`: API blob/tree/commit, SSH
  fallback, `cant_write`/`limited`) shared by `card_bake` and `tts_save` (card_bake 966 -> 777 lines, tts_save
  1022 -> 891); `github_api.token_from_env`/`blob_sha`/`fetch_verified`; `font_check --all` in a process pool
  (98 s -> 9.8 s, outputs byte-identical); dead code out of `psd_regions`/`psd_edit` (`region_masks`, `_union`, KIND
  fallback), `fx_pad`/`union_box` public, Args/Returns on the gates, truthful `__all__`; the website's `valid_verdict`
  accepts only `baked`/`refused`. 13 new tests (SSH fallback, 409, verdict skipped when the edit changed, Royal
  Eradicator's two-row atk, the `font` refusal on every fork run, import allowlist, workflow shapes).

Not done in this pass (later; audits on `docs/Finished/optimization-pass-2026-10-20/`):
- A repeat `on_ready` after a failed RESUME raises in `start()` and re-runs the inits;
  `GIT_TERMINAL_PROMPT=0` / SSH BatchMode for the git calls; a failed blob still records the tree ETag (B9)
  (network-robustness.md items 8, 4, 5).
- Lazy GitPython in update_bot; a `requests.Session` per download thread (network-speed-results.md).
- `/purge`, `/update_metadata`, `/mass_replace`, `/alias`, `/set_repo`, `/delete_song` and `/delete_all_music` have
  no admin gate; with `/purge` any member can delete another user's messages if the bot has Manage Messages
  (docs.md §1.6).

Not done in pass 2 (2026-10-21; `docs/Finished/optimization-pass-2-2026-10-21/README.md` lists the rest):
- A bake is 23 s per card and 95 % of it is fork compositing: write the merged PNG as the PSD preview
  (`PSDImage.save(preview=...)`, -9 s) and crop effects to the viewport (-10 s). Fork-side, after the rendering
  worktrees are free (perf.md 1, 2).
- `tests/test_card_bake.py` cannot collect under the fork venv (no `requests`, no pip there), so the end-to-end bake test
  runs nowhere on the Mac; `pyyaml` is listed in `requirements-dev.txt` but not installed, so the two workflow-shape
  tests skip (tests.md 1, 8).
- Website build: cache the accent colour by blob hash (-2.9 s of a 5.3 s warm build) and hardlink the dist copies
  (perf.md 5).

### 2.6 Explicitly parked

- **Replacing pickles-in-git for state sync** — current mechanism works; any replacement must beat: zero-loss handoff both directions, no new infrastructure to babysit. "Do nothing" is the standing choice. (`autocroissant-research-frontier` §4)

---

## Maintenance

- Re-verify the OPEN bugs before working on them (one-liners in each owning skill's
  "Provenance and maintenance" section); line numbers drift.
- Library note: eight skill frontmatter descriptions contain unquoted `word: word` constructs (2026-10-20: analysis-toolkit,
  change-control, debugging-playbook, diagnostics-and-tooling, failure-archaeology, research-frontier, validation-and-qa,
  ttscardmaker-companion) — fine for the current lenient loader, need quoting if a strict YAML parser is ever adopted.
  Descriptions over about 1,500 characters are cut off in the skill listing; four were trimmed on 2026-10-20.
- This file is an index, not a second home for the stories. Keep entries to a few lines; link the owner skill.
