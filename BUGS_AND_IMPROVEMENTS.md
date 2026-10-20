# AutoCroissant — Known Bugs and Future Improvements

Compiled 2026-07-11 from a full repo audit; revised 2026-10-19 and 2026-10-20. Detail, evidence and procedures
live in the skill library under `.claude/skills/`; each entry points at its owner (`ttscardmaker-companion`
covers TTSCardMaker: icon font, website, fonts, Photoshop scripting). Fixes route through
`autocroissant-change-control`. When an entry is fixed or retired, move its story to
`autocroissant-failure-archaeology` and delete the row here. The next session's plan is `NEXT_SESSION_PLAN.md`.

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

---

## 1. Bugs

### 1.1 OPEN — high priority

| # | Bug | Where | Impact | Workaround / candidate fix | Detail owner |
|---|-----|-------|--------|---------------------------|--------------|
| B4 | **`init_vc` crashes when the caller is not in a voice channel.** `await queue_message(...)` awaits a plain function's `None` return → `TypeError` instead of the intended "You are not in a voice channel." message. | `commands/music_player.py` (~line 177) | `/play` from outside a voice channel errors instead of replying | Candidate fix: drop the `await` (queue_message is sync) | `autocroissant-debugging-playbook` §music |
| B5 | **The dispatch loop can die permanently.** Exceptions outside the caught set stop a `tasks.loop` for good: (a) deleted/unavailable channel → `client.get_channel()` returns `None` → `AttributeError` **inside the fallback**; (b) >2000-char content that fails in the fallback path too. After that, the whole bot goes silent (all sends flow through this loop). **Partly mitigated 2026-10-19:** reminders now skip channels the bot can't see (`_channel_missing` in analytics.py), so the scheduler no longer feeds the loop a dead channel; every other sender still can, and `process_dispatch_queue`/`process_edit_queue` in main.py are unchanged. | `main.py` (`process_dispatch_queue`, fallback `client.get_channel(...).send(...)`) | One poisoned message silences the entire bot until restart | Candidate fix: wrap each drain iteration in `try/except` + log; guard `get_channel()` None | `autocroissant-architecture-contract` §known-weak points |
| B7 | **A GitHub 5xx at startup leaves the bot without its card list.** `populate_files` calls `response.json()` on the error page (JSONDecodeError in `run_command`) and the emoji sync fails; /query finds nothing until the next /update_stats. Seen 2026-10-20 on the jetson; a restart fixed it. | `commands/query_card.py` `populate_files`, `commands/type_emojis.py` | Cards missing from /query after an unlucky restart | Check the status, retry with backoff, keep the last good tree | `autocroissant-debugging-playbook` |

### 1.2 OPEN — data debt (as of 2026-10-19)

| # | Issue | Evidence | Suggested action | Detail owner |
|---|-------|----------|------------------|--------------|
| D1 | **4 latent problem cards invisible in the pickled count.** stats.pkl stores only parse-time problems (count: 1); full re-validation reports 5: Computer Virus (all stats missing), Anubisath Guardian, Qiraji Soldier, Silithid (blank ability layers, June 2026 WoW batch). | Re-validation sweep, `autocroissant-validation-and-qa` baselines | Fix the four PSDs in TTSCardMaker (card content, not bot code) or certify as intentional | `autocroissant-validation-and-qa` §baselines |
| D2 | **"20 Creature Types" rulebook page has a `tornado` misspelling** (layer name in `MISSPELT_CARD_TYPES`). The single pickled problem card. | `inspect_pickle.py --problems` | Fix the PSD layer name in TTSCardMaker | `impossibility-cards-reference` §validation |
| D3 | **Stale exclusion: Shadow Duelist** sits in `ABILITY_EXCLUSIONS` but now has ability text. | Campaign burn-down census | Remove from the exclusion set after a golden-checked reparse | `autocroissant-psd-extraction-campaign` Phase 3.4 |
| D4 | **A second, never-reverted old_stats balloon** (`637698b`, 2025-11-25): ~26 unchanged cards breadth-archived in one sweep still pollute old_stats.pkl history. | Byte trace in archaeology; snapshot forensics | Optional cleanup; low value, only with a diff_stats-gated PICKLE commit | `autocroissant-failure-archaeology` Entry 3 |
| D7 | **Multi-creature stats are a parser guess.** Pix, Twin Emperors and Tainted Lazarus carry `SEVERAL CREATURES`; Twin Emperors' 2nd DEF reads 4 but the card shows 10 (an empty row next to a shared one is ambiguous). | `StatTrackers.creatures` | Fix per creature on the website, then tick "parser notes checked" | `impossibility-cards-reference` §3.4 |
| D6 | **No card uses the stoogedings font yet (0 of 813 PSDs).** About 45 PSDs (81 smart-object icon layers below the midline) still use image-layer icons, so they still depend on gap injection. Counting any layer kind gives 61 PSDs / 148 layers, some of them decoration rather than inline icons. | TTSCardMaker migration audit (2026-10-19) | Migrate them in TTSCardMaker (card content). The JSX prototype and its overlay-layer caveat are in the companion skill | `ttscardmaker-companion` |

### 1.3 OPEN — docs and cosmetics

| # | Issue | Where |
|---|-------|-------|
| C1 | `/pull` help + description claim "hard reset, then a git pull" — the code does fetch+merge (pickles-ours/code-theirs); the hard reset only exists behind `/update force_reset:True`. The main.py description also says "than" (help.py already says "then"). | `commands/help.py:10`, `main.py:166` |
| C2 | Four commands missing from help.py: `/view_old_metadata`, `/list_guild_members`, `/list_guild_channels`, `/get_channel_messages` (50 registered, 46 documented as of 2026-10-20). | `commands/help.py` |
| C3 | `/export_cards only_ability` — help says default True, code default is False. | `commands/help.py:76` vs `main.py:374` |
| C5 | Duplicate function names in main.py (`slash_set_ratio` ×2, `slash_delete_song` ×2). Harmless today (decorators registered at def time) but fragile — rename carefully, never "clean up" casually. | `main.py` |
| C6 | `/list_guild_members` requires the members intent, which `main.py` never enables — the command always fails with the ClientException message. Enable the intent (and in the Discord dev portal) or remove the command. | `main.py` intents block, `commands/management.py` |
| C14 | `/query` and ability-search images use `raw.githubusercontent.com/.../main/...` links, so old Discord messages show the newest image. Use the latest commit sha of the card's PNG instead (planned: NEXT_SESSION_PLAN.md). | `commands/query_card.py` `get_card_url` |
| C12 | Pillow is still not declared in `requirements.txt`. `commands/type_emojis.py:8` is now the only module that imports it at top level, and it arrives only through psd-tools. | `requirements.txt` |

### 1.5 Pending live verification (as of 2026-10-19)

| # | Item | What was verified | What is still missing | Detail owner |
|---|------|-------------------|-----------------------|--------------|
| V1 | Startup emoji sync | Live on the jetson 2026-10-20: "126 icons ..., 0 uploaded, 126 ready" | `/sync_emojis refresh:True` not run live yet | `autocroissant-run-and-operate` §11 |
| V2 | Reminder rewrite (run-as-creator, name:value commands, migration of reminder.pkl) | Code read 2026-10-19 | Not known to have fired against live Discord yet. Watch the first startup's `migrated command` log lines and the first firing | `autocroissant-run-and-operate` §9 |
| V3 | Cython build on a host | Done on the jetson 2026-10-20 (Cython 3.3.0; 8 modules then, 7 since the AI move) | Nothing | `autocroissant-build-and-env` §5 |

### 1.6 Companion repo (TTSCardMaker) items that affect the bot

Owned by `ttscardmaker-companion`:
- The website and editor are live (pushed 2026-10-20); the bot reads `MDW/Card Authors.txt` and `MDW/Card Edits.json`.
- The website unpickles the bot's pickles with stub classes for `commands.*`: anyone who can push AutoCroissant
  `main` can run code in the TTSCardMaker Actions job, and renaming CardInfo attributes blanks website fields.
- 7 fonts are still missing from `MDW/Fonts` (Windows/Office ones, BabaUI).
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
5. **Exclusion-list burn-down** — 23 cards (18 ability + 5 stats); per-card recipe in the campaign; D3 above is the first easy win. Result when: both lists reach zero with problem count stable.

### 2.2 AI that knows the game (on branch `archive/ai-generation`)

Main has no AI code since 2026-10-20; image generation lives on the branch. Any AI feature below would start there.

Steps and milestones: `.claude/skills/autocroissant-research-frontier/SKILL.md`.
1. **Game-grounded Q&A (`/ask`)** — corpus already exportable (48 rulebook pages + 813 structured cards); build the 20-question eval set FIRST, then a zero-dependency keyword baseline, only then heavier retrieval. Result when: ≥15/20 eval answers correct with citations. No LLM/embedding dep may enter core `requirements.txt`.
2. **In-style card art LoRA** — 1003 PNGs organized by series (Kirby 58 is the best corpus); train offline, drop into `models/loras/`, blind A/B with the friend group. Result when: ≥40% of generated cards mistaken for real in a shuffled test.
3. **Round-trip extraction verification** — re-render extracted text and image-diff against the card; single-card feasibility prototype first.

### 2.3 Operational hardening (small, high-value)

1. A defensive dispatch loop (B5): the single point of silence.
2. Fix B4's `await` and C6's intent mismatch.
3. Sync help.py with reality (C1–C3) in one docs pass — checklist in `.claude/skills/autocroissant-docs-and-style/SKILL.md`.
4. Small cleanup: C12 (declare Pillow).

### 2.4 Planned next (NEXT_SESSION_PLAN.md; Task 3 first)

1. PSD edits from website edits with the psd-tools fork, no Photoshop or browser (Task 3, the main effort; plan
   decision D4). Photopea is the reference renderer and emergency fallback until the fork meets its targets. Runs
   on GitHub Actions, else on the jetson once it has Python 3.11+ (Task 0, in parallel).
2. Commit-sha image links for `/query` (C14; Task 1).
3. Emoji sync inside `/update_stats`: upload or replace only icons whose blob changed, then drop the weekly
   `/sync_emojis` reminder (Task 1b).
4. TTS save automation (Task 2): audit the save against the repo, then a script (and a command) that adds new cards
   to their decks, fills in names, pins image links to commit shas only when a card changed, and updates the global
   Lua's large-card list.
5. A website "Table" tab to view and edit the save (Task 2b).

### 2.5 Deep optimization and refinement pass (all code)

One pass over every module, measured before and after, with no behavior change unless listed:
- **Network and I/O:** count requests per command; conditional requests and caches everywhere (done for
  /update_stats); no repeated downloads; streaming for large files.
- **Speed:** profile startup, /update_stats, searches and reminders; cache parsed data; avoid pandas copies in
  `ability_search_engine`; check Cython-friendliness of hot loops.
- **Robustness:** B4, B5 and C6; every `tasks.loop` survives exceptions; timeouts on all HTTP calls; temp files
  always cleaned.
- **Code health:** dead code and duplicate helpers (vulture, grep); consistent naming and types; split
  `psd_analyzer.py` (2,000+ lines) into parser / traverser / edits / models; one place for repo-path constants.
- **Concurrency:** confirm every blocking call runs off the event loop; one lock per pickle.
- **Tests:** a small offline suite (parser goldens on ~30 PSDs, CardEdits, duplicates, creature split,
  update_stats with a fake tree) run before every push.
- **Docs:** help.py matches the commands (C1-C3); skills' line numbers refreshed.

### 2.6 Explicitly parked

- **Replacing pickles-in-git for state sync** — current mechanism works; any replacement must beat: zero-loss handoff both directions, no new infrastructure to babysit. "Do nothing" is the standing choice. (`autocroissant-research-frontier` §4)

---

## Maintenance

- Re-verify the OPEN bugs before working on them (one-liners in each owning skill's
  "Provenance and maintenance" section); line numbers drift.
- Library note: eight skill frontmatter descriptions contain unquoted `word: word` constructs —
  fine for the current lenient loader, need quoting if a strict YAML parser is ever adopted.
- This file is an index, not a second home for the stories. Keep entries to a few lines; link the owner skill.
