# AutoCroissant — Known Bugs and Future Improvements

Compiled 2026-07-11 from a full repo audit (code, git history, live data). Detail, evidence, and
procedures live in the skill library under `.claude/skills/` — each entry points at its owner.
Fixes route through `.claude/skills/autocroissant-change-control/SKILL.md`; nothing here is
license to hot-patch. When an entry is fixed or retired, move its story to
`autocroissant-failure-archaeology` and delete the row here.

---

## 1. Bugs

### 1.1 OPEN — high priority

| # | Bug | Where | Impact | Workaround / candidate fix | Detail owner |
|---|-----|-------|--------|---------------------------|--------------|
| B1 | **Local-mode `/update_stats` computes broken relative paths.** `removeprefix("TTSCardMaker")` is a no-op on the absolute paths `os.walk` yields, so every card classifies as UNKNOWN ("Users" top folder) and mass path-change archiving corrupts stats.pkl in one run. Introduced by the cythonization commit `f7c915c` (2025-12-02), which replaced the working `split("TTSCardMaker")[-1]`. `use_local_repo:True` is the slash default. | `commands/psd_analyzer.py` (`_process_local_files`, `relative_path = full_path.removeprefix(...)`) | One default-flag run corrupts the card database | **Always pass `use_local_repo:False`.** Candidate fix: `os.path.relpath(full_path, local_path)` — specced in the campaign skill Phase 0, not yet applied | `autocroissant-failure-archaeology` (story), `autocroissant-psd-extraction-campaign` Phase 0 (fix plan) |
| B2 | **B1 amplifier: scheduled reminders inherit the broken default.** A reminder created with `command:"/update_stats"` (no args) runs local mode unattended. | `commands/analytics.py` (`check_reminder` → `slash_registry`) | Corruption can trigger on a schedule with nobody watching | Any scheduled update must spell out `use_local_repo:False`; audit existing reminders with `/list_reminders all:True` | `autocroissant-run-and-operate` §reminders |
| B3 | **`/ai` wedges permanently after any mid-generation exception.** `in_progress = True` is set with no `try/finally`; an exception (bad URL, OOM, model misconfig) leaves it stuck and every later `/ai` only prints "Request queued." until restart. | `commands/diffusion.py` (`diffusion()`, `in_progress` set ~line 681, cleared ~line 780) | AI capability silently dead until bot restart | Restart bot to clear; candidate fix: wrap generation in `try/finally: in_progress = False` + drain queue | `autocroissant-debugging-playbook` §hangs |
| B4 | **`init_vc` crashes when the caller is not in a voice channel.** `await queue_message(...)` awaits a plain function's `None` return → `TypeError` instead of the intended "You are not in a voice channel." message. | `commands/music_player.py` (~line 177) | `/play` from outside a voice channel errors instead of replying | Candidate fix: drop the `await` (queue_message is sync) | `autocroissant-debugging-playbook` §music |
| B5 | **The dispatch loop can die permanently.** Exceptions outside the caught set stop a `tasks.loop` for good: (a) deleted/unavailable channel → `client.get_channel()` returns `None` → `AttributeError` **inside the fallback**; (b) >2000-char content that fails in the fallback path too. After that, the whole bot goes silent (all sends flow through this loop). | `main.py` (`process_dispatch_queue`, fallback `client.get_channel(...).send(...)`) | One poisoned message silences the entire bot until restart | Candidate fix: wrap each drain iteration in `try/except` + log; guard `get_channel()` None | `autocroissant-architecture-contract` §known-weak points |

### 1.2 OPEN — data debt (as of 2026-07-11)

| # | Issue | Evidence | Suggested action | Detail owner |
|---|-------|----------|------------------|--------------|
| D1 | **4 latent problem cards invisible in the pickled count.** stats.pkl stores only parse-time problems (count: 1); full re-validation reports 5: Computer Virus (all stats missing), Anubisath Guardian, Qiraji Soldier, Silithid (blank ability layers, June 2026 WoW batch). | Re-validation sweep, `autocroissant-validation-and-qa` baselines | Fix the four PSDs in TTSCardMaker (card content, not bot code) or certify as intentional | `autocroissant-validation-and-qa` §baselines |
| D2 | **"20 Creature Types" rulebook page has a `tornado` misspelling** (layer name in `MISSPELT_CARD_TYPES`). The single pickled problem card. | `inspect_pickle.py --problems` | Fix the PSD layer name in TTSCardMaker | `impossibility-cards-reference` §validation |
| D3 | **Stale exclusion: Shadow Duelist** sits in `ABILITY_EXCLUSIONS` but now has ability text. | Campaign burn-down census | Remove from the exclusion set after a golden-checked reparse | `autocroissant-psd-extraction-campaign` Phase 3.4 |
| D4 | **A second, never-reverted old_stats balloon** (`637698b`, 2025-11-25): ~26 unchanged cards breadth-archived in one sweep still pollute old_stats.pkl history. | Byte trace in archaeology; snapshot forensics | Optional cleanup; low value, only with a diff_stats-gated PICKLE commit | `autocroissant-failure-archaeology` Entry 3 |

### 1.3 OPEN — docs and cosmetics

| # | Issue | Where |
|---|-------|-------|
| C1 | `/pull` help + description claim "hard reset, than a git pull" — the code does fetch+merge (pickles-ours/code-theirs); the hard reset only exists behind `/update force_reset:True`. Also "than" → "then". | `commands/help.py:10`, `main.py:172` |
| C2 | Four commands missing from help.py: `/view_old_metadata`, `/list_guild_members`, `/list_guild_channels`, `/get_channel_messages`. | `commands/help.py` |
| C3 | `/export_cards only_ability` — help says default True, code default is False. | `commands/help.py:68` vs `main.py` signature |
| C4 | `/help` usage line omits the working "stats" category. | `commands/help.py:7` |
| C5 | Duplicate function names in main.py (`slash_set_ratio` ×2 ~303/313, `slash_delete_song` ×2 ~687/694). Harmless today (decorators registered at def time) but fragile — rename carefully, never "clean up" casually. | `main.py` |
| C6 | `/list_guild_members` requires the members intent, which `main.py` never enables — the command always fails with the ClientException message. Enable the intent (and in the Discord dev portal) or remove the command. | `main.py` intents block, `commands/management.py` |
| C7 | Remote-mode `update_stats` leaks temp PSD downloads (`urlretrieve`, never cleaned). Harmless at current scale. | `commands/psd_analyzer.py` |
| C8 | On macOS, `.DS_Store` leaks into the known-types list during local type population (harmless 36th "type"). | `commands/psd_analyzer.py` (`_populate_types_from_local`) |
| C9 | Programmatic trap: `update_stats()` Python default is `force_update=True`; only the slash wrapper passes False. Calling it bare from code force-reparses ~813 cards. | `commands/psd_analyzer.py` vs `main.py` |

### 1.4 Doctrine leak (design smell, owner-acknowledged)

| # | Issue | Detail owner |
|---|-------|--------------|
| S1 | Music's yt-dlp Safari-cookie behavior is keyed off `vram_usage == "mps"` — an AI config field steering a core feature. Candidate: dedicated config field (e.g. `browser_cookies`), getattr-defaulted off. | `autocroissant-ai-boundary` §known leaks |

---

## 2. Future improvements

Owner's stated ambitions (2026-07-11): **perfect extraction**, **AI that knows the game**, and
"it's a hobby — keep it fun." Everything below is CANDIDATE — measured milestones, no promises.

### 2.1 Perfect extraction (the active campaign)

Executable plan with gates and expected numbers: `.claude/skills/autocroissant-psd-extraction-campaign/SKILL.md`.
Ranked menu:
1. **`relpath` fix for B1 + a local-mode regression harness** — unlocks fast local iteration (no API budget). Result when: local and remote sweeps of an unchanged clone produce identical stats (diff_stats shows timestamp-only deltas).
2. **Restore gap/type mismatch detection as a high-precision validator check** — it existed (`fed8a83`) and thrashed (`081b1fd`); the precision conditions are derived in the campaign.
3. **Bbox-anchored type injection** — place `[type]` by icon coordinates instead of gap ordinals; the line-position estimation sub-problem is genuinely open.
4. **Exclusion-list burn-down** — 23 cards (18 ability + 5 stats); per-card recipe in the campaign; D3 above is the first easy win. Result when: both lists reach zero with problem count stable.

### 2.2 AI that knows the game (behind the AI boundary)

Steps and milestones: `.claude/skills/autocroissant-research-frontier/SKILL.md`.
1. **Game-grounded Q&A (`/ask`)** — corpus already exportable (48 rulebook pages + 813 structured cards); build the 20-question eval set FIRST, then a zero-dependency keyword baseline, only then heavier retrieval. Result when: ≥15/20 eval answers correct with citations. No LLM/embedding dep may enter core `requirements.txt`.
2. **In-style card art LoRA** — 1003 PNGs organized by series (Kirby 58 is the best corpus); train offline, drop into `models/loras/`, blind A/B with the friend group. Result when: ≥40% of generated cards mistaken for real in a shuffled test.
3. **Round-trip extraction verification** — re-render extracted text and image-diff against the card; single-card feasibility prototype first.

### 2.3 Operational hardening (small, high-value)

1. `try/finally` for `in_progress` (B3) and a defensive dispatch loop (B5) — the two single-points-of-silence.
2. Fix B4's `await` and C6's intent mismatch.
3. Sync help.py with reality (C1–C4) in one docs pass — checklist in `.claude/skills/autocroissant-docs-and-style/SKILL.md`.
4. Uncommitted as of 2026-07-11: the core/AI requirements split (`requirements.txt` = core, `requirements2.txt` = AI). Review and commit it; from-scratch checklists live in `autocroissant-build-and-env`.

### 2.4 Explicitly parked

- **Replacing pickles-in-git for state sync** — current mechanism works; any replacement must beat: zero-loss handoff both directions, no new infrastructure to babysit. "Do nothing" is the standing choice. (`autocroissant-research-frontier` §4)

---

## Maintenance

- Re-verify the OPEN bugs before working on them (one-liners in each owning skill's
  "Provenance and maintenance" section); line numbers drift.
- Library note: six skill frontmatter descriptions contain unquoted `word: word` constructs —
  fine for the current lenient loader, need quoting if a strict YAML parser is ever adopted.
- This file is an index, not a second home for the stories. Keep entries to a few lines; link the owner skill.
