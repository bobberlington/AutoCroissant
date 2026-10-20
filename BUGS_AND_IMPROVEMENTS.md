# AutoCroissant — Known Bugs and Future Improvements

Compiled 2026-07-11 from a full repo audit (code, git history, live data); revised 2026-10-19 after the
B1/B2/C7/C8/C9 fix batch and 2026-10-20 after C13's fix and the archiving of local mode (retiring B6) and of
dead helpers (retiring C10)
(all uncommitted working tree as of those dates). Detail, evidence, and
procedures live in the skill library under `.claude/skills/` — each entry points at its owner
(`ttscardmaker-companion` covers the TTSCardMaker side: icon font, website, fonts, Photoshop scripting).
Fixes route through `.claude/skills/autocroissant-change-control/SKILL.md`; nothing here is
license to hot-patch. When an entry is fixed or retired, move its story to
`autocroissant-failure-archaeology` and delete the row here.

Retired 2026-10-19 (stories in `autocroissant-failure-archaeology` Entry 1 and Entry 10): B1 local-mode
path bug, B2 scheduled-update amplifier, C7 temp-PSD leak, C8 `.DS_Store` type, C9 `force_update=True`
Python default, C11 `reminder.pkl.tmp` not gitignored; also fixed: management.py did not compile under
Cython 3.3, and `.gitignore`'s `*.pyd`/`*.dll` patterns that matched nothing. Fixed 2026-10-20: C13
parser token spellings looser than the font, and stale manifest entries (Entry 10.11). Retired 2026-10-20: B6
local-timestamp re-archiving, by archiving local mode itself (`/update_stats` is GitHub-only; the old traversal is
`archive/local_update_stats.py` on branch `archive/local-update-stats`, not in main, sandbox use only; Entries 10.10
and 10.12). Retired 2026-10-20: C10 `convert_value` (deleted with `parse_named_args`; Entry 10.7). Dead helpers
`url_to_pilimage`/`cv2_to_pil`/`pil_to_cv2` and `CardRepository._exact_match_pattern` were deleted too (Entry
10.13). The old code is still in a7a4bb3: `git show a7a4bb3:commands/utils.py`. Fixed 2026-10-20: stars counted
from the PSD counted every layer of a multi-layer star. Indoctrination had 20 instead of 4, Michael's Blessing 11
instead of 5, Spawn of the Forbidden One 10 instead of 4, and cards without a "Stars" group 0. Now
`_count_stars` counts one per star, and 0 is flagged (`NO STARS FOUND`; impossibility-cards-reference §2). Also
fixed 2026-10-20: stat bars read per row (impossibility-cards-reference §3.4). Sonic's speed was 23, now 20 (row
groups "2" and "1" were added in); Sans's defense was 10, now 1 (two exposures visible). Heart bars were 55 and are
now counted: Blank Card 6/1, Dirty Bed 3/0, Tainted Lazarus 12/8/11/9, which is its two sides. Twin Emperors' health
was 3, now 7. The stored values update at the next `/update_stats` + `/push`.

---

## 1. Bugs

### 1.1 OPEN — high priority

| # | Bug | Where | Impact | Workaround / candidate fix | Detail owner |
|---|-----|-------|--------|---------------------------|--------------|
| B3 | **`/ai` wedges permanently after any mid-generation exception.** `in_progress = True` is set with no `try/finally`; an exception (bad URL, OOM, model misconfig) leaves it stuck and every later `/ai` only prints "Request queued." until restart. | `commands/diffusion.py` (`diffusion()`, `in_progress` set ~line 681, cleared ~line 780) | AI capability silently dead until bot restart | Restart bot to clear; candidate fix: wrap generation in `try/finally: in_progress = False` + drain queue | `autocroissant-debugging-playbook` §hangs |
| B4 | **`init_vc` crashes when the caller is not in a voice channel.** `await queue_message(...)` awaits a plain function's `None` return → `TypeError` instead of the intended "You are not in a voice channel." message. | `commands/music_player.py` (~line 177) | `/play` from outside a voice channel errors instead of replying | Candidate fix: drop the `await` (queue_message is sync) | `autocroissant-debugging-playbook` §music |
| B5 | **The dispatch loop can die permanently.** Exceptions outside the caught set stop a `tasks.loop` for good: (a) deleted/unavailable channel → `client.get_channel()` returns `None` → `AttributeError` **inside the fallback**; (b) >2000-char content that fails in the fallback path too. After that, the whole bot goes silent (all sends flow through this loop). **Partly mitigated 2026-10-19:** reminders now skip channels the bot can't see (`_channel_missing` in analytics.py), so the scheduler no longer feeds the loop a dead channel; every other sender still can, and `process_dispatch_queue`/`process_edit_queue` in main.py are unchanged. | `main.py` (`process_dispatch_queue`, fallback `client.get_channel(...).send(...)`) | One poisoned message silences the entire bot until restart | Candidate fix: wrap each drain iteration in `try/except` + log; guard `get_channel()` None | `autocroissant-architecture-contract` §known-weak points |

### 1.2 OPEN — data debt (as of 2026-10-19)

| # | Issue | Evidence | Suggested action | Detail owner |
|---|-------|----------|------------------|--------------|
| D1 | **4 latent problem cards invisible in the pickled count.** stats.pkl stores only parse-time problems (count: 1); full re-validation reports 5: Computer Virus (all stats missing), Anubisath Guardian, Qiraji Soldier, Silithid (blank ability layers, June 2026 WoW batch). | Re-validation sweep, `autocroissant-validation-and-qa` baselines | Fix the four PSDs in TTSCardMaker (card content, not bot code) or certify as intentional | `autocroissant-validation-and-qa` §baselines |
| D2 | **"20 Creature Types" rulebook page has a `tornado` misspelling** (layer name in `MISSPELT_CARD_TYPES`). The single pickled problem card. | `inspect_pickle.py --problems` | Fix the PSD layer name in TTSCardMaker | `impossibility-cards-reference` §validation |
| D3 | **Stale exclusion: Shadow Duelist** sits in `ABILITY_EXCLUSIONS` but now has ability text. | Campaign burn-down census | Remove from the exclusion set after a golden-checked reparse | `autocroissant-psd-extraction-campaign` Phase 3.4 |
| D4 | **A second, never-reverted old_stats balloon** (`637698b`, 2025-11-25): ~26 unchanged cards breadth-archived in one sweep still pollute old_stats.pkl history. | Byte trace in archaeology; snapshot forensics | Optional cleanup; low value, only with a diff_stats-gated PICKLE commit | `autocroissant-failure-archaeology` Entry 3 |
| D5 | **stats.pkl is stale for 61 cards.** The 2026-10-19 sandboxed local sweep found 59 ability and 2 star differences from the stored data. The git-HEAD parser and the new parser agree on every PSD, so these are PSD edits made after the bot's last sweep, not parser changes. | Sandboxed sweep (`autocroissant-validation-and-qa` §4.1) | Run `/update_stats` (GitHub-only since 2026-10-20), run the diff_stats gate (expect these 61 cards among the modified rows; other cards touched by newer TTSCardMaker commits may show timestamp-only changes; 0 path/type changes), then `/push`. A pickle push now also refreshes the TTSCardMaker website | `autocroissant-run-and-operate` §7 |
| D6 | **No card uses the stoogedings font yet (0 of 813 PSDs).** About 45 PSDs (81 smart-object icon layers below the midline) still use image-layer icons, so they still depend on gap injection. Counting any layer kind gives 61 PSDs / 148 layers, some of them decoration rather than inline icons. | TTSCardMaker migration audit (2026-10-19) | Migrate them in TTSCardMaker (card content). The JSX prototype and its overlay-layer caveat are in the companion skill | `ttscardmaker-companion` |

### 1.3 OPEN — docs and cosmetics

| # | Issue | Where |
|---|-------|-------|
| C1 | `/pull` help + description claim "hard reset, than a git pull" — the code does fetch+merge (pickles-ours/code-theirs); the hard reset only exists behind `/update force_reset:True`. Also "than" → "then". | `commands/help.py:10`, `main.py:175` |
| C2 | Four commands missing from help.py: `/view_old_metadata`, `/list_guild_members`, `/list_guild_channels`, `/get_channel_messages` (55 registered, 51 documented as of 2026-10-19). | `commands/help.py` |
| C3 | `/export_cards only_ability` — help says default True, code default is False. | `commands/help.py:76` vs `main.py` signature |
| C4 | `/help` usage line omits the working "stats" category. | `commands/help.py:7` |
| C5 | Duplicate function names in main.py (`slash_set_ratio` ×2, `slash_delete_song` ×2). Harmless today (decorators registered at def time) but fragile — rename carefully, never "clean up" casually. | `main.py` |
| C6 | `/list_guild_members` requires the members intent, which `main.py` never enables — the command always fails with the ClientException message. Enable the intent (and in the Discord dev portal) or remove the command. | `main.py` intents block, `commands/management.py` |
| C12 | Pillow is still not declared in `requirements.txt`, and a third module now imports it at top level (`commands/type_emojis.py`). It arrives only through psd-tools. | `requirements.txt` |

### 1.4 Doctrine leak (design smell, owner-acknowledged)

| # | Issue | Detail owner |
|---|-------|--------------|
| S1 | Music's yt-dlp Safari-cookie behavior is keyed off `vram_usage == "mps"` — an AI config field steering a core feature. Candidate: dedicated config field (e.g. `browser_cookies`), getattr-defaulted off. | `autocroissant-ai-boundary` §known leaks |

### 1.5 Pending live verification (as of 2026-10-19)

| # | Item | What was verified | What is still missing | Detail owner |
|---|------|-------------------|-----------------------|--------------|
| V1 | `/sync_emojis` and the startup emoji sync (126 application emojis) | Offline, with a fake client and real GitHub downloads | Not yet run against live Discord. Watch the first startup log line `Type emojis: ... uploaded, ... ready.` and run `/query_rulebook` once to see emojis | `autocroissant-run-and-operate` §11 |
| V2 | Reminder rewrite (run-as-creator, name:value commands, migration of reminder.pkl) | Code read 2026-10-19 | Not known to have fired against live Discord yet. Watch the first startup's `migrated command` log lines and the first firing | `autocroissant-run-and-operate` §9 |
| V3 | Cython build of all 8 INCLUDE_FILES modules with Cython 3.3.0 | Scratch build | No host has run the compiled build since. After building on a host, check for stale `.so` files | `autocroissant-build-and-env` §5 |

### 1.6 Companion repo (TTSCardMaker) items that affect the bot

Owned by `ttscardmaker-companion`:
- The card website (`site/`, `.github/workflows/site.yml`) is built but not yet pushed or enabled in GitHub Pages
  (Settings → Pages → Source: GitHub Actions). Once live, every bot `/push` of pickles publishes to it.
- The website unpickles the bot's pickles with stub classes only for `commands.*`, so anyone who can push to
  AutoCroissant `main` can run code in the TTSCardMaker Actions job; and renaming CardInfo attributes silently
  blanks website fields. Both repos are in the same trusted circle today.
- 25 fonts used in card text are missing from `MDW/Fonts` (Consolas affects the whole rulebook), and `MDW/Fonts`
  already holds proprietary fonts, which matters if the repo is public.
- The TTSCardMaker work (font, manifest, site, workflow) is uncommitted, and the migration prototype and font-audit
  scripts exist only in `/private/tmp/fontexplore/`, which macOS clears on reboot.

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

### 2.2 AI that knows the game (behind the AI boundary)

Steps and milestones: `.claude/skills/autocroissant-research-frontier/SKILL.md`.
1. **Game-grounded Q&A (`/ask`)** — corpus already exportable (48 rulebook pages + 813 structured cards); build the 20-question eval set FIRST, then a zero-dependency keyword baseline, only then heavier retrieval. Result when: ≥15/20 eval answers correct with citations. No LLM/embedding dep may enter core `requirements.txt`.
2. **In-style card art LoRA** — 1003 PNGs organized by series (Kirby 58 is the best corpus); train offline, drop into `models/loras/`, blind A/B with the friend group. Result when: ≥40% of generated cards mistaken for real in a shuffled test.
3. **Round-trip extraction verification** — re-render extracted text and image-diff against the card; single-card feasibility prototype first.

### 2.3 Operational hardening (small, high-value)

1. `try/finally` for `in_progress` (B3) and a defensive dispatch loop (B5) — the two single-points-of-silence.
2. Fix B4's `await` and C6's intent mismatch.
3. Sync help.py with reality (C1–C4) in one docs pass — checklist in `.claude/skills/autocroissant-docs-and-style/SKILL.md`.
4. Review and commit the 2026-10-19 working-tree batch (psd_analyzer fixes and icon-font support, management.py
   Cython fix, type_emojis.py, the reminder rewrite, help/main updates, `discord.py>=2.5`, the `.gitignore`
   update, the 2026-10-20 remote-mode default and exact token spellings, the dead-helper deletions (archived on
   branches), the diagnostics-script updates,
   and the skill-library updates). Keep it a code-only
   commit; the D5 pickle refresh is a separate bot-made PICKLE. (The 2026-07-11 requirements split was committed
   in a7a4bb3.)
5. Small cleanup: C12 (declare Pillow).

### 2.4 Explicitly parked

- **Replacing pickles-in-git for state sync** — current mechanism works; any replacement must beat: zero-loss handoff both directions, no new infrastructure to babysit. "Do nothing" is the standing choice. (`autocroissant-research-frontier` §4)

---

## Maintenance

- Re-verify the OPEN bugs before working on them (one-liners in each owning skill's
  "Provenance and maintenance" section); line numbers drift.
- Library note: six skill frontmatter descriptions contain unquoted `word: word` constructs —
  fine for the current lenient loader, need quoting if a strict YAML parser is ever adopted.
- This file is an index, not a second home for the stories. Keep entries to a few lines; link the owner skill.
