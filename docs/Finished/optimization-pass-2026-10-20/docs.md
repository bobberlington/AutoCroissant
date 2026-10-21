# Docs audit (read-only), 2026-10-20

Repo: `/Users/michaelsrouji/Desktop/AutoCroissant`, main at 7f6bd05, clean tree. Nothing in the repo was edited.
Scratch tooling for part 2 lives in `/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/`.

**Order of work.** Code changes in this pass will move lines again. So: apply the part 1 edits (help.py / main.py)
with the code phases, then **re-run the part 2 script last** (section 2.1) and apply its table, then the
part 2.3 claim edits and the part 3 edits. Line numbers below are HEAD 7f6bd05 readings.

Summary of findings:
- 51 commands registered, 47 in help.py. Still missing: `/view_old_metadata`, `/list_guild_members`,
  `/list_guild_channels`, `/get_channel_messages` (C2). C1 and C3 are still open as described.
- There are 7 more help/describe mismatches the bug list doesn't have: `/update force_reset` (help says the reset
  happens "first", but it replaces the push and pull), `/react`'s parameter is named `reactions`, `/mass_replace`'s are
  `old_value` and `new_value`, `/purge num` is how many messages are searched, not deleted, and some describe strings
  give no default.
- Skills: 559 `file:line` references. 537 point at main.py / commands/*.py / global_config.py / setup.py on main.
  186 are still exact, **305 are stale** (replacement table in 2.4), 28 are deliberately historical (keep), and
  21 point at archive branches (all checked correct). About 25 more bare line numbers sit in provenance tables (2.2).
- False claims after today: the jetson "Python 3.10" (9 places), the perms_check gate table (12 call sites, now 13 with
  `/update_tts_save`), the "weekly emoji sync" reminder (2 places), docs-and-style counts. BUGS rows: C14, V3 and
  §2.3.4 are already done, and §2.4 items 2-5 are done.

---

## 1. Commands vs help.py vs describe strings

Method: `ast` over main.py (51 `@tree.command`, signature defaults, every `@app_commands.describe` key) compared
with the five dicts in commands/help.py, plus reading every worker. Every parameter has a describe string. No
describe string is longer than 100 characters, and no describe string names a parameter that doesn't exist.

### 1.1 Coverage (C2): 51 registered, 47 documented

`comm` of the anchored extraction (docs-and-style §3.1) gives exactly these four. Insert them as shown.

**commands/help.py, `general_commands`: insert after line 16 (`/sync_global`)**:

```python
    "/list_guild_members <guild_id>:"           : "Lists the members of a guild by its ID (admin only).",
    "/list_guild_channels <guild_id>:"          : "Lists the channels of a guild by its ID (admin only).",
    "/get_channel_messages <guild_id> <channel_id> [limit]:" : "Lists a text channel's most recent messages, oldest first (admin only).\n"
                                                  "- limit: How many messages (default 50, -1 = all)",
```

If C6 is NOT fixed in this pass (the members intent stays off), use this `/list_guild_members` value instead:
`"Lists the members of a guild by its ID (admin only). Needs the server members intent, which the bot does not enable, so it fails today."`.
If C6 is fixed by removing the command, drop that line. If it is fixed by enabling the intent, use the line above.

**commands/help.py, `stats_commands`: insert after line 89 (the end of `/update_metadata`), before `/mass_replace`**:

```python
    "/view_old_metadata <query> [version]:"     : "Shows an older stored version of a card's data.\n"
                                                  "- version: 0 = most recent old version (default), 1, 2, ... = older ones, -1 = oldest",
```

(Semantics checked against `view_old_metadata`, psd_analyzer.py:2095-2171: -1 means index 0 (the oldest), and N >= 0
means index len-1-N.)

### 1.2 Behaviour drift: exact fixes

| # | Command | Where | Now | Code truth | Replace with |
|---|---|---|---|---|---|
| C1 | /pull | help.py:10 | "Does a hard reset, then a git pull, and reports status (admin only)." | `git_pull` (update_bot.py:42-110) runs fetch, then merge-pull. On a conflict it keeps the local pickles and takes the remote code (`--ours`/`--theirs`). It never resets. | `"Fetches and merges the latest code, keeping the local pickles on a conflict, and reports status (admin only).",` |
| C1 | /pull | main.py:169 | `description="Does a hard reset, than a git pull, and reports status."` | same | `description="Fetches and merges the latest code (keeps local pickles), and reports status (admin only)."` (90 chars) |
| C3 | /export_cards | help.py:80 | "- only_ability: Export only ability text (default: True)\n" | `only_ability: Optional[bool] = False` (main.py:387). It only affects the **text** export: the CSV branch ignores it (psd_analyzer.py:2201-2220) | `"- only_ability: Text export only: just names, aliases and ability text (default: False)\n"` |
| C3+ | /export_cards | help.py:81 | "- as_csv: Export as CSV instead of text (default: True)" | the CSV keeps only `df["ability"].notna()` rows | `"- as_csv: Export as CSV instead of text; cards with no ability field are left out (default: True)",` |
| C3 | /export_cards | main.py:383-384 | `only_ability='Should we only output the ability of cards?'`, `as_csv='Should we output the file as a csv?'` | no defaults stated (that is how C3 survived) | `only_ability='Text export only: just names, aliases and ability text. Default = False.'` (72) / `as_csv='Should we output the file as a csv? Default = True.'` |
| new | /update | help.py:12 | "...force_reset: hard reset to origin first (dangerous!)." | `update_bot` (update_bot.py:131-156): with force_reset it runs `git_reset_hard` **instead of** the push and pull, so local pickles are lost and nothing is pushed | two lines: `"Pushes the pickles, pulls, then restarts (admin only).\n"` / `"- force_reset: Skip the push and pull; hard reset to origin/main instead, discarding local pickles (dangerous! default: False)",` |
| new | /update | main.py:191 | `force_reset="If True, discard all local changes and reset to remote (dangerous!)"` | true, but no default and no mention of skipping the push | `force_reset="Skip the push and pull: hard reset to remote, losing local pickles (dangerous!). Default = False."` (97) |
| new | /react | help.py:17-18 | key `/react <emojis> [message_id]:`, bullet `- emojis:` | the parameter is `reactions` (main.py:744) | key `"/react <reactions> [message_id]:"`, bullet `"- reactions: Space-separated emojis (👍 👎 or <:custom:123456>)\n"` |
| new | /mass_replace | help.py:90 | key `/mass_replace <field> <old> <new>:` | parameters are `old_value`, `new_value` (main.py:479-481) | `"/mass_replace <field> <old_value> <new_value>:" : "Replace all instances of one field value with another across the entire database (author, series, subtype, card_type).",` |
| new | /purge | help.py:13 | "Deletes messages. Can specify user, number of messages (default 100), and bulk mode." | `channel.purge(limit=num, check=author==user)` (management.py:27). `num` is how many recent messages are **searched**. user defaults to the bot. -1 = all | 4 lines: `"Deletes one user's messages among the most recent ones in this channel.\n"` / `"- user: Whose messages (default: the bot's own)\n"` / `"- num: How many recent messages to search (default 100, -1 = all)\n"` / `"- bulk: Bulk delete; faster, needs Manage Messages (default: False)",` |
| new | /purge | main.py:628 | `num='The number of messages to delete. Default = 100.'` | as above | `num='How many recent messages to search (-1 = all). Default = 100.'` |
| new | /update_stats | help.py:71 | "...and re-uploads edited type emojis." | `sync_changed_emojis` (type_emojis.py:331-335) uploads new **or** changed icons and deletes emojis whose icon is gone | `"...re-parses only PSDs that changed, and uploads new or edited type emojis (deleting removed ones). "` |

Aligned copies of the multi-line help.py entries (50-space continuation, `:` at column 49, like the
neighbours):

```python
    "/update [force_reset]:"                    : "Pushes the pickles, pulls, then restarts (admin only).\n"
                                                  "- force_reset: Skip the push and pull; hard reset to origin/main instead, discarding local pickles (dangerous! default: False)",
    "/purge [user] [num] [bulk]:"               : "Deletes one user's messages among the most recent ones in this channel.\n"
                                                  "- user: Whose messages (default: the bot's own)\n"
                                                  "- num: How many recent messages to search (default 100, -1 = all)\n"
                                                  "- bulk: Bulk delete; faster, needs Manage Messages (default: False)",
    "/react <reactions> [message_id]:"          : "Adds reactions to a message.\n"
                                                  "- reactions: Space-separated emojis (👍 👎 or <:custom:123456>)\n"
                                                  "- message_id: Optional. If omitted, reacts to the most recent message in the channel.",
```

### 1.3 Admin markers missing from main.py descriptions (docs-and-style §4 "known inconsistency")

All four are gated by `perms_check` in code, and their help.py entries already say "(admin only)":

| main.py | Now | Replace with |
|---|---|---|
| 155 | `@tree.command(name="stop_bot", description="Stop the bot.")` | `description="Stop the bot (admin only)."` |
| 169 | /pull | see C1 above (it includes the marker) |
| 179 | `description="Does a git push, and reports status."` | `description="Commits the pickles, rebases onto new code, pushes, and reports status (admin only)."` (84; matches `git_push`, update_bot.py:16-39) |
| 189 | `description="Push pickles, pull latest code, and restart the bot."` | `description="Push pickles, pull latest code, and restart the bot (admin only)."` |

### 1.4 Describe strings that omit a default (convention: "State the default, in the string")

| main.py | Now | Replace with |
|---|---|---|
| 232 | `all="List all reminders in the server instead of just this channel.",` | `all="List all reminders in the server instead of just this channel. Default is false.",` (80) |
| 275 | `limit='The number of cards to output before stopping.',` | `limit='The number of cards to output before stopping. Default = all.',` |
| 276 | `howmany='Instead of returning card images, return the number of matches.',` | `howmany='Instead of returning card images, return the number of matches. Default = False.',` (80) |
| 358 | `output_problematic_cards='Should we output cards with consistency errors?',` | `output_problematic_cards='Should we output cards with consistency errors? Default = True.',` |
| help.py:41 | `"- howmany: If True, only returns the count instead of images\n",` (the trailing `\n` adds an empty line) | `"- howmany: If True, only returns the count instead of images (default: False)",` |

Optional, cosmetic: main.py:356 `description="Manual call to update card stats."` could become
`"Updates the card database from GitHub (re-parses changed PSDs)."` (63).

### 1.5 Verified consistent (no edit)

/help, /restart_bot, /stop_bot (help), /push (help), /sync_global, /list_guilds, /leave_guild, /set_reminder,
/list_reminders, /remove_reminder, /query, /website, /query_rulebook, Query Syntax, /sync_emojis, /set_ratio,
/set_repo, /alias, /frankenstein, /update_stats (parameters and defaults), /update_tts_save, /export_rulebook,
/update_metadata (all 13 parameters), /list_orphans, and all 17 music commands. Names, parameters and defaults
match.

### 1.6 Not docs: noted for the owner (no change proposed in this pass)

These commands have no `perms_check` and help.py doesn't claim they do: /restart_bot (already noted in
config-and-flags), /purge (any member can have the bot delete **another user's** messages if the bot has Manage
Messages), /update_metadata, /mass_replace, /alias, /set_repo, /set_ratio, /update_stats, /delete_song,
/delete_all_music. Changing that is a behaviour change, so it isn't in this pass's charter.

### 1.7 After 1.1-1.4: docs-and-style edits that go with them

- Frontmatter: `the known doc-vs-code drift instances (e.g. /pull claims a hard reset that git_pull never does; 4 of 51 registered commands missing from help.py as of 2026-10-20)`
  → `the doc-vs-code drift audit (help.py documents all 51 registered commands as of <date>; the /pull hard-reset claim was fixed)`.
- §2.1 table: `music_commands` line `90` → `94`. `general_commands` entries `11` → `14`, `stats_commands` `7` → `8`.
  `Total: 48 keys, 47 real commands documented.` → `Total: 52 keys, 51 real commands documented.`
  §2 intro: `(142 lines as of 2026-10-20)` → the new `wc -l` (146 today, before the inserts).
- §3.1: replace the table with "None (0 of 51 missing, <date>)". Keep the extraction snippet.
- §3.2: delete D1 and D2, and add one line pointing to the archaeology entry (part 3, 10.20).
- §4: delete the "Known inconsistency, do not copy" paragraph (fixed by 1.3).
- §2.2 quotes `/react <emojis>`: update the quoted example to the new `/react <reactions>` entry and its `help.py:` lines.
- Provenance block: `# lines 6,22,35,70,90,114` → `6,22,35,70,94,118` (before the inserts; re-read after).
  `# 142` → the new count. `# 122-123` → `126-127`. `# 135` stays 135 if no line above it moves.
  `# 47` → `# 51`. Delete the D1/D2 grep lines, or change them to expect no `hard reset` in help.py:10 / main.py:169.
  Census lines: `# 128` from-imports → `165`. `# 9` plain imports → `10` (new: `import commands.repo_history as repo_history` in
  tts_save.py). `# 15` len-1 idiom → `19`. `psd_analyzer:2288, query_card:714` → `2295, 718`.
- §5.2 prose: `128 from-import lines vs 9 plain` → `165 vs 10`, and add `import commands.repo_history as repo_history`
  (tts_save.py) to the list of exceptions. Also flag it as a style outlier: the house rule is `from x import y`.
- §5.4: `15 x[len(x) - 1] occurrences` → `19`.

---

## 2. Skills (.claude/skills, 16 SKILL.md)

### 2.1 Method (scripted; re-run it after the code phases)

`docs_refaudit/refs5.py` extracts every `path:N`, `path:N-M`, `path:N, M`, `path (lines N-M)`, `path ~N` and every
bare `(:N)` that follows a path on the same line. It finds 559 references. For each one:

1. **Anchor.** `git blame` the skill line to get the commit that last wrote it. In that commit, if a backticked or
   snake_case symbol from the surrounding text sits at the cited line ±8, that commit is the anchor. If not, walk back
   through the target file's history to the newest commit where the symbol is at the cited line ±3.
2. **Map.** A difflib line map from the anchor version of the file to HEAD gives the new line (or range).
3. **Verify.** Check that the symbol is near the new line (`symbol`). Where no symbol exists, the result is the
   blame-anchored map only (`blame`). The fallbacks are a unique `def`/`class`/assignment lookup (`def-lookup`) and
   35 hand overrides in `overrides.json` (`manual`), each one read.
4. **Historical.** A ref dated before today ("on 2026-10-19", "2026-07-11 ones", "was ... before the archive",
   failure-archaeology Entry 1's chain) is kept as written.

Re-run (about 10 s), then apply:

```bash
cd /Users/michaelsrouji/Documents/ac-optimization/docs_refaudit
P=/Users/michaelsrouji/.venvs/ac-py314-psd125/bin/python
$P refs5.py && $P final.py && $P table.py && $P to_md.py > refs_table.md   # 305 stale today
$P apply.py                                                                # edits the 16 SKILL.md in place
```

`apply.py` asserts that each old string is still at its recorded column before replacing it. A dry run against a
copy of the skills applied 305/305 and touched 270 lines. If the skills are edited before the re-run, re-run
`refs5.py` first so the positions are fresh.

Results today: 186 exact, **305 stale**, 28 historical (keep), 21 on archive branches. All 21 are labeled as
branch-only in their text and checked against `git show archive/...`: diffusion.py 25/127/208/255/490/683/782,
main.py 524/571, local_update_stats.py:115. One more is mislabeled:
`autocroissant-research-frontier:163` cites `` `/set_lora` (main.py:571 `` → write `archive/ai-generation:main.py:571`
(set_lora is not on main).

Spread of the 305: architecture-contract 49, impossibility-cards-reference 48, docs-and-style 41,
run-and-operate 33, validation-and-qa 26, config-and-flags 25, debugging-playbook 25, psd-extraction-campaign 17,
analysis-toolkit 16, research-frontier 15, change-control 13, failure-archaeology 8, research-methodology 7,
build-and-env 5. Check used: 188 symbol, 97 blame, 16 manual, 4 def-lookup. The remaining overrides mark refs as
historical or unchanged.

### 2.2 Bare line numbers the script can't see (provenance "expected" values)

| Skill:line | Written | Now (HEAD 7f6bd05) |
|---|---|---|
| config-and-flags:102 | `(all 12 call sites, verified 2026-10-20 ...)` | `(all 13 call sites, ...)` |
| config-and-flags:107-118 | table `/stop_bot 154, /pull 168, /push 178, /update 191, /set_reminder 220, /list_guilds 629, /leave_guild 644, /sync_global 653, /sync_emojis 668, /list_guild_members 684, /list_guild_channels 699, /get_channel_messages 719` | `157, 171, 181, 194, 223, 642, 657, 666, 681, 697, 712, 732`, and **add a row** `` | `/update_tts_save` (added 2026-10-20) | 373 | `` after `/set_reminder` |
| architecture-contract:391 | `queue_command(check_reminder)` (751) | 764 |
| architecture-contract:396 | to_thread (67, passthrough 69-71) | unchanged |
| architecture-contract:397 | run_command (740) | 753 |
| architecture-contract:399 | slash_registry (105) | 107 |
| architecture-contract:402 | (107, 121) | unchanged |
| architecture-contract:404 | duplicate defs (304/314, 597/604) | 307/317, 610/617 |
| architecture-contract:407 | mkstemp/remove (1433, 1668) | 1435, 1659 |
| architecture-contract:409 | psd_analyzer 26, type_emojis 60; query_card (334, 379, 461; 15) | 25, 98; (338, 383, 465; 16) |
| architecture-contract:410 | (2288), (714), (45), (61, 26) | 2295, 718, 45, 61, 30 |
| docs-and-style:466-518 (provenance block) | see 1.7 | see 1.7 |
| failure-archaeology:608 | `anything above 9fb3093 → review entries` | `anything above 7f6bd05` (after adding 10.17-10.19) |
| failure-archaeology:614 | (1668, 1502, 1446/1829); analytics (41, 37, 221) | (1659, 1505, 1448/1831); 41, 37, 221 unchanged. The pattern `startswith(.\.)` never matches `startswith('.')` (the `.` sits before the quote): use `grep -n "startswith('.')"` |
| failure-archaeology:618-619 | `:1146 on 2026-10-20`, `:1244 on 2026-10-20` | `:1148`, `:1246` |
| failure-archaeology:624 | `→ 1827 / 686 / 50 / 1262 as of 2026-10-20` | `→ 1829 / 688 / 52 / 1264` |
| debugging-playbook:330 | `BREAK_LEN` (~19) | unchanged |
| diagnostics-and-tooling:624 | `TYPE_REGION_RATIO` (line 33 as of 2026-10-19) | 40 (keep the date or write "40 as of 2026-10-20") |
| diagnostics-and-tooling:628 | `grep -n "relpath(full_path" commands/psd_analyzer.py  # ... (line 1226)` | **broken command**: relpath left main with local mode. Replace with `git grep -n "relpath(full_path" archive/local-update-stats -- archive/local_update_stats.py   # 115 on the archive branch` |
| ttscardmaker-companion:643 | ICON_FONT_PREFIX / ICON_MANIFEST_PATH `# 54, 55` | `# 56, 57` |
| ttscardmaker-companion:644 | MAX_NAME_LEN / NO_SUFFIX / ICON_FOLDERS `# 19, 21, 20` | `# 23, 25, 24` |
| build-and-env:265 | `grep -n "from PIL" commands/*.py` (type_emojis:8 only) | type_emojis:11 only |

### 2.3 Claims that are false after today's changes (exact edits)

**Python on the jetson (3.10 → 3.14 venvs).** NEXT_SESSION_PLAN Task 0: Python 3.14.8 is built, the venvs are
`~/venvs/autocroissant` (psd-tools 1.25.0) and `~/venvs/ojwikibot`, and both units' ExecStart point at their venv.
3.10 is still installed until step 6.

| Skill:line | Replace | With |
|---|---|---|
| change-control:24 | `(systemd \`autocroissant.service\`, Python 3.10)` | `(systemd \`autocroissant.service\`, Python 3.14 venv \`~/venvs/autocroissant\` since 2026-10-20)` |
| change-control:35 | `` `python3.10 setup.py build_ext --inplace` `` | `` `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` `` |
| build-and-env:26 | `Production (the jetson) runs Python 3.10 with psd-tools 1.18. Guidance: prefer 3.11+ for new envs.` | `Production (the jetson) runs Python 3.14.8 from \`~/venvs/autocroissant\` (psd-tools 1.25.0) since 2026-10-20; 3.10 stays installed until NEXT_SESSION_PLAN Task 0 step 6. Keep code 3.10-compatible.` |
| build-and-env:26 (same cell) | `The only env with psd-tools is a scratch venv, \`/private/tmp/fontexplore/venv\` ... /tmp is wiped on reboot. Create a durable \`.venv\` (section 7a) before any parser or bot work.` | `A durable venv with every dependency now exists: \`~/.venvs/ac-py314-psd125\` (3.14.8, psd-tools 1.25.0, discord.py 2.7.1; no Cython, so install it there before a build). The old scratch venv in /private/tmp is gone on reboot.` |
| build-and-env:83 | `Python 3.10, systemd \`autocroissant.service\`, compiled \`.so\` (section 5)` | `Python 3.14 venv \`~/venvs/autocroissant\`, systemd \`autocroissant.service\` (ExecStart = the venv's python), compiled \`.so\` (section 5)` |
| build-and-env:133 | `` `python3.10 setup.py build_ext --inplace` `` | `` `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` `` |
| build-and-env:178 | `# use python3.11+ if available (guidance); 3.10.20 is verified` | `# 3.14 is what both machines run (2026-10-20); code stays 3.10-compatible` |
| build-and-env:209 | `compiles with \`python3.10\`` | `compiles with \`~/venvs/autocroissant/bin/python\`` |
| build-and-env:283 | `(3.10.20 verified; 3.11+ preferred for new envs — guidance)` | `(3.14.8 on the Mac venv and the jetson, 2026-10-20)` |
| build-and-env:267 | `(fails on 2026-10-19)` | `(fails on 2026-10-19; use ~/.venvs/ac-py314-psd125/bin/python on the Mac)` |
| run-and-operate:111 | `` `python3.10 setup.py build_ext --inplace` `` | `` `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` `` |
| debugging-playbook:232 | `` `python3.10 setup.py build_ext --inplace` `` | `` `~/venvs/autocroissant/bin/python setup.py build_ext --inplace` ``, and the dependency install becomes `~/venvs/autocroissant/bin/python -m pip install -r requirements.txt` |

(diagnostics-and-tooling:582 "Python 3.10.20" is a dated 2026-07-11 verification: keep it.)

**The weekly /sync_emojis reminder is gone** (f590a57e removed. /update_stats syncs emojis since e13d754.)

| Skill:line | Replace | With |
|---|---|---|
| ttscardmaker-companion:612 | `reminders such as the weekly emoji sync` | `reminders` |
| run-and-operate:419 | `The second is a weekly full emoji sync, now just an example: /update_stats keeps emojis current` | `The second is a weekly full emoji sync, only an example: the live one (f590a57e) was removed on 2026-10-20 because /update_stats keeps emojis current` |
| run-and-operate:3 (frontmatter) | `"reminder didn't run" / "weekly emoji sync",` | `"reminder didn't run",` (it advertises a reminder that no longer exists) |

**/update_tts_save exists; 51 commands.** run-and-operate (§TTS save, provenance), architecture-contract (module
map), config-and-flags (`TTS_SAVE_PATH`, GIT_TOKEN) and docs-and-style (51/47) already say so. Two gaps remain:
the config-and-flags gate table (2.2: it needs a 13th row) and docs-and-style §2.1 after the C2 fix (1.7). I found no
"50 commands" text in any skill. BUGS C2 still says 50/46 (part 3).

**Other drift found on the way:**

| Skill:line | Issue | Fix |
|---|---|---|
| docs-and-style:115-128 | §3.1 table line numbers (439/676/691/706) | 452/689/704/719 (in the 2.4 table). The whole table goes away with C2 |
| docs-and-style:150 / run-and-operate:172 | quote `main.py:166` for the /pull description | 169 (in 2.4). Both sentences go away with C1 |
| psd-extraction-campaign:575 / failure-archaeology:609 | historical psd_analyzer.py:1226 (relpath) | correct as written: keep |
| frontmatter length | config-and-flags (1,790 chars), run-and-operate (1,982) and ttscardmaker-companion (1,707) descriptions are cut off with "…" in the session's skill listing, so their trailing trigger words are never seen. The other 13 are 953-1,489 and show in full | trim each to ≤1,500 chars, keeping the front-loaded triggers |
| BUGS "Maintenance" note | "eight skill frontmatter descriptions contain unquoted `word: word`" | measured: **seven** (analysis-toolkit, change-control, debugging-playbook, failure-archaeology, research-frontier, run-and-operate ×3, validation-and-qa) |

### 2.4 Replacement table (305 stale refs; HEAD 7f6bd05)

Column "check" (see 2.1): symbol = symbol confirmed next to the new line; blame = git-blame-anchored line map
(code text identical at both ends); def-lookup / manual = resolved by definition lookup or by hand. Each row is
one in-line substring edit; `apply.py` does them all. Ranges keep their shape.

#### .claude/skills/autocroissant-analysis-toolkit/SKILL.md (16 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 79 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | blame | self._gap_pattern = re_compile(r'\s{3,}') |
| 127 | `:1307` | `:1309` | symbol | problems.append("ABILITY TEXT NOT FOUND") |
| 127 | `psd_analyzer.py:772` | `psd_analyzer.py:774` | symbol | card.problems.append("NO ABILITY LAYER") |
| 129 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 130 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 148 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 287 | `(:1870` | `(:1872` | symbol | if num_new > 0 or traverser.changed: |
| 287 | `psd_analyzer.py:1868` | `psd_analyzer.py:1870` | blame | if num_new > 0: |
| 291 | `psd_analyzer.py:1678` | `psd_analyzer.py:1669` | symbol | num_new += 1 |
| 340 | `psd_analyzer.py:543-545` | `psd_analyzer.py:545-547` | blame | print("Git token found, API limited to 5000 requests/hour.") |
| 362 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 373 | `main.py:111-114` | `main.py:113-117` | symbol | queue_command(init_reminder) |
| 376 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 378 | `main.py:759-786` | `main.py:772-799` | blame | async def process_dispatch_queue(): |
| 390 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 443 | `psd_analyzer.py:1673-1674` | `psd_analyzer.py:1664-1665` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |

#### .claude/skills/autocroissant-architecture-contract/SKILL.md (47 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 39 | `main.py:750` | `main.py:763` | symbol | async def process_command_queue(): |
| 39 | `main.py:759` | `main.py:772` | symbol | async def process_dispatch_queue(): |
| 40 | `main.py:791` | `main.py:804` | blame | async def process_edit_queue(): |
| 48 | `main.py:750-754` | `main.py:763-767` | symbol | async def process_command_queue(): |
| 49 | `main.py:759-786` | `main.py:772-799` | symbol | async def process_dispatch_queue(): |
| 50 | `main.py:791-803` | `main.py:804-816` | symbol | async def process_edit_queue(): |
| 56 | `main.py:740-745` | `main.py:753-758` | symbol | async def run_command(func, args, kwargs): |
| 59 | `main.py:742-743` | `main.py:755-756` | symbol | if kwargs.get("executor"): |
| 86 | `main.py:778-786` | `main.py:791-799` | symbol | except (HTTPException, AttributeError): |
| 93 | `main.py:101-119` | `main.py:103-122` | symbol | async def on_ready(): |
| 94 | `main.py:111-114` | `main.py:113-117` | symbol | queue_command(init_reminder) |
| 97 | `main.py:204` | `main.py:207` | symbol | bind_reminder_client(client)  # lets the scheduler skip remi |
| 106 | `main.py:116-118` | `main.py:119-121` | manual | process_command_queue.start() |
| 106 | `main.py:145-147` | `main.py:148-150` | symbol | process_command_queue.stop() |
| 106 | `main.py:157-159` | `main.py:160-162` | symbol | process_command_queue.stop() |
| 106 | `main.py:194-196` | `main.py:197-199` | manual | process_command_queue.stop() |
| 108 | `psd_analyzer.py:2288` | `psd_analyzer.py:2295` | symbol | stats_db = StatsDatabase() |
| 108 | `query_card.py:48-51` | `query_card.py:50-53` | blame | if GIT_TOKEN: |
| 108 | `query_card.py:714` | `query_card.py:718` | symbol | card_repo = CardRepository() |
| 108 | `type_emojis.py:26` | `type_emojis.py:30` | symbol | type_emojis: dict[str, str] = {} |
| 109 | `psd_analyzer.py:26` | `psd_analyzer.py:25` | symbol | from commands.query_card import card_repo |
| 109 | `query_card.py:15` | `query_card.py:16` | symbol | from commands.type_emojis import render_type_tokens |
| 110 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 111 | `main.py:154-156` | `main.py:157-159` | symbol | if perms_check(interaction): |
| 122 | `main.py:256-258` | `main.py:259-261` | symbol | async def slash_query_name(interaction: Interaction, |
| 123 | `main.py:251` | `main.py:254` | symbol | query_name = to_thread(query_name) |
| 132 | `main.py:759` | `main.py:772` | symbol | async def process_dispatch_queue(): |
| 144 | `main.py:363` | `main.py:366` | symbol | await interaction.response.defer() |
| 146 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 148 | `psd_analyzer.py:1460-1464` | `psd_analyzer.py:1462-1466` | blame | if not force_update and not self.only and _tree_etags.get(re |
| 152 | `psd_analyzer.py:1379-1391` | `psd_analyzer.py:1381-1393` | blame | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 155 | `psd_analyzer.py:1429-1440, 1657-1701` | `psd_analyzer.py:1431-1442, 1648-1692` | blame | pool = ThreadPoolExecutor(DOWNLOAD_WORKERS) |
| 160 | `psd_analyzer.py:2288` | `psd_analyzer.py:2295` | symbol | stats_db = StatsDatabase() |
| 180 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 194 | `main.py:740` | `main.py:753` | symbol | async def run_command(func, args, kwargs): |
| 250 | `psd_analyzer.py:26` | `psd_analyzer.py:25` | blame | from commands.query_card import card_repo |
| 252 | `query_card.py:379` | `query_card.py:383` | symbol | from commands.psd_analyzer import get_stats_as_dict |
| 254 | `query_card.py:15` | `query_card.py:16` | symbol | from commands.type_emojis import render_type_tokens |
| 255 | `type_emojis.py:60` | `type_emojis.py:98` | manual | from commands.query_card import card_repo  # lazy, like quer |
| 285 | `main.py:304` | `main.py:307` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 286 | `main.py:314` | `main.py:317` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 286 | `main.py:597` | `main.py:610` | symbol | async def slash_delete_song(interaction: Interaction, |
| 287 | `main.py:604` | `main.py:617` | symbol | async def slash_delete_song(interaction: Interaction): |
| 297 | `main.py:779` | `main.py:792` | symbol | send_kwargs.pop("ephemeral", None) |
| 302 | `psd_analyzer.py:1667-1668, 1695-1701` | `psd_analyzer.py:1658-1659, 1686-1692` | symbol | finally: |
| 320 | `main.py:165-172` | `main.py:168-175` | blame | git_pull = to_thread(git_pull) |
| 352 | `main.py:742-743` | `main.py:755-756` | symbol | if kwargs.get("executor"): |

#### .claude/skills/autocroissant-build-and-env/SKILL.md (3 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 55 | `commands/psd_analyzer.py:22` | `commands/psd_analyzer.py:21` | blame | import pandas as pd |
| 223 | `psd_analyzer.py:543-545` | `psd_analyzer.py:545-547` | symbol | print("Git token found, API limited to 5000 requests/hour.") |
| 223 | `query_card.py:49-51` | `query_card.py:51-53` | symbol | print("Git token found, API limited to 5000 requests/hour.") |

#### .claude/skills/autocroissant-change-control/SKILL.md (13 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 19 | `global_config.py:5-8` | `global_config.py:5-9` | blame | ALIAS_PKL       = "aliases.pkl" |
| 45 | `main.py:165-198` | `main.py:168-201` | blame | git_pull = to_thread(git_pull) |
| 61 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 83 | `main.py:188` | `main.py:191` | symbol | force_reset="If True, discard all local changes and reset to |
| 154 | `main.py:352-364` | `main.py:355-367` | symbol | update_stats = to_thread(update_stats) |
| 181 | `main.py:178-180` | `main.py:181-183` | blame | if perms_check(interaction): |
| 182 | `main.py:659-672` | `main.py:672-685` | symbol | sync_type_emojis = to_thread(sync_type_emojis) |
| 191 | `help.py:114` | `help.py:118` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 204 | `main.py:100-119` | `main.py:102-122` | symbol | @client.event |
| 206 | `main.py:92-96` | `main.py:94-98` | symbol | async def sync_guild_commands(guild): |
| 207 | `main.py:651` | `main.py:664` | symbol | @tree.command(name="sync_global", description="Sync all slas |
| 229 | `:214` | `:216` | symbol | class CardStats: |
| 229 | `psd_analyzer.py:444` | `psd_analyzer.py:446` | symbol | class CardInfo: |

#### .claude/skills/autocroissant-config-and-flags/SKILL.md (21 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 54 | `commands/query_card.py:48-51` | `commands/query_card.py:50-53` | blame | if GIT_TOKEN: |
| 60 | `main.py:811` | `main.py:824` | symbol | client.run(TOKEN) |
| 126 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 175 | `commands/psd_analyzer.py (lines 31-63` | `commands/psd_analyzer.py (lines 33-65` | blame | # ======================== |
| 182 | `psd_analyzer.py:1693` | `psd_analyzer.py:1684` | symbol | if num_updated % UPDATE_RATE == 0: |
| 185 | `psd_analyzer.py:2208, 2215` | `psd_analyzer.py:2215, 2222` | symbol | filename = f"{EXPORTED_STATS_NAME}.txt" |
| 186 | `psd_analyzer.py:2269` | `psd_analyzer.py:2276` | symbol | filename = f"{EXPORTED_RULES_NAME}.txt" |
| 187 | `psd_analyzer.py:2065, 2150, 2199` | `psd_analyzer.py:2072, 2157, 2206` | symbol | existing_columns = [col for col in COLUMN_ORDER if col in df |
| 198 | `psd_analyzer.py:1255-1261` | `psd_analyzer.py:1257-1263` | symbol | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 199 | `psd_analyzer.py:1262-1281` | `psd_analyzer.py:1264-1283` | symbol | ABILITY_EXCLUSIONS: set[str] = { |
| 206 | `commands/query_card.py (lines 19-24` | `commands/query_card.py (lines 21-26` | blame | # ======================== |
| 214 | `query_card.py:714` | `query_card.py:718` | blame | card_repo = CardRepository() |
| 246 | `commands/type_emojis.py (lines 14-26` | `commands/type_emojis.py (lines 18-30` | symbol | # ======================== |
| 293 | `setup.py (lines 8-45` | `setup.py (lines 8-44` | blame | # Files that should NOT be compiled (will cause issues) |
| 319 | `query_card.py:670-687` | `query_card.py:674-691` | symbol | async def set_match_ratio(interaction: Interaction, value: O |
| 320 | `query_card.py:690-708` | `query_card.py:694-712` | symbol | async def set_repository(interaction: Interaction, new_repo: |
| 335 | `main.py:314` | `main.py:317` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 343 | `main.py:361` | `main.py:364` | blame | force_update: Optional[bool] = False, |
| 343 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | blame | force_update: bool = False, |
| 423 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 423 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | symbol | force_update: bool = False, |

#### .claude/skills/autocroissant-debugging-playbook/SKILL.md (23 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 34 | `commands/psd_analyzer.py:1827` | `commands/psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 44 | `psd_analyzer.py:1673` | `psd_analyzer.py:1664` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |
| 45 | `psd_analyzer.py:581` | `psd_analyzer.py:583` | symbol | def prune_clean_cards(self) -> None: |
| 47 | `psd_analyzer.py:1888-1889` | `psd_analyzer.py:1895-1896` | symbol | queue_command(card_repo.populate_files, traverser.tree) |
| 100 | `main.py:758-786` | `main.py:771-799` | blame | @tasks.loop(seconds=1) |
| 107 | `main.py:779` | `main.py:792` | blame | send_kwargs.pop("ephemeral", None) |
| 111 | `main.py:790-803` | `main.py:803-816` | blame | @tasks.loop(seconds=1) |
| 120 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 121 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 123 | `main.py:779` | `main.py:792` | symbol | send_kwargs.pop("ephemeral", None) |
| 124 | `main.py:796-803` | `main.py:809-816` | blame | except (HTTPException, AttributeError): |
| 124 | `psd_analyzer.py:1731` | `psd_analyzer.py:1722` | symbol | def _send_progress(interaction: Optional[Interaction], count |
| 124 | `psd_analyzer.py:34` | `psd_analyzer.py:36` | symbol | UPDATE_RATE = 25 |
| 131 | `main.py:145-147, 157-159, 194-196` | `main.py:148-150, 160-162, 197-199` | blame/symbol | process_command_queue.stop() |
| 141 | `main.py:92-109` | `main.py:94-111` | blame | async def sync_guild_commands(guild): |
| 152 | `main.py:304` | `main.py:307` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 152 | `main.py:597` | `main.py:610` | symbol | async def slash_delete_song(interaction: Interaction, |
| 179 | `main.py:810-816` | `main.py:823-829` | symbol | try: |
| 180 | `main.py:87` | `main.py:89` | symbol | intents.message_content = True |
| 182 | `psd_analyzer.py:554-579` | `psd_analyzer.py:556-581` | symbol | def load(self) -> None: |
| 229 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 234 | `main.py:166` | `main.py:169` | symbol | @tree.command(name="pull", description="Does a hard reset, t |
| 242 | `main.py:751` | `main.py:764` | blame | queue_command(check_reminder) |

#### .claude/skills/autocroissant-docs-and-style/SKILL.md (39 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 77 | `help.py:114-134` | `help.py:118-138` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 79 | `help.py:122-123` | `help.py:126-127` | symbol | elif help_wanted == "stats": |
| 80 | `help.py:124-125` | `help.py:128-129` | symbol | else: |
| 83 | `main.py:132-136` | `main.py:135-139` | blame | Choice(name="text", value="text"), |
| 90 | `main.py:131-137` | `main.py:134-140` | symbol | @app_commands.choices(help_type=[ |
| 122 | `main.py:439` | `main.py:452` | symbol | @tree.command(name="view_old_metadata", description="View ol |
| 123 | `main.py:676` | `main.py:689` | symbol | @tree.command(name="list_guild_members", description="Lists  |
| 124 | `main.py:691` | `main.py:704` | symbol | @tree.command(name="list_guild_channels", description="Lists |
| 125 | `main.py:706` | `main.py:719` | symbol | @tree.command(name="get_channel_messages", description="Get  |
| 129 | `main.py:132-136` | `main.py:135-139` | blame | Choice(name="text", value="text"), |
| 141 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 150 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 153 | `help.py:76` | `help.py:80` | manual | "- only_ability: Export only ability text (default: True)\n" |
| 155 | `main.py:374` | `main.py:387` | symbol | only_ability: Optional[bool] = False, |
| 156 | `main.py:370` | `main.py:383` | blame | only_ability='Should we only output the ability of cards?', |
| 168 | `main.py:660-672` | `main.py:673-687` | manual | @tree.command(name="sync_emojis", description="Uploads missi |
| 173 | `main.py:335` | `main.py:338` | symbol | blend_mode='How to blend: slice, or average. Default = slice |
| 173 | `main.py:615` | `main.py:628` | symbol | num='The number of messages to delete. Default = 100.', |
| 174 | `main.py:543` | `main.py:556` | symbol | multiplier='The multiplier of the volume. 0.5 = half as loud |
| 175 | `main.py:627` | `main.py:640` | symbol | @tree.command(name="list_guilds", description="Lists all gui |
| 176 | `main.py:131-137` | `main.py:134-140` | symbol | @app_commands.choices(help_type=[ |
| 176 | `main.py:337-340` | `main.py:340-343` | symbol | @app_commands.choices(blend_mode=[ |
| 176 | `main.py:459-464` | `main.py:472-477` | symbol | @app_commands.choices(field=[ |
| 178 | `main.py:302` | `main.py:305` | symbol | value='If empty, posts the current tolerance. Otherwise sets |
| 178 | `main.py:312` | `main.py:315` | symbol | new_repo='If empty, posts the current repository. Otherwise  |
| 179 | `main.py:640, 680, 713-714` | `main.py:653, 693, 726-727` | symbol | guild_id: str, |
| 199 | `psd_analyzer.py:29` | `psd_analyzer.py:31` | blame | getLogger("psd_tools").setLevel(CRITICAL) |
| 202 | `query_card.py:25` | `query_card.py:26` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 209 | `psd_analyzer.py:2288-2289` | `psd_analyzer.py:2295-2296` | symbol | stats_db = StatsDatabase() |
| 210 | `query_card.py:714-715` | `query_card.py:718-719` | symbol | card_repo = CardRepository() |
| 220 | `main.py:123-125` | `main.py:126-128` | blame | #################### |
| 247 | `query_card.py:61-69` | `query_card.py:63-71` | symbol | """ |
| 269 | `help.py:129` | `help.py:133` | symbol | help_msg += f"{cmd}\n{desc}\n\n" |
| 270 | `main.py:96` | `main.py:98` | symbol | print(f"Synced commands to guild: {guild.name} ({guild.id})" |
| 278 | `help.py:114-134` | `help.py:118-138` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 287 | `query_card.py:25` | `query_card.py:26` | symbol | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 483 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 488 | `help.py:76` | `help.py:80` | symbol | "- only_ability: Export only ability text (default: True)\n" |
| 488 | `main.py:374` | `main.py:387` | symbol | only_ability: Optional[bool] = False, |

#### .claude/skills/autocroissant-failure-archaeology/SKILL.md (7 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 132 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | symbol | self._gap_pattern = re_compile(r'\s{3,}') |
| 156 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | symbol | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 233 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 301 | `psd_analyzer.py:1262` | `psd_analyzer.py:1264` | symbol | ABILITY_EXCLUSIONS: set[str] = { |
| 306 | `psd_analyzer.py:841-844` | `psd_analyzer.py:843-846` | symbol | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 307 | `psd_analyzer.py:50` | `psd_analyzer.py:52` | symbol | MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error'] |
| 399 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |

#### .claude/skills/autocroissant-psd-extraction-campaign/SKILL.md (16 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 42 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | symbol | self._gap_pattern = re_compile(r'\s{3,}') |
| 43 | `psd_analyzer.py:841` | `psd_analyzer.py:843` | symbol | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 45 | `psd_analyzer.py:38,743` | `psd_analyzer.py:40,745` | symbol | card_mid_y = int(psd.height * TYPE_REGION_RATIO) |
| 46 | `psd_analyzer.py:1206` | `psd_analyzer.py:1208` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 47 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 304 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | symbol | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 408 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 483 | `psd_analyzer.py:1305` | `psd_analyzer.py:1307` | symbol | not card.ability and |
| 531 | `main.py:388` | `main.py:401` | symbol | @tree.command(name="update_metadata", description="Edit meta |
| 531 | `psd_analyzer.py:1947` | `psd_analyzer.py:1954` | symbol | def manual_metadata_entry(interaction: Interaction, |
| 536 | `psd_analyzer.py:1997` | `psd_analyzer.py:2004` | symbol | stats_db.stats[name] = CardInfo( |
| 537 | `psd_analyzer.py:1379` | `psd_analyzer.py:1381` | symbol | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 580 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 582 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | symbol | force_update: bool = False, |
| 584 | `psd_analyzer.py:1495` | `psd_analyzer.py:1498` | symbol | def _populate_types_from_local(self, local_path: str) -> Non |
| 608 | `main.py:388` | `main.py:401` | symbol | @tree.command(name="update_metadata", description="Edit meta |

#### .claude/skills/autocroissant-research-frontier/SKILL.md (14 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 43 | `commands/query_card.py:266` | `commands/query_card.py:287` | symbol | def rulebook_search_engine(self, search_text: str) -> DataFr |
| 44 | `main.py:285` | `main.py:289` | blame | @tree.command(name="query_rulebook", description="Search the |
| 44 | `query_card.py:472` | `query_card.py:564` | blame | def query_rulebook(interaction: Interaction, |
| 46 | `query_card.py:194` | `query_card.py:215` | symbol | def ability_search_engine(self, ability: str) -> DataFrame: |
| 57 | `commands/psd_analyzer.py:1718` | `commands/psd_analyzer.py:2269` | def-lookup | def export_rulebook_to_file(interaction: Interaction) -> Non |
| 57 | `main.py:382` | `main.py:394` | symbol | @tree.command(name="export_rulebook", description="Exports t |
| 58 | `main.py:369` | `main.py:381` | symbol | @tree.command(name="export_cards", description="Exports all  |
| 58 | `psd_analyzer.py:1623` | `psd_analyzer.py:2174` | def-lookup | def export_stats_to_file(interaction: Interaction, |
| 59 | `psd_analyzer.py:593` | `psd_analyzer.py:1208` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 141 | `main.py:495` | `main.py:508` | blame | await play_music(interaction, song, play_next) |
| 218 | `psd_analyzer.py:1206-1250` | `psd_analyzer.py:1208-1252` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 282 | `commands/psd_analyzer.py:123` | `commands/psd_analyzer.py:446` | symbol | class CardInfo: |
| 292 | `psd_analyzer.py:138` | `psd_analyzer.py:472` | symbol | def to_dict(self) -> dict: |
| 293 | `psd_analyzer.py:210` | `psd_analyzer.py:549` | symbol | def save(self) -> None: |

#### .claude/skills/autocroissant-research-methodology/SKILL.md (7 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 155 | `psd_analyzer.py:1870` | `psd_analyzer.py:1872` | symbol | if num_new > 0 or traverser.changed: |
| 196 | `psd_analyzer.py:282` | `psd_analyzer.py:621` | blame | "MDW": lambda: {"type": CardType.MDW.value, "ability": None} |
| 196 | `psd_analyzer.py:31` | `psd_analyzer.py:41` | symbol | EXCLUDE_FOLDERS = ["Markers", "MDW"] |
| 231 | `psd_analyzer.py:42` | `psd_analyzer.py:44` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 232 | `query_card.py:25` | `query_card.py:26` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 232 | `type_emojis.py:17` | `type_emojis.py:21` | manual | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 267 | `analytics.py:209` | `analytics.py:426` | manual | func = slash_registry.get(name) |

#### .claude/skills/autocroissant-run-and-operate/SKILL.md (32 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 32 | `main.py:9` | `main.py:11` | blame | from config import TOKEN |
| 46 | `psd_analyzer.py:542-545` | `psd_analyzer.py:544-547` | blame | if GIT_TOKEN: |
| 47 | `query_card.py:48-51` | `query_card.py:50-53` | blame | if GIT_TOKEN: |
| 48 | `main.py:102` | `main.py:104` | symbol | print(f'We have logged in as {client.user}') |
| 50 | `main.py:116-119` | `main.py:119-122` | blame | process_command_queue.start() |
| 51 | `main.py:96` | `main.py:98` | blame | print(f"Synced commands to guild: {guild.name} ({guild.id})" |
| 67 | `main.py:101-119` | `main.py:103-122` | symbol | async def on_ready(): |
| 71 | `main.py:116-118` | `main.py:119-121` | blame | process_command_queue.start() |
| 77 | `psd_analyzer.py:2289` | `psd_analyzer.py:2296` | symbol | def init_psd() -> None: |
| 78 | `query_card.py:715` | `query_card.py:719` | symbol | def init_query() -> None: |
| 79 | `type_emojis.py:218` | `type_emojis.py:389` | symbol | async def init_type_emojis(client: Client) -> None: |
| 83 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 102 | `main.py:152-162` | `main.py:155-165` | symbol | @tree.command(name="stop_bot", description="Stop the bot.") |
| 103 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 104 | `main.py:812-813` | `main.py:825-826` | blame | except KeyboardInterrupt: |
| 129 | `main.py:92-96` | `main.py:94-98` | symbol | async def sync_guild_commands(guild): |
| 133 | `main.py:651-656` | `main.py:664-669` | symbol | @tree.command(name="sync_global", description="Sync all slas |
| 172 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 194 | `main.py:186-199` | `main.py:189-202` | blame | @tree.command(name="update", description="Push pickles, pull |
| 218 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 265 | `main.py:352-364` | `main.py:355-367` | blame | update_stats = to_thread(update_stats) |
| 265 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | blame | def update_stats(interaction: Optional[Interaction] = None, |
| 291 | `main.py:360-362` | `main.py:363-365` | blame | output_problematic_cards: Optional[bool] = True, |
| 302 | `psd_analyzer.py:1745` | `psd_analyzer.py:1736` | blame | msg = f"{num_new} cards changed or new, {num_old} unchanged, |
| 302 | `psd_analyzer.py:34,1693` | `psd_analyzer.py:36,1684` | blame | if num_updated % UPDATE_RATE == 0: |
| 309 | `psd_analyzer.py:1868-1871` | `psd_analyzer.py:1870-1873` | blame | if num_new > 0: |
| 327 | `psd_analyzer.py:1410-1440` | `psd_analyzer.py:1412-1442` | symbol | def _blob(self, sha: str, stream: bool = False): |
| 377 | `psd_analyzer.py:1429-1440` | `psd_analyzer.py:1431-1442` | symbol | def _download(self, path: str, sha: str) -> Optional[str]: |
| 385 | `global_config.py:10` | `global_config.py:14` | symbol | TIMEZONE        = ZoneInfo("America/Los_Angeles") |
| 386 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 390 | `main.py:206` | `main.py:209` | symbol | @tree.command(name="set_reminder", description="Set a remind |
| 470 | `main.py:612-623` | `main.py:625-636` | symbol | @tree.command(name="purge", description="Deletes messages.") |

#### .claude/skills/autocroissant-validation-and-qa/SKILL.md (25 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 51 | `psd_analyzer.py:831` | `psd_analyzer.py:1257` | manual | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 51 | `psd_analyzer.py:838` | `psd_analyzer.py:1264` | manual | ABILITY_EXCLUSIONS: set[str] = { |
| 63 | `psd_analyzer.py:199` | `psd_analyzer.py:538` | symbol | self.stats: dict[str, CardInfo] = {} |
| 65 | `psd_analyzer.py:487` | `psd_analyzer.py:670` | blame | elif len(folders) > 2: |
| 66 | `psd_analyzer.py:521` | `psd_analyzer.py:704` | symbol | existing_card = self.stats_db.stats.get(name) |
| 67 | `psd_analyzer.py:45` | `psd_analyzer.py:52` | symbol | MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error'] |
| 69 | `psd_analyzer.py:595` | `psd_analyzer.py:781` | blame |  |
| 71 | `psd_analyzer.py:628-660` | `psd_analyzer.py:814-846` | blame | shape = layer.numpy("shape") |
| 76 | `psd_analyzer.py:230` | `psd_analyzer.py:500` | manual | result["problem"] = self.problems |
| 80 | `psd_analyzer.py:863-898` | `psd_analyzer.py:1051-1086` | symbol | Stars drawn on a card whose star count comes from the PSD. |
| 82 | `psd_analyzer.py:900` | `psd_analyzer.py:1325` | def-lookup | def _validate_stats(card: CardInfo) -> list[str]: |
| 84 | `psd_analyzer.py:1213` | `psd_analyzer.py:1744` | manual | def _format_problems( |
| 85 | `psd_analyzer.py:1165` | `psd_analyzer.py:1695` | manual | return (self._format_problems(problematic_cards), num_new +  |
| 90 | `psd_analyzer.py:892-894` | `psd_analyzer.py:1317-1319` | manual | if ("NO ABILITY LAYER" in card.problems and |
| 92 | `psd_analyzer.py:1155` | `psd_analyzer.py:1351` | blame | return problems |
| 156 | `psd_analyzer.py:379-380, 929` | `psd_analyzer.py:719-720, 1649` | blame | downloads = [pool.submit(self._download, path, sha) for path |
| 196 | `global_config.py:6-7` | `global_config.py:6-8` | blame | STATS_PKL       = "stats.pkl" |
| 234 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | blame | def update_stats(interaction: Optional[Interaction] = None, |
| 261 | `psd_analyzer.py:1379` | `psd_analyzer.py:1381` | symbol | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 263 | `psd_analyzer.py:1719` | `psd_analyzer.py:1710` | symbol | commits = repo.get_commits(path=path) |
| 271 | `psd_analyzer.py:1888-1889` | `psd_analyzer.py:1895-1896` | symbol | queue_command(card_repo.populate_files, traverser.tree) |
| 381 | `psd_analyzer.py:670` | `psd_analyzer.py:1285` | symbol | @staticmethod |
| 382 | `psd_analyzer.py:707` | `psd_analyzer.py:1324` | symbol | @staticmethod |
| 386 | `psd_analyzer.py:456` | `psd_analyzer.py:856` | blame | card.problems.append(f"MISSPELT TYPE: {layer.name.lower()}") |
| 481 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |

#### .claude/skills/impossibility-cards-reference/SKILL.md (42 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 38 | `commands/psd_analyzer.py:361` | `commands/psd_analyzer.py:701` | blame | name = splitext(basename(relative_path))[0].replace('_', ' ' |
| 44 | `commands/psd_analyzer.py:256-340` | `commands/psd_analyzer.py:595-679` | symbol | class CardClassifier: |
| 68 | `psd_analyzer.py:285, 290, 309, 335` | `psd_analyzer.py:624, 629, 648, 674` | blame | "stars": int(folders[len(folders) - 1].split()[0]), |
| 113 | `psd_analyzer.py:39` | `psd_analyzer.py:41` | symbol | EXCLUDE_FOLDERS = ["Markers", "MDW"] |
| 114 | `psd_analyzer.py:1648, 1568` | `psd_analyzer.py:1639, 1571` | blame | if (not m or any(folder in EXCLUDE_FOLDERS for folder in pat |
| 127 | `psd_analyzer.py:428-456` | `psd_analyzer.py:829-856` | symbol | def _process_layer(self, layer, card: CardInfo, |
| 157 | `psd_analyzer.py:434` | `psd_analyzer.py:835` | blame | if layer.kind == "type": |
| 165 | `psd_analyzer.py:469, 407` | `psd_analyzer.py:869, 747` | manual | is_rulepage = "Rulebook" in relative_path |
| 166 | `psd_analyzer.py:460-467` | `psd_analyzer.py:868-889` | manual | engine_text = layer.engine_dict["Editor"]["Text"] |
| 179 | `psd_analyzer.py:529-539` | `psd_analyzer.py:1139-1149` | blame | def _process_abilities(self, abilities: list[tuple[str, Boun |
| 180 | `psd_analyzer.py:422` | `psd_analyzer.py:774` | blame | card.problems.append("NO ABILITY LAYER") |
| 187 | `psd_analyzer.py:1038` | `psd_analyzer.py:1498` | def-lookup | def _populate_types_from_local(self, local_path: str) -> Non |
| 194 | `psd_analyzer.py:442-447` | `psd_analyzer.py:843-848` | blame | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 196 | `psd_analyzer.py:30, 405` | `psd_analyzer.py:40, 745` | blame | card_mid_y = int(psd.height * TYPE_REGION_RATIO) |
| 205 | `psd_analyzer.py:473-515` | `psd_analyzer.py:997-1047` | blame | def _is_stat_layer(self, layer) -> bool: |
| 238 | `psd_analyzer.py:556-583` | `psd_analyzer.py:1171-1198` | symbol | @staticmethod |
| 248 | `psd_analyzer.py:593-637` | `psd_analyzer.py:1208-1252` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 254 | `psd_analyzer.py:30` | `psd_analyzer.py:40` | symbol | TYPE_REGION_RATIO = 0.5 |
| 255 | `psd_analyzer.py:348` | `psd_analyzer.py:688` | blame | self._gap_pattern = re_compile(r'\s{3,}') |
| 256 | `psd_analyzer.py:557` | `psd_analyzer.py:1172` | blame | def _sort_by_position(items: list[tuple[str, BoundingBox]],  |
| 257 | `psd_analyzer.py:590` | `psd_analyzer.py:1205` | symbol | max_height = max(bboxes[len(bboxes) - 1][1].y // 3, card_mid |
| 258 | `psd_analyzer.py:538` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 273 | `psd_analyzer.py:632-635` | `psd_analyzer.py:1247-1250` | blame | if type_index < len(types) and result_lines: |
| 400 | `psd_analyzer.py:640-733` | `psd_analyzer.py:1255-1351` | symbol | class CardValidator: |
| 405 | `psd_analyzer.py:422` | `psd_analyzer.py:774` | blame | card.problems.append("NO ABILITY LAYER") |
| 405 | `psd_analyzer.py:455-456` | `psd_analyzer.py:855-856` | symbol | elif (layer.name.lower() in MISSPELT_CARD_TYPES and layer.is |
| 426 | `psd_analyzer.py:642-668` | `psd_analyzer.py:1257-1283` | blame | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 465 | `psd_analyzer.py:44-48` | `psd_analyzer.py:46-50` | symbol | COLUMN_ORDER = [ |
| 470 | `psd_analyzer.py:1706` | `psd_analyzer.py:1697` | blame | def _get_remote_timestamp(self, |
| 475 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 475 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 506 | `psd_analyzer.py:1673-1674` | `psd_analyzer.py:1664-1665` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |
| 507 | `psd_analyzer.py:581-590` | `psd_analyzer.py:583-592` | symbol | def prune_clean_cards(self) -> None: |
| 509 | `psd_analyzer.py:1367-1377` | `psd_analyzer.py:1369-1379` | symbol | def _update_old_stats_paths(self) -> None: |
| 519 | `:626` | `:718` | symbol | card_repo = CardRepository() |
| 519 | `commands/query_card.py:27` | `commands/query_card.py:29` | symbol | @dataclass |
| 526 | `query_card.py:23` | `query_card.py:25` | symbol | DEFAULT_MATCH_RATIO = 0.6 |
| 538 | `query_card.py:526-579` | `query_card.py:618-671` | manual | async def manage_alias(interaction: Interaction, key: Option |
| 543 | `query_card.py:194-264` | `query_card.py:215-285` | symbol | def ability_search_engine(self, ability: str) -> DataFrame: |
| 577 | `query_card.py:257-260` | `query_card.py:278-281` | symbol | scores = self.cards_dff["ability"].apply( |
| 582 | `query_card.py:266-290` | `query_card.py:287-311` | symbol | def rulebook_search_engine(self, search_text: str) -> DataFr |
| 584 | `main.py:268-281` | `main.py:272-285` | symbol | @tree.command(name="query_ability", description="Query a car |


### 2.5 Kept as written (historical or dated refs, 28)

- autocroissant-analysis-toolkit:167 `psd_analyzer.py:1002`
- autocroissant-diagnostics-and-tooling:288 `commands/psd_analyzer.py:398`
- autocroissant-failure-archaeology:42 `main.py:365`
- autocroissant-failure-archaeology:48 `psd_analyzer.py:1173`
- autocroissant-failure-archaeology:48 `psd_analyzer.py:1207`
- autocroissant-failure-archaeology:49 `psd_analyzer.py:991`
- autocroissant-failure-archaeology:50 `psd_analyzer.py:1002`
- autocroissant-failure-archaeology:97 `psd_analyzer.py:1226`
- autocroissant-failure-archaeology:338 `main.py:524`
- autocroissant-failure-archaeology:370 `main.py:365`
- autocroissant-failure-archaeology:385 `psd_analyzer.py:1178-1181`
- autocroissant-failure-archaeology:392 `psd_analyzer.py:1079`
- autocroissant-failure-archaeology:399 `psd_analyzer.py:1401`
- autocroissant-failure-archaeology:475 `main.py:365`
- autocroissant-failure-archaeology:476 `psd_analyzer.py:1399`
- autocroissant-failure-archaeology:506 `main.py:361-365`
- autocroissant-failure-archaeology:506 `psd_analyzer.py:1287`
- autocroissant-failure-archaeology:609 `psd_analyzer.py:1226`
- autocroissant-psd-extraction-campaign:97 `psd_analyzer.py:1226`
- autocroissant-psd-extraction-campaign:575 `psd_analyzer.py:1226`
- autocroissant-research-frontier:163 `main.py:571 [code left main]`
- autocroissant-research-methodology:57 `commands/psd_analyzer.py:1002`
- autocroissant-research-methodology:99 `psd_analyzer.py:1226`
- autocroissant-research-methodology:153 `psd_analyzer.py:1046`
- autocroissant-research-methodology:194 `psd_analyzer.py:642-668`
- autocroissant-run-and-operate:58 `query_card.py:401`
- autocroissant-run-and-operate:377 `psd_analyzer.py:1668`
- autocroissant-run-and-operate:473 `main.py:86-87`

---

## 3. docs/BUGS_AND_IMPROVEMENTS.md: rows already fixed or fixed by this pass

### 3.1 Status of every open row (checked against HEAD 7f6bd05)

| Row | State today | Evidence | Action |
|---|---|---|---|
| C14 sha links | **FIXED** in e13d754 | `get_card_url` → `raw_url(self.repository, path, self.blobs.get(path))  # pinned to the card's last commit` (query_card.py:442). The diff removed `.../{self.repository}/main/{quote(...)}` | delete the row; archaeology 10.17 |
| V3 Cython on a host | done (its own row says "Nothing" missing) | | delete the row; add it to "Retired" |
| §2.3 item 4 (C12) | done | | delete the item |
| §2.4 items 2-5 (Tasks 1, 1b, 2, 2b) | done | e13d754 (sha links + emoji sync), reminder f590a57e removed; bad0496, ce94585, c4caaf9 (TTS save); TTSCardMaker 0d49893 (Table tab) per NEXT_SESSION_PLAN statuses | replace with one "done" line; archaeology 10.18-10.19 |
| §2.4 item 1 | stale premise | "else on the jetson once it has Python 3.11+ (Task 0)": the jetson runs 3.14 since 2026-10-20 | edit (3.2) |
| C1, C2, C3 | OPEN; fixed by this pass (docs) | part 1 | delete after the fix; archaeology 10.20 |
| B4 | OPEN | `await queue_message(...)` still at music_player.py:178 | charter (§2.5 Robustness); archaeology 10.21 when fixed |
| B5 | OPEN | main.py:791-799 / 809-816 fallbacks unchanged | charter; archaeology 10.22 when fixed |
| C6 | OPEN | `Intents.default()` + `message_content` only (main.py:88-89) | charter; archaeology 10.23 when fixed |
| B7 | OPEN | `populate_files` still does `response.json()` without a status check (query_card.py:80) | in scope if "timeouts/robustness on all HTTP calls" covers it; archaeology 10.24 if fixed |
| C5 duplicate names | OPEN | `slash_set_ratio` 307/317, `slash_delete_song` 610/617 | only if the pass renames them (change-control: rename the second def only, e.g. `slash_set_repo`, `slash_delete_all_music`) |
| D1-D7, V1, V2, §1.6 | unchanged by this pass | | none |

### 3.2 Exact edits to docs/BUGS_AND_IMPROVEMENTS.md

Line refs in rows that stay open until their fix lands:
- B4: `` `commands/music_player.py` (~line 177) `` → `` `commands/music_player.py` (line 178) ``.
- C1: `` `commands/help.py:10`, `main.py:166` `` → `` `commands/help.py:10`, `main.py:169` ``.
- C2: `(50 registered, 46 documented as of 2026-10-20)` → `(51 registered, 47 documented as of 2026-10-20)`.
- C3: `` `commands/help.py:76` vs `main.py:374` `` → `` `commands/help.py:80` vs `main.py:387` ``.

Now (already true):
1. **Retired** list, add after the "2026-10-20 operations" bullet:
   `- 2026-10-20 (later): C14 sha-pinned image links (10.17); emoji sync inside /update_stats, weekly /sync_emojis reminder removed (10.18); TTS save updater and /update_tts_save (10.19); V3 (Cython built on the jetson).`
2. Delete the C14 row in §1.3 and the V3 row in §1.5.
3. §2.3: delete `4. ~~C12 (declare Pillow)~~ done 2026-10-20: Pillow and PyNaCl are declared.`
4. §2.4: replace the heading and items with:
   ```markdown
   ### 2.4 Planned next (docs/NEXT_SESSION_PLAN.md; Task 3 first)

   1. PSD edits from website edits with the psd-tools fork, no Photoshop or browser (Task 3, the main effort; plan
      decision D4). Photopea is the reference renderer and emergency fallback until the fork meets its targets. Runs
      on GitHub Actions, else on the jetson (Python 3.14 venv since 2026-10-20; Task 0 step 6, removing 3.10, after a
      stable week).

   Done 2026-10-20: commit-sha image links (Task 1, e13d754), emoji sync inside /update_stats with the weekly
   reminder removed (Task 1b, e13d754), TTS save automation and /update_tts_save (Task 2: bad0496, ce94585, c4caaf9),
   the website Table tab (Task 2b, TTSCardMaker 0d49893).
   ```
5. Maintenance: `eight skill frontmatter descriptions contain unquoted` → `seven skill frontmatter descriptions contain unquoted`,
   then add: `Three descriptions (config-and-flags, run-and-operate, ttscardmaker-companion; 1,707-1,982 chars) are cut off in the skill listing; trim to ≤1,500.`

After this pass (only for what actually landed):
6. Retired list: `- <date> optimization pass: C1, C2, C3 (10.20); B4 (10.21); B5 (10.22); C6 (10.23); B7 (10.24).` (drop the ones that didn't land).
7. Delete the rows B4, B5, B7 (§1.1) and C1, C2, C3, C6 (§1.3), and C5 if it was renamed. Delete §2.3 items 1-3.
8. §2.5: delete the bullets that are done, or mark them done with the measured numbers. If all is done, replace
   §2.5 with `Done <date>: see failure-archaeology 10.20-10.24 and the pass report.` Its **Docs** bullet
   ("help.py matches the commands (C1-C3); skills' line numbers refreshed") is done once part 1 and the 2.1 re-run are applied.

### 3.3 failure-archaeology entries to add

Format is the file's own Entry 10 style (`### 10.N <title> (was <row>, <date>)` with **Symptom / Root cause /
Evidence / Status** bullets; "How to add an entry"). Also update the **Index** row 10 (append
"sha links, emoji sync in /update_stats, TTS save; <pass> docs/robustness fixes" and the commits), the Entry 10
intro sentence ("10.17-10.19 committed in e13d754, bad0496, ce94585, c4caaf9"), and the provenance row
`anything above 9fb3093` → the new HEAD.

Ready now (the facts are verified):

```markdown
### 10.17 Card image links pinned to commit shas (was C14, 2026-10-20)

- **Symptom**: `/query` and ability-search images linked `raw.githubusercontent.com/<repo>/main/<path>`, so an old
  Discord message showed whatever the PNG looks like today.
- **Root cause**: `get_card_url` built the URL from `main`, not from the commit that last changed that PNG.
- **Evidence**: e13d754, query_card.py: `-return f"https://raw.githubusercontent.com/{self.repository}/main/{quote(...)}"`
  → `+return raw_url(self.repository, path, self.blobs.get(path))`; new commands/repo_history.py walks the history
  clone (`~/Documents/TTSCardMaker-history.git`) for {path: (blob, commit)} into repo_history.json; refreshed at
  startup and on /update_stats with a changed tree; falls back to `main` when a path is unknown.
- **Status**: fixed in e13d754.

### 10.18 Emoji sync moved inside /update_stats; weekly reminder retired (was Task 1b, 2026-10-20)

- **Symptom**: the startup sync only uploaded MISSING emojis, so an edited icon kept its old emoji. A weekly
  reminder (`f590a57e`, `/sync_emojis`, Sun 3 AM) covered new icons between restarts.
- **Root cause**: no record of which icon blob each emoji was made from.
- **Evidence**: e13d754, type_emojis.py `sync_changed_emojis` (+ `_load_blobs`/`_save_blobs`, type_emojis.json):
  on a changed tree, upload or replace emojis whose blob changed (with `_no` variants) and delete the ones whose icon
  is gone; psd_analyzer.py `update_stats` queues it when `traverser.tree is not None`. Then the reminder was removed
  with `/remove_reminder reminder_id:f590a57e`.
- **Status**: fixed in e13d754. `/sync_emojis refresh:True` is still the manual full re-upload (V1).

### 10.19 TTS save kept current by the bot (was Task 2, 2026-10-20)

- **Symptom**: the Tabletop Simulator save drifted from the repo (new cards missing from decks, unpinned links,
  old names), and it was fixed by hand.
- **Root cause**: nothing tied the save to card changes.
- **Evidence**: bad0496 (`python -m commands.tts_save`), ce94585 (`/update_tts_save`; `update_after_stats` after a
  /update_stats whose tree touched images, TTS/decks.json, MDW/Card Edits.json or the save), c4caaf9 (when
  `GIT_TOKEN` can't write: commit through the history clone and push over SSH, no force).
- **Status**: shipped; operations in autocroissant-run-and-operate §TTS save.
```

For this pass (fill in `<hash>` and the measured lines; delete the ones that don't land):

```markdown
### 10.20 help.py and describe strings matched to the code (was C1, C2, C3, <date>)

- **Symptom**: /help said /pull "does a hard reset" (main.py also said "than"); 4 of 51 commands had no /help
  entry; /export_cards said only_ability defaults to True; /update force_reset, /react, /mass_replace and /purge
  entries named the wrong parameters or behaviour.
- **Root cause**: help.py and the describe strings were edited separately from the code (docs-and-style §2.4's
  checklist was skipped).
- **Evidence**: `git show <hash> -- commands/help.py main.py`; anchored `comm` audit: 0 missing (was 4).
  `grep -n 'hard reset' commands/help.py main.py` → only the /update force_reset lines.
- **Status**: fixed in <hash>.

### 10.21 /play outside a voice channel raised TypeError (was B4, <date>)

- **Symptom**: `/play` with the caller not in voice errored instead of replying "You are not in a voice channel."
- **Root cause**: `init_vc` did `await queue_message(...)`. queue_message is sync and returns None, so the await
  raised `TypeError: object NoneType can't be used in 'await' expression`.
- **Evidence**: `git show <hash> -- commands/music_player.py` (the `await` dropped at line 178); <how verified:
  e.g. FakeInteraction with user.voice None → message queued, no exception>.
- **Status**: fixed in <hash>.

### 10.22 Dispatch/edit loops no longer die on one bad message (was B5, <date>)

- **Symptom**: one poisoned send (deleted channel → `client.get_channel()` None → AttributeError inside the
  fallback; or >2000 chars failing twice) stopped `process_dispatch_queue`/`process_edit_queue` for good; the bot
  went silent until a restart.
- **Root cause**: a `tasks.loop` stops on an uncaught exception, and the fallback path was outside any try.
- **Evidence**: `git show <hash> -- main.py`; <experiment: queue a message for a missing channel id → logged,
  next message still delivered>.
- **Status**: fixed in <hash>. The 2026-10-19 `_channel_missing` reminder guard stays.

### 10.23 /list_guild_members and the members intent (was C6, <date>)

- **Symptom**: `/list_guild_members` always answered "Intents.members must be enabled to use this."
- **Root cause**: main.py built `Intents.default()` + `message_content` only.
- **Evidence**: `git show <hash> -- main.py commands/management.py commands/help.py`; <which fix: intent enabled
  in code and in the Developer Portal, or the command removed>.
- **Status**: fixed in <hash>.

### 10.24 Startup survives a GitHub 5xx (was B7, <date>)

- **Symptom**: after a restart during a GitHub outage, /query found nothing until the next /update_stats and the
  emoji sync failed (seen on the jetson 2026-10-20).
- **Root cause**: `populate_files` called `response.json()` on the error page (JSONDecodeError in `run_command`).
- **Evidence**: `git show <hash> -- commands/query_card.py commands/type_emojis.py`; <experiment: fake 503 →
  retried, last good tree kept>.
- **Status**: fixed in <hash>.
```

If C5 is renamed in this pass, add `### 10.25 Duplicate slash function names renamed (was C5)`, giving the old and new
names and noting that registration was unaffected (decorators bind at def time).
