---
name: autocroissant-docs-and-style
description: Load this skill whenever the task touches AutoCroissant's documentation or code conventions — trigger phrases include "help text", "/help", "update the docs", "describe strings", "app_commands.describe", "docstring", "style", "conventions", "house style", "new module", "module layout", "add a file to commands/", "section banners", "__all__", "import style", "writing a skill", "new skill", "update a skill", "provenance", or "the docs are wrong / out of date". The docs of record here are (1) the six command dictionaries in commands/help.py (user-facing /help), (2) the @app_commands.describe strings in main.py (Discord UI hints), and (3) this .claude/skills/ library itself; this skill owns their formats, the keep-them-in-sync checklist, the known doc-vs-code drift instances (e.g. /pull claims a hard reset that git_pull never does; 4 commands missing from help.py), the house code style (module skeleton with # ==== banners, from-imports, x[len(x)-1] in cythonized files, queue_* discipline, Args/Returns docstrings), ready-to-copy templates for a help.py entry, a new module, and a SKILL.md, and the rules for maintaining skill frontmatter, provenance sections, and the one-home-per-fact ownership map.
---

# AutoCroissant docs and style

House style and docs-of-record maintenance for a personal Discord bot with **no
tests, no CI, no README**. Because there is no README, the documentation users
actually see is generated from code: the `/help` command reads dictionaries in
`commands/help.py`, and Discord's slash-command UI shows the
`@app_commands.describe` strings from `main.py`. Those two surfaces plus this
skill library are the only docs this project has — keeping them truthful is the
whole job. Facts and line numbers verified 2026-07-11 (repo at commit 284d13c);
re-verify with the commands in "Provenance and maintenance" before trusting a
line number.

The repo was READ-ONLY during this skill's authoring, so the drift found below
is **recorded, not fixed**. Fixing it is a real change: route it through
`autocroissant-change-control` like any other edit.

---

## 1. The three docs of record

| Doc | Where | Audience | Renders as |
|---|---|---|---|
| Help dictionaries | `commands/help.py` (six dicts) | Discord users running `/help` | Code-fenced message, split at 1950 chars |
| Describe strings | `main.py` `@app_commands.describe(...)` on each command | Discord users typing a command | Grey hint text under each parameter in the Discord UI |
| Skill library | `.claude/skills/*/SKILL.md` | Engineers and models working on the repo | Loaded on demand by trigger match |

Nothing else counts. Comments in module code are helpful but nobody is promised
they exist; these three surfaces are the ones with a maintenance contract.

## 2. help.py anatomy

`commands/help.py` (156 lines as of 2026-07-11) is six module-level dicts, a
router function, and `__all__`. Note help.py is itself a cythonized module
(`setup.py` INCLUDE_FILES), so the style rules in section 5 apply to it too.

### 2.1 The six dicts

| Dict | help.py line | Entries (2026-07-11) | /help category |
|---|---|---|---|
| `general_commands` | 6 | 11 | `general` (also the fallback) |
| `text_commands` | 22 | 3 | `text` (reminders) |
| `card_commands` | 31 | 8 keys = 7 commands + 1 `"Query Syntax:"` pseudo-entry | `card` |
| `stats_commands` | 61 | 6 | `stats` |
| `ai_commands` | 80 | 6 | `ai` |
| `music_commands` | 102 | 17 | `music` |

Total: 51 keys, 50 real commands documented. The `"Query Syntax:"` key
(help.py:38) is a sanctioned pseudo-entry — a non-command key used to document
the `/query_ability` DSL. Pseudo-entries are allowed; they just must end with a
colon like real keys so they render uniformly.

### 2.2 Entry format

Key = the command signature string, `<angle>` for required, `[square]` for
optional, ending in `:`. Value = the description; multi-line via `\n` with
hyphen bullets, one bullet per parameter. Keys are padded with spaces so the
`:` column roughly aligns within a dict (cosmetic, but keep it). A real entry,
quoted exactly (help.py:17-19):

```python
    "/react <emojis> [message_id]:"             : "Adds reactions to a message.\n"
                                                  "- emojis: Space-separated emojis (👍 👎 or <:custom:123456>)\n"
                                                  "- message_id: Optional. If omitted, reacts to the most recent message in the channel.",
```

Conventions visible across the dicts: state defaults inline (`"(default 100)"`,
`"(default: all)"`), mark admin gating with `"(admin only)"`, keep the first
line a one-sentence summary and push parameter detail into bullets.

### 2.3 How print_help routes

`print_help(interaction, help_wanted)` (help.py:126-148) is a plain if/elif
chain: `"text"` → text_commands, `"card"` → card_commands, `"ai"` →
ai_commands, `"music"` → music_commands, `"stats"` → stats_commands
(help.py:136-137), and **anything else falls through to general_commands**
(else at help.py:138-139). So the code DOES route `"stats"` — it does not fall
to general — and the `"general"` choice works only via the fallback (there is
no explicit `elif help_wanted == "general"`). The slash command's Choice list
(main.py:136-143) offers exactly: text, card, ai, music, stats, general.

Rendering: entries are concatenated as `f"{cmd}\n{desc}\n\n"` inside a
` ``` ` code fence, split with `split_long_message` (1950-char BREAK_LEN), and
sent via `queue_message` — never a direct await (see section 5.6).

Adding a category = three edits in lockstep: new dict in help.py, new `elif` in
`print_help`, new `Choice` in main.py:136-143 — plus updating `/help`'s own
signature key (see drift instance D3 for what happens when you skip that).

### 2.4 THE CHECKLIST (the rule this skill exists to enforce)

Every new or changed slash command updates **all three** in the SAME change:

1. `main.py`: the `@tree.command(name=, description=)` + `@app_commands.describe(...)` strings.
2. `commands/help.py`: the entry in the correct category dict (matching
   signature, matching defaults, matching admin marker).
3. If behavior/defaults changed: re-read the existing help.py entry and
   describe strings and correct them.

Rationale: the audit below found 4 commands and 3 factual lies that
accumulated precisely because these were treated as separate chores. The full
add-a-slash-command procedure (sync, registration pattern, gates) is owned by
`autocroissant-change-control`; this checklist is the documentation slice of it.

## 3. Known drift (audited 2026-07-11 — recorded, not fixed)

### 3.1 Coverage drift: 4 registered commands have no help.py entry

main.py registers **54** commands; help.py documents **50**. Missing:

| Command | Registered at | Belongs in |
|---|---|---|
| `/view_old_metadata` | main.py:440 | `stats_commands` |
| `/list_guild_members` | main.py:750 | `general_commands` (admin only) |
| `/list_guild_channels` | main.py:765 | `general_commands` (admin only) |
| `/get_channel_messages` | main.py:780 | `general_commands` (admin only) |

Nothing in help.py refers to a nonexistent command (the reverse direction is
clean). **Extraction trap**: do NOT grep for bare `name=` — `Choice(name=...)`
lines (e.g. main.py:137-142) will pollute the list. Use the anchored pattern:

```bash
grep -oE '@tree\.command\(name="[^"]+"' main.py | sed 's/@tree.command(name="//;s/"//' | sort > /tmp/main_cmds.txt
grep -oE '^    "/[a-z_]+' commands/help.py | sed 's|    "/||' | sort -u > /tmp/help_cmds.txt
comm -23 /tmp/main_cmds.txt /tmp/help_cmds.txt   # registered but undocumented
comm -13 /tmp/main_cmds.txt /tmp/help_cmds.txt   # documented but unregistered
```

### 3.2 Behavior drift: the docs lie about the code (3 verified instances)

**D1 — /pull claims a hard reset that never happens.** help.py:10 says "Does a
hard reset, then a git pull" and main.py:172 says the same ("...a hard reset,
than a git pull" — note the "than" typo lives in the Discord UI). The actual
implementation `git_pull` (commands/update_bot.py:33-101) does `git.fetch()`
then a merge-based pull with pickles-ours/code-theirs conflict resolution —
**no reset of any kind**. A hard reset exists only in the separate
`git_reset_hard` (update_bot.py:104-113), reachable solely via
`/update force_reset:True` (update_bot.py:131-133). This lie is dangerous in
the scary direction: an admin might avoid `/pull` fearing it destroys local
pickle changes, when it is actually the safe merge path. Fix = docs-only edit
to help.py:10 and main.py:172; still route through change-control (description
edits change what Discord re-syncs).

**D2 — /export_cards documents the wrong default.** help.py:68 says
"only_ability: Export only ability text (default: True)". The signature says
`only_ability: Optional[bool] = False` (main.py:375). Code wins: the default
is False (full metadata export). The main.py describe string (main.py:371)
dodges the question by stating no default at all — which is how the lie
survived.

**D3 — /help's own entry omits its "stats" category.** help.py:7 advertises
`/help <text/card/ai/music/general>`, but the Choice list includes `stats`
(main.py:141) and `print_help` routes it (help.py:136-137). The docs
undersell the code: users reading `/help` never learn the stats category
exists. Classic "added a category, forgot the /help signature key" — the
exact lockstep edit section 2.3 mandates.

When you fix any of these, delete its row here and re-run the section 3.1
extraction to confirm 0 missing — this section must describe the current tree,
not history (history belongs to `autocroissant-failure-archaeology`).

## 4. Command description conventions (main.py describe strings)

Derived from the 54 live registrations; follow these when writing new ones.

| Convention | Real example (verified 2026-07-11) |
|---|---|
| State the default, in the string | `'How long to spend generating the image. Default = 50.'` (main.py:492); `'The number of messages to delete. Default = 100.'` (main.py:705) |
| State units and scale | `'How many pixels tall should the image be. Default = 512.'` (main.py:493); `'0.5 = half as loud, 1 = default, 2 = twice as loud'` (main.py:633) |
| Mark admin gating in the command description | `description="Lists all guilds the bot is currently a member of (admin only)."` (main.py:717; also 726, 741, 750, 765, 780) |
| Enums use `@app_commands.choices` with a `Choice` list, not free text | `help_type` (main.py:136-143), `blend_mode` (main.py:336-339), `field` for /mass_replace (main.py:460-465) |
| Optional params carry their default in the Python signature | `steps: Optional[int] = 50`, `num: Optional[int] = 100`, `force_update: Optional[bool] = False` |
| "Leave blank to view" pattern for getter/setter commands | `'The scheduler you want. Leaving blank tells you the current scheduler.'` (main.py:533; same shape for /set_ratio, /set_repo, /set_device, /set_model, /set_lora) |
| Big ints from Discord come in as `str` | `/ai`'s `seed: Optional[str]` with in-body `int()` conversion — Discord ints cap at ~15 digits (main.py:513-519). Copy this pattern, don't fight it |
| Sentence case, trailing period | pervasive; keep it |

Known inconsistency, do not copy: `/pull`, `/push`, `/update`, `/stop_bot` are
perms-gated in code but their main.py descriptions lack "(admin only)" (their
help.py entries have it). New admin commands put the marker in BOTH places.

## 5. House code style (derived from the code, not aspiration)

### 5.1 Module layout skeleton

Verified against `commands/psd_analyzer.py`, `commands/query_card.py`,
`commands/diffusion.py`, `commands/music_player.py`, `commands/help.py`,
`commands/update_bot.py`. Canonical order (psd_analyzer.py and query_card.py
are the cleanest exemplars):

1. **Imports** — `from x import y` block; then a blank line; then `import config`,
   `from global_config import ...`, `from commands.utils import ...`.
2. Logger/warnings silencing if needed (psd_analyzer.py:24, music_player.py:25).
3. **`# Configuration` banner + CONSTANTS** — UPPER_SNAKE constants; config.py
   fields read via `getattr(config, "field", default)` so a sparse config.py
   never crashes an import (query_card.py:24, diffusion.py:24-29, utils.py:21).
4. **Dataclasses / classes** (`@dataclass` heavily used; Enum where apt).
5. **Helper functions** (private-ish; `_underscore` names inside classes).
6. **Public API functions** — the sync functions main.py wraps as commands
   (banner named "Public API Functions", "Discord Command Functions", "Music
   Functions", etc. — name it for the content).
7. **`# Initialization` banner** — module-level singleton + `init_x()` function
   (`stats_db = StatsDatabase()` + `init_psd` at psd_analyzer.py:1641-1642;
   `card_repo = CardRepository()` + `init_query` at query_card.py:626-627).
8. **`# Module Exports` banner + `__all__`** — every command module ends with
   an explicit `__all__` list (even help.py, whose `__all__ = ['print_help']`),
   with one standing exception: utils.py, the shared toolbox, has none.

Sanctioned variants (don't "normalize" them): diffusion.py keeps its globals
in a `# Global State` section near the top (diffusion.py:119-134) with
`init_pipeline` under `# Pipeline Initialization` (190-193); music_player.py
puts `state = MusicState()` immediately after its dataclass (music_player.py:44)
and has no bottom init section. main.py uses a different banner style —
`####################` boxes per command category (e.g. main.py:128-130) — keep
that style in main.py only.

Section banner format everywhere else, exactly 24 `=`:

```python
# ========================
# Configuration
# ========================
```

### 5.2 Import style

`from x import y` named imports are the house rule: 113 from-import lines vs 9
plain `import` lines repo-wide (2026-07-11 census; the 9 are `import config`
×5, `import pandas as pd`, `import logging`, and setup.py's `shutil`/`sys`).
Even pickle is imported as `from pickle import load, dump` (query_card.py:5,
psd_analyzer.py:10). The deliberate exception: `import config` as a module, so
optional secret fields can be read with `getattr(config, "X", default)`. Follow
both halves of that pattern.

### 5.3 Docstrings

Triple-quoted. One-liners for simple functions
(`"""Send a message to the last active channel."""`, music_player.py:75).
Args/Returns sections for anything with non-obvious parameters — real shapes:
`populate_files` has a `Returns:` block (query_card.py:56-61); `update_bot`
has an `Args:` block (update_bot.py:122-129). No Sphinx/Google tooling
enforces this; match the neighbors.

### 5.4 The `x[len(x) - 1]` idiom (REQUIRED in cythonized modules)

Never write `x[-1]` (or any negative index) in the 8 modules setup.py
compiles: analytics, frankenstein, help, psd_analyzer, query_card, management,
music_player, diffusion (setup.py:21-30). They build with `wraparound=False`,
where negative indexing is undefined behavior, and a past `bboxes[-1]` was
mechanically purged for exactly this reason (commit dd800bc; the one that
escaped became the live removeprefix bug). Current census: 15
`x[len(x) - 1]` occurrences, **zero** negative indexes in commands/ or
main.py. Why the directive is set and the stale-.so trap it interacts with:
`autocroissant-build-and-env`. Why you must not "clean up" the idiom:
`autocroissant-change-control` non-negotiables.

### 5.5 f-strings

Standard for all interpolation (`f"{cmd}\n{desc}\n\n"` help.py:143;
`f"Synced commands to guild: {guild.name} ({guild.id})"` main.py:101). No
`%` or `.format()` in new code.

### 5.6 queue_* discipline in command functions

Worker functions are **sync** and never await Discord directly; they call
`queue_message` / `queue_file` / `queue_edit` / `queue_command`
(commands/utils.py:38-62) and main.py's 1-second loops do the actual sends.
`print_help` (help.py:126-148) is the minimal canonical example: sync def,
takes `Interaction`, ends in `queue_message`. The whole execution model and
why violating it deadlocks or drops messages: `autocroissant-architecture-contract`.

### 5.7 Type hints

Modern built-in generics and unions: `dict[str, str]`, `deque[str]`,
`VoiceClient | None` (music_player.py:41), `Optional[...]` in slash
signatures. Annotate public API signatures; module constants get annotations
when non-obvious (`GIT_TOKEN: str = getattr(...)`, query_card.py:24).

## 6. Templates

### 6.1 help.py entry

```python
    "/my_command <required_arg> [optional_arg]:" : "One-sentence summary of what it does (admin only).\n"
                                                   "- required_arg: What it means, units if any\n"
                                                   "- optional_arg: What it means (default: False)",
```

Drop "(admin only)" if not perms-gated. Pad the key so the `:` aligns with its
dict's column. Defaults in the bullets MUST match the main.py signature.

### 6.2 New module skeleton (commands/my_module.py)

```python
from discord import Interaction

import config
from commands.utils import queue_message

# ========================
# Configuration
# ========================
MY_CONSTANT = 42
MY_TOKEN: str = getattr(config, "MY_TOKEN", "")

# ========================
# Helper Functions
# ========================
def _helper() -> int:
    """One-line summary."""
    return MY_CONSTANT

# ========================
# Public API Functions
# ========================
def my_command(interaction: Interaction, arg: str) -> None:
    """Sync worker: never awaits Discord; ends in a queue_* call."""
    queue_message(interaction, f"Result: {_helper()} {arg}")

# ========================
# Initialization
# ========================
def init_my_module() -> None:
    """Called from on_ready via queue_command, not at import time."""

# ========================
# Module Exports
# ========================
__all__ = [
    'my_command',
    'init_my_module',
]
```

(Real modules put a blank line before each banner and use Args/Returns
docstrings on non-trivial functions — §5.3; dataclasses go between the
Configuration block and the helpers.)

Remember: a new module is not live until main.py imports it, wraps the
function (`my_command = to_thread(my_command)`), registers `@tree.command`,
and — if it should be compiled — setup.py INCLUDE_FILES gains a line (then the
len-1 ban applies). Gates for all of that: `autocroissant-change-control`.

### 6.3 SKILL.md skeleton

```markdown
---
name: autocroissant-<topic>
description: Load this skill when <trigger words, symptoms, command names —
  front-loaded>. Contains <one dense sentence of contents>.
---

# AutoCroissant <topic>

One-paragraph scope statement. Date-stamp: facts verified YYYY-MM-DD.

## <Body sections: tables and checklists over prose; a story/rationale
## attached to every rule; file:line for every load-bearing claim>

## When NOT to use this skill

| Task | Use instead |
|---|---|
| <adjacent task> | <sibling-skill-name> |

## Provenance and maintenance

- Fact X: `grep -n "..." file.py` (expect: ...)
- Refresh all date-stamps when you touch this file.
```

## 7. Maintaining this skill library

### 7.1 Frontmatter contract

`name` + `description`, nothing else. The description is the **retrieval
surface**: a cheap model decides whether to load your skill by reading only
that paragraph. Front-load trigger words; name concrete symptoms ("my edits
don't take effect"), command names (/update_stats), file names
(requirements2.txt), and error strings. Then one dense sentence of what the
skill contains. A vague description ("helpful information about X") means the
skill never loads and might as well not exist.

### 7.2 Provenance and maintenance sections

Every skill ends with one. Every volatile fact — line numbers, counts,
defaults, dates — gets a one-line re-verification command with the expected
value in a comment. When you touch a skill for ANY reason: re-run its
provenance lines, fix what drifted, and refresh its "as of" date-stamps in the
same edit. A skill whose provenance lines fail is worse than no skill — it
states falsehoods with confidence.

### 7.3 One home per fact — ownership map

Each fact lives in exactly ONE skill; everyone else points there with a 1-2
line summary at most. When editing a skill and tempted to explain a
neighboring fact, check this map and link instead.

| Skill | Owns |
|---|---|
| autocroissant-change-control | Change classes/gates, pickle-commit discipline, never-commit list, add-a-slash-command checklist, force_reset danger |
| autocroissant-debugging-playbook | Symptom → triage tables for live failures |
| autocroissant-failure-archaeology | Incident timeline, root causes with commit hashes, the removeprefix live-bug story, dead ends/reverts |
| autocroissant-architecture-contract | Queue trio + to_thread model, module graph, singletons, registration pattern, invariants, known-weak points |
| impossibility-cards-reference | Card domain: folder classification, PSD layer semantics, type injection, validation, pickle schemas, query DSL |
| autocroissant-config-and-flags | Every config axis: config.py field names, constants, runtime-settable vs restart-required |
| autocroissant-build-and-env | Env setup, requirements split, Cython build/clean, negative-indexing ban rationale, stale-.so trap |
| autocroissant-run-and-operate | Start/stop/deploy, command sync, self-update flow ops, multi-machine handoff, artifact locations |
| autocroissant-diagnostics-and-tooling | The 5 read-only scripts + golden outputs |
| autocroissant-validation-and-qa | Evidence standards, golden-card inventory, parser-change acceptance procedure, pickle-push gate |
| autocroissant-ai-boundary | Toggleable-AI doctrine, lazy imports, requirements2 enablement, VRAM modes |
| autocroissant-psd-extraction-campaign | The executable extraction-improvement campaign |
| autocroissant-analysis-toolkit | First-principles investigation recipes with worked examples |
| autocroissant-research-frontier | The two ambitions + first steps + milestones |
| autocroissant-research-methodology | Evidence bar, idea lifecycle, keep-it-fun clause |
| autocroissant-docs-and-style (THIS) | help.py/describe formats + sync checklist + drift audit; house code style; skill-library meta-rules; templates |

### 7.4 Add a new skill vs extend an existing one

Extend when the fact has an owner in the map (a new incident → archaeology; a
new flag → config-and-flags; a new command → its checklist already exists).
Add a new skill only when a genuinely new AXIS appears — a capability or
workflow no current owner covers — and then: claim ownership by adding a row
to the map above (this file), write the frontmatter to the 7.1 contract,
include "When NOT to use" naming the nearest siblings, and end with
provenance. Prefer 16 sharp skills over 30 overlapping ones; overlap is how
two skills drift into contradicting each other.

### 7.5 Skills never route around change-control

No skill may instruct a reader to commit, push, edit pickles, or ship a code
change by a path that skips `autocroissant-change-control`'s gates, and no
skill may contradict `autocroissant-architecture-contract`'s invariants. If
your new content collides with either, the fix is a conversation with those
skills' text (and the owner), not a competing instruction. A skill library
that disagrees with itself trains readers to ignore all of it.

## When NOT to use this skill

| Task | Use instead |
|---|---|
| Actually shipping the doc fix / any commit, push, command add/rename, pickle touch | autocroissant-change-control (gates and procedure) |
| Understanding the queue/threading rules the style encodes, or where new code goes architecturally | autocroissant-architecture-contract |
| A live symptom (bot won't start, command missing, message not sending) | autocroissant-debugging-playbook |
| What a flag/constant does or its current default | autocroissant-config-and-flags |
| Cython build mechanics, why wraparound=False exists | autocroissant-build-and-env |
| Card-domain semantics referenced by command descriptions | impossibility-cards-reference |

## Provenance and maintenance

All claims verified 2026-07-11 against commit 284d13c. Re-run before trusting;
line numbers drift. Expected values in comments are the 2026-07-11 readings.

```bash
# Six dicts + router exist; help.py length
grep -nE '^(general|text|card|stats|ai|music)_commands = \{|^def print_help' commands/help.py   # lines 6,22,31,61,80,102,126
wc -l commands/help.py                                                                          # 156

# print_help routes "stats"; else-fallback is general
grep -n -A1 'help_wanted == "stats"' commands/help.py        # 136-137 → stats_commands

# Slash choices for /help include stats (Choice noise is exactly why extraction is anchored)
grep -n 'Choice(name="stats"' main.py                        # 141

# Registered-vs-documented drift audit (expect 54 vs 50; 4 missing as listed in §3.1)
grep -oE '@tree\.command\(name="[^"]+"' main.py | wc -l      # 54
grep -oE '^    "/[a-z_]+' commands/help.py | sort -u | wc -l # 50
# full diff procedure: §3.1 comm commands

# D1 /pull drift: docs claim hard reset; git_pull has none
grep -n 'hard reset' commands/help.py main.py                # help.py:10, main.py:172
grep -n 'def git_pull\|def git_reset_hard' commands/update_bot.py   # 33, 104
sed -n '33,101p' commands/update_bot.py | grep -c reset      # 0 — git_pull's body never resets

# D2 /export_cards default mismatch
grep -n 'only_ability' commands/help.py main.py              # help.py:68 "(default: True)" vs main.py:375 "= False"

# D3 /help entry omits stats
grep -n '"/help <' commands/help.py                          # 7: no "stats" in the category list

# Module skeleton: banners in the four exemplar modules
grep -n -A1 '^# ========================$' commands/psd_analyzer.py commands/query_card.py commands/diffusion.py commands/music_player.py | grep -v '^\-\-$'

# Singletons + init functions
grep -n 'stats_db = StatsDatabase()\|card_repo = CardRepository()\|state = MusicState()' commands/*.py   # psd_analyzer:1641, query_card:626, music_player:44

# __all__ ends every command module except utils.py
grep -L '__all__' commands/*.py                              # only commands/utils.py

# Import-style census (from-imports should dwarf plain imports)
grep -hE '^(from|import) ' main.py global_config.py setup.py commands/*.py | grep -cE '^from '    # 113
grep -hE '^(from|import) ' main.py global_config.py setup.py commands/*.py | grep -cE '^import '  # 9

# len-1 idiom census; negative-index ban holds
grep -rnE '\[len\([a-z_]+\) - 1\]' commands/ main.py | wc -l # 15
grep -rnE '\[-1\]|\[-2\]' commands/*.py main.py | wc -l      # 0

# Cythonized module list (len-1 ban scope)
grep -n -A9 'INCLUDE_FILES = \[' setup.py                    # 8 modules incl. help.py

# queue_* wrappers
grep -n 'def queue_' commands/utils.py                       # defs at 38,43,49,55,60: queue_any/message/file/edit/command

# Skill inventory matches the §7.3 ownership map (16 dirs when all agents land)
ls .claude/skills/
```

Maintenance rules for this file: when a §3 drift instance is fixed, delete its
row and re-run the audit; when a command is added, re-run the §3.1 diff and
update the counts (54/50) and dict-entry table; when a 17th skill appears, add
its ownership row in §7.3. Refresh every "2026-07-11" stamp whenever you touch
this file.
