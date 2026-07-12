---
name: autocroissant-analysis-toolkit
description: First-principles analysis recipes for AutoCroissant — load this when you must PROVE something rather than pattern-match it. Triggers: "root cause" a mis-parsed card or wrong [type] injection; "prove" / "which code path produced this data" / forensic attribution of pickle contents; "bisect" / "when did X change" / "which commit introduced" (git log -S, git show time-travel); "measure before running" an API budget for /update_stats vs GitHub rate limits; "why is the bot slow to reply" queue-timing reasoning; "pickle forensics" / comparing stats.pkl or old_stats.pkl snapshots across git history without running the bot; any "analysis method" question where the debugging-playbook triage tables don't cover the symptom. Contains six worked recipes with real commands and real outputs from this repo's history: the gap_trace 6-step pipeline on The Lich King, the path-style experiment that root-caused the removeprefix bug, the "am forehead" diff-reading lesson, the 815-request /update_stats budget arithmetic, the ≤2s queue-latency derivation, and an actually-run cca0aaf-vs-eb9aa84 old_stats comparison.
---

# AutoCroissant analysis toolkit

Date-stamped 2026-07-11. All commands assume cwd = repo root `/Users/michaelsrouji/Desktop/AutoCroissant`.

This skill is the project's method book: how to turn "this looks wrong" into a mechanism you can
defend. The house evidence bar (owned by autocroissant-research-methodology): **one mechanism must
explain ALL observations, including the negative ones, and your hypothesis should predict numbers
BEFORE you run the measurement.** Every recipe below follows that bar and ends with a worked example
that was actually executed against this repo on 2026-07-11 — the outputs shown are real, not
illustrative.

Why this matters here specifically: this repo has NO tests and NO CI, the pickles ARE the database,
and history contains commits whose messages are jokes ("am forehead") and commits whose messages
look routine but ballooned the database ("PICKLE"). Pattern-matching on names, messages, or vibes
has burned this project repeatedly; the recipes below are the antidote.

## Ground rules for every recipe

- The repo is treated READ-ONLY for analysis: read-only git only (`log`/`show`/`diff` — never
  add/commit/checkout/reset), never run the bot (`python3 main.py`), never call `update_stats()`,
  never write any pickle in the repo. Snapshots you extract go to `/tmp` or a scratch dir.
- `config.py` (repo root) is SECRET (tokens). Import chains read it; never print its values.
- Anything that imports bot modules (the diagnostics scripts, the pickle one-liners) prints a
  token-presence line on import. Append `2>&1 | grep -v "Git token"` to keep output clean.
- The diagnostics scripts live in
  `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/` and are read-only and safe.
  **Pass card PSDs by full path** (e.g. `~/Desktop/TTSCardMaker/...`): the scripts resolve relative
  arguments against the REPO root, not the card clone — a bare `Creatures/...` path fails with
  FileNotFoundError (verified 2026-07-11).

Jargon used below, once: a **PSD** is a layered Photoshop file, one per card, in the separate
`MichaelJSr/TTSCardMaker` repo cloned at `~/Desktop/TTSCardMaker`. A **gap** is a run of 3+ spaces
in a card's ability text where a type icon visually sits. The **midline** is `height *
TYPE_REGION_RATIO` (0.5); icons above it are the card's own types, icons below it are candidates
for inline injection into the ability text. A **sweep** is one `/update_stats` traversal of all
cards, in **local mode** (walk the clone) or **remote mode** (GitHub API).

## When NOT to use this skill

- **Routine symptom triage** (bot won't start, commands missing, messages not sending, music
  silent, OOM): use **autocroissant-debugging-playbook** first. This skill is for when its tables
  don't cover the symptom and you must build the explanation yourself.
- **Running the extraction-improvement campaign** (baselines, gates, ranked solution menu): use
  **autocroissant-psd-extraction-campaign**. It sequences these recipes; this file teaches them.
- **Accepting a parser change / pushing pickles** (golden cards, sandbox sweep, problem-count
  gate): use **autocroissant-validation-and-qa**.
- **Script flag reference and full golden-output catalog**: **autocroissant-diagnostics-and-tooling**.
- **The incident stories themselves** (what happened, when, status): **autocroissant-failure-archaeology**
  owns the timeline; this skill only borrows incidents as worked examples of method.

---

## Recipe 1 — Root-cause a mis-parsed card

The domain's core method. Use it whenever a card's stored ability text, types, stats, or stars are
wrong, or `[type]` markers land in the wrong place (the owner's stated hardest live problem).

**When to reach for it:** a specific card is wrong in `stats.pkl` or in `/query` output, and you
need to know WHICH stage of the parse diverged — not just that it did.

**Steps**

1. **Dump the layers** the way the parser sees them:
   `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py ~/Desktop/TTSCardMaker/"<path>.psd"`
   (add `--text` to see raw text-layer contents). Identify the candidate layers: text layers
   (kind `type` in psd-tools means TEXT), type-icon layers (name ∈ known types), stat digit layers,
   star layers. Full layer semantics: **impossibility-cards-reference**.
2. **Trace the injection pipeline** with
   `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py ~/Desktop/TTSCardMaker/"<path>.psd"`.
   It prints six numbered stages — [1] types collected above the midline, [2] raw ability text
   layers with bboxes, [3] joined text with `<GAP:n>` markers (gap regex `\s{3,}`,
   psd_analyzer.py:348), [4] below-midline icons, sort order, and the prune decision
   (`max(last_y//3, card_mid_y)`), [5] gap count vs kept types, [6] final injected text.
3. **Localize the divergent stage**: compare each stage against what the card art shows. The first
   stage whose numbers disagree with the card is where the mechanism lives.
4. **Form a mechanism hypothesis that explains EVERY symptom** — including why other cards are
   fine. "The icon is 30px above the midline" explains a missing injection AND the icon showing up
   as a creature type; "the parser is flaky" explains nothing.
5. **Confirm with the real parser**:
   `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py ~/Desktop/TTSCardMaker/"<path>.psd"`
   — it runs the actual `PSDParser` + `CardValidator` end to end, no Discord, nothing saved. The
   output should match your predicted values field for field.

**Worked example — The Lich King as a healthy trace** (all outputs real, 2026-07-11):

`dump_psd_layers.py` on `~/Desktop/TTSCardMaker/"Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd"`
shows, among ~80 layers: size `1200 x 2400` (midline 1200); a `Types` group top-left with `Ice`
(bbox y 158–308) and `Undead` (y 21–135) — both above the midline; a text layer `Ability` (kind
`type`) at bbox (23, 1773, 1176, 2285); `Darks` digit groups where the visible digits are Hp `7`,
Def `6`, ATK `7`, SPD `5`; and a lone `Undead` smartobject at (734, 2239, 808, 2299) — below the
midline.

`gap_trace.py` then prints exactly:

```
PSD 1200x2400; TYPE_REGION_RATIO=0.5 -> card_mid_y=1200
[1] creature-types collected ABOVE midline: ['ice', 'undead']
[2] ability text layers found: 1
    at (x=23, y=1773): "'(OFS) Equip Frostmourne.\n...Minions inherit       .\n'"
[3] ... Minions inherit<GAP:7>.
[4] type icons BELOW midline (candidate inline types): 1
    undead          at (x=734, y=2239)
    prune threshold = max(last_y//3=746, card_mid_y=1200)
    kept after prune: ['undead']   dropped: []
[5] gap count = 1 vs kept types = 1   (match)
[6] FINAL ability text ... Minions inherit [undead].
```

`parse_one.py` confirms: `type creature, stars 5, series world of warcraft, types ['ice','undead'],
hp 7 def 6 atk 7 spd 5`, ability ending `Minions inherit [undead].`, `validator problems: NONE` —
every number predicted by the trace. (These golden values are a snapshot of 2026-07-11 — canonical
current expected values live in autocroissant-validation-and-qa §3; if the trace stops matching,
check there before assuming a regression.) (It also prints `known types (36)` including a leaked
`.ds_store` entry — a real artifact of populating types from the local `Types/` folder on macOS.)

**Failure signatures** — how each classic failure looks in the same trace:

| Signature | What you see | Mechanism it points to |
|---|---|---|
| No text layer | `[2] ability text layers found: 0`; parse_one shows `NO ABILITY LAYER` (appended at psd_analyzer.py:422) and/or `ABILITY TEXT NOT FOUND` (validator, :692) | Text layer missing, or not named `ability` on a non-rulebook card — check the dump for a `type`-kind layer with an unexpected name |
| Icon above midline | The icon appears in `[1]`, not `[4]`; the gap in `[3]` stays unfilled | Icon y < 1200 → it's classified as a creature type, never an injection candidate. TYPE_REGION_RATIO=0.5 is deliberate — do NOT "fix" by injecting top-half icons (fenced off in autocroissant-psd-extraction-campaign) |
| Gap/type mismatch, extra types | `[5] gap count = N vs kept types = M   <-- MISMATCH, leftover types get appended to last line!` | `_inject_type_names` appends leftovers as ` [x] [y]` onto the LAST line (psd_analyzer.py:631-635) — the classic "types stuck at the end of the ability" symptom |
| Gap/type mismatch, extra gaps | `[5]` MISMATCH the other way; final text keeps raw multi-space runs — and a gap directly before punctuation silently VANISHES (cleanup `\s+([:;,\.\?!])` → `\1`, psd_analyzer.py:538) | Fewer icons survived than gaps exist — check `[4]`'s `dropped:` list (prune threshold ate them) before blaming the text |

**Conclusive result:** you can name the stage ([1]–[6]), state the mechanism in one sentence, it
predicts the exact stored (wrong) text AND the healthy cards' correctness, and parse_one reproduces
both. WHITESPACE IS LOAD-BEARING: never "fix" a card by collapsing spaces — that path was tried and
reverted (3bbaa2b; story in autocroissant-failure-archaeology).

---

## Recipe 2 — Prove which code path produced data (forensic attribution)

**When to reach for it:** stored data (usually a pickle) could have been produced by more than one
code path — local vs remote traversal, old vs new schema, bot vs manual script — and the answer
changes what you do next (e.g. "is the live bug already in our data?").

**Steps**

1. **Enumerate the candidate code paths.** Read the actual dispatch point (here:
   `update_stats`, psd_analyzer.py:1173, branches on `use_local_repo`).
2. **For each path, derive an observable fingerprint it MUST leave** in the artifact. Prefer
   fingerprints that differ per path and cannot be produced accidentally by the other path.
3. **Check the artifact** with standalone reads (Recipe 6 mechanics). Every fingerprint must agree;
   one disagreement kills the attribution.

**Worked example — which mode built today's stats.pkl?** This is the experiment that established
the removeprefix bug (introduced by f7c915c; full incident owned by
autocroissant-failure-archaeology) had NOT corrupted production data.

Candidate paths and their fingerprints:

| Path | Path-string fingerprint | Timestamp fingerprint |
|---|---|---|
| Local mode (`_process_local_files`, psd_analyzer.py:1002) | `relative_path = full_path.removeprefix("TTSCardMaker").strip('/')` on an ABSOLUTE walk path is a no-op → stored paths start `Users/michaelsrouji/...`, and classification sees top folder "Users" → mass UNKNOWN type | `getmtime` → local clone mtimes (whenever the clone was last pulled) |
| Remote mode (`_process_files_from_response`) | Tree-API paths → repo-relative, starting `Creatures/`, `Auxiliary/`, ... | `_get_remote_timestamp` → GitHub COMMIT times |

The exact one-liners, and their real output (2026-07-11):

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from pickle import load
from collections import Counter
db = load(open('stats.pkl','rb'))
print('entries:', len(db))
print('top-level path components:', Counter(c.path.split('/')[0] for c in db.values()).most_common(8))
print('paths starting with Users/:', sum(1 for c in db.values() if c.path.startswith('Users')))
" 2>&1 | grep -v "Git token"
# entries: 813
# top-level path components: [('Creatures', 297), ('Auxiliary', 258), ('Items', 182), ('Field', 50), ('N.M.E', 26)]
# paths starting with Users/: 0

python3 -c "
import sys; sys.path.insert(0, '.')
from pickle import load
from datetime import datetime, timezone
db = load(open('stats.pkl','rb'))
for c in sorted(db.values(), key=lambda c: c.timestamp)[-3:]:
    print(c.name, '|', c.path, '|', datetime.fromtimestamp(c.timestamp, tz=timezone.utc).isoformat())
" 2>&1 | grep -v "Git token"
# Z!!!!! | Creatures/Other/5 Stars/Z!!!!!.psd | 2026-06-18T06:02:49+00:00
# Anubisath Guardian | Auxiliary/Minions/Anubisath_Guardian.psd | 2026-06-18T09:47:02+00:00
# Qiraji Soldier | Auxiliary/Minions/Qiraji_Soldier.psd | 2026-06-18T09:47:02+00:00

git -C ~/Desktop/TTSCardMaker log -1 --format='%cI  %s' -- "Auxiliary/Minions/Anubisath_Guardian.psd"
# 2026-06-18T02:47:02-07:00  bot errands        <- EXACTLY the stored timestamp
stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S%z' ~/Desktop/TTSCardMaker/Auxiliary/Minions/Anubisath_Guardian.psd
# 2026-07-11T13:49:17-0700                      <- what LOCAL mode would have stored instead
```

Two independent fingerprints, both pointing the same way: all 813 paths repo-relative (zero
`Users/`-style), and the newest stored timestamps equal TTSCardMaker COMMIT times to the second
while the local mtime is a different day entirely. **Remote mode produced the current data; the
local-mode bug has not touched it.** (`inspect_pickle.py` packages the path check as its
`entries with suspicious (non repo-relative) paths: 0` line.)

A third corroborating measurement (2026-07-11): the 813 stored timestamps collapse to only
**103 distinct values** — 769 cards share theirs with at least one other card, and the largest
cluster is **64 cards at exactly 2026-02-08 04:11:26 UTC** (a TTSCardMaker bulk-commit time).
Second-exact clustering is what per-commit remote timestamps produce and per-file local mtimes
cannot.

```bash
python3 -c "
import sys, pickle
from collections import Counter
sys.path.insert(0, '.')
s = pickle.load(open('stats.pkl','rb'))
c = Counter(x.timestamp for x in s.values())
print(len(s), len(c), c.most_common(1), sum(1 for x in s.values() if c[x.timestamp] > 1))
" 2>&1 | grep -v "Git token"
# 813 103 [(1770523886.0, 64)] 769     <- epoch 1770523886 = 2026-02-08T04:11:26Z
```

**Conclusive result:** every fingerprint you predicted for one path is present, every fingerprint
unique to the rival paths is absent, and you checked at least two INDEPENDENT observables. One
matching observable is a hint; two independent ones that can't co-occur accidentally is proof.

---

## Recipe 3 — Bisect behavior with git (when did X change, and what actually changed)

**When to reach for it:** you need the commit that introduced/removed a behavior, or you suspect a
commit message is lying to you (here they sometimes joke).

**Steps**

1. **String-track with `git log -S`** — finds commits where the COUNT of occurrences of a string
   changed (i.e., additions/removals, not context moves):

   ```bash
   git log -S removeprefix --oneline -- commands/psd_analyzer.py
   # f7c915c cythonized the code                <- introduced the live local-path bug
   git log -S 'TYPE / WHITESPACE MISMATCH' --oneline
   # 081b1fd fix bug with types being improperly placed in abilities   <- removed it
   # fed8a83 output a problem if num types does not match num gaps in ability   <- added it
   ```

   Both real (verified 2026-07-11). The second one instantly answers "we used to WARN on gap/type
   mismatch — where did that go?": added fed8a83, removed 081b1fd (why it thrashed:
   autocroissant-psd-extraction-campaign).

2. **Time-travel read with `git show <hash>:<file>`** to see the exact before/after line, not the
   diff hunk's guess:

   ```bash
   git show f7c915c~1:commands/psd_analyzer.py | grep -n 'TTSCardMaker'
   # 960:  relative_path = full_path.split("TTSCardMaker")[-1].strip('/')   <- correct
   git show f7c915c:commands/psd_analyzer.py | grep -n 'removeprefix'
   # 961:  relative_path = full_path.removeprefix("TTSCardMaker").strip('/')  <- no-op on absolute paths
   ```

3. **Read the diff with discipline** — semantics from code, never from names or messages:
   - Commit messages lie or joke. Verify with `git show <hash>` before citing.
   - Variable names lie too. Derive a counter's meaning from where it increments, not what it's called.

**Worked example — the canonical "am forehead" lesson.** The "save pickles only when something
changed" gate looks, from messages, like it shipped in fb47b5d ("fix bug with unknown type, and
dont overwrite stats with empty stats", 2026-01-15 17:23). The diff says otherwise:

- `git show fb47b5d` — added `if (num_updated > 0): stats_db.prune_clean_cards(); stats_db.save()`.
  But in that version `num_updated += 1` sits at LOOP level, once per PSD processed regardless of
  change (verify: `git show fb47b5d:commands/psd_analyzer.py | sed -n '855,940p'`). Any non-empty
  sweep makes it > 0 → the gate never skipped a save → **ineffective as written**. (The message is
  half-true: its "fix bug with unknown type" part is real — it reverted f7c915c's other regression,
  `folders = dirname(relative_path)`, which made `top_folder = folders[0]` the first CHARACTER of a
  string → every card UNKNOWN. Same commit, one working fix and one non-working gate: diffs
  arbitrate claim by claim, not commit by commit.)
- `git show 4e03190` — message: **"am forehead"** (same day, 17:43, twenty minutes later). Looks
  like a junk commit. The diff is the actual fix: every `return ..., num_updated` becomes
  `num_new`, and the gate becomes `if (num_new > 0):` — `num_new` only increments for new or
  content-changed cards, so the gate finally does its job. (Still true at HEAD:
  psd_analyzer.py:1222.)
- `git show 2ff52b2` ("Update psd_analyzer.py", 18:55) — despite the plausible message, it's ONLY
  type annotations and parenthesization. Attributing the gate to it would be wrong.
- Bonus name-lie inside the same loop: at HEAD, `num_new` ALSO counts content-updated existing
  cards (the `else: num_new += 1` at psd_analyzer.py:948/:1030) — it is "new + changed", not "new".
  Only the increment sites tell you that.

**Conclusive result:** you can name the introducing (and, if applicable, removing) commit, quote
the exact changed line from `git show <hash>:<file>`, and your story survives reading the
neighboring commits from the same day. If two commits within minutes touch the same lines, the
LAST one is usually the real fix and the first one the attempt — check both.

---

## Recipe 4 — Measure an API budget before running (remote /update_stats)

**When to reach for it:** BEFORE any run that talks to the GitHub API in a loop — remote-mode
`update_stats`, or local mode with `use_local_timestamp:False`. Predict the request count first;
if you can't, you're not ready to run it. (Do not run update_stats during analysis at all — this
recipe exists so the OPERATOR knows the cost; operations guidance lives in
autocroissant-run-and-operate.)

**Steps**

1. **Count the inputs:**

   ```bash
   find ~/Desktop/TTSCardMaker -name "*.psd" | wc -l                      # 904
   find ~/Desktop/TTSCardMaker -name '*.psd' | grep -cE '/(Markers|MDW)/' # 91  (EXCLUDE_FOLDERS, skipped)
   find ~/Desktop/TTSCardMaker -name '*.psd' | grep -vcE '/(Markers|MDW)/'# 813 (processed — equals stats.pkl entry count)
   ```

2. **Read the loop and tag every network call** (`commands/psd_analyzer.py`, verified 2026-07-11):
   - `traverse_remote` — one `get(...git/trees/main?recursive=1)` (line 820) + one
     `Github(...).get_repo(...)` (line 826): **2 requests fixed overhead**.
   - `_process_files_from_response` — for EVERY non-excluded PSD, `_get_remote_timestamp` (line
     929) calls `repo.get_commits(path=path)` (line 1066) **BEFORE** the should-update decision
     (line 932). The update check cannot save you these requests: **813 requests minimum, every
     sweep, even if nothing changed**. (New cards can cost an extra page: the original-author
     lookup indexes the oldest commit, line 1072. Today 0 stored cards lack an author, so routine
     sweeps don't pay this.)
   - Per CHANGED card only: one `urlretrieve` of the PSD from raw.githubusercontent.com (line
     952) — a raw download, not a REST API call, so it doesn't consume the API quota (it does
     consume time and bandwidth).

3. **Do the arithmetic against the limits.** The code prints its own reminder
   (psd_analyzer.py:859): 60 req/hr anonymous, 5000 req/hr with GIT_TOKEN.

   | Mode | Requests per no-change sweep | Verdict |
   |---|---|---|
   | Anonymous (no GIT_TOKEN) | 2 + 813 ≈ 815 | **Infeasible** — rate-limited after ~58 PSDs, ~7% of one sweep |
   | With GIT_TOKEN | ≈ 815 of 5000 | 815/5000 ≈ 16%, about a SIXTH of the hourly budget per no-change sweep — six back-to-back sweeps (≈4,890) would squeak under the cap with almost no headroom for the rest of the bot, so treat 2–3 sweeps/hour as the practical ceiling |

**Conclusive result:** a predicted request count derived from the loop, checked against the input
count, BEFORE anything runs — and the prediction is falsifiable (if a sweep rate-limits earlier
than predicted, your reading of the loop is wrong; re-read it). The general form: fixed calls +
(per-item calls × items surviving filters) + (per-changed-item calls × expected changes).

---

## Recipe 5 — Reason about queue timing

**When to reach for it:** "the bot took N seconds to answer / the reminder fired late / is the
queue stuck?" — decide from first principles whether the delay can even BE queue delay before
digging anywhere else.

**The mechanism** (verify at main.py:823-871; queue helpers utils.py:38-62): three module-level
deques (`command_queue`, `dispatch_queue`, `edit_queue`) each drained by a `@tasks.loop(seconds=1)`
coroutine. Each tick drains the WHOLE deque (`while q: popleft()`), so backlog does not add
one-tick-per-item delay. Worker code never awaits Discord; it appends via
`queue_message`/`queue_file`/`queue_edit`/`queue_command`.

**Derived latency budget:**

| Path | Hops | Queue-added latency |
|---|---|---|
| Slash command reply | wrapper awaits the `to_thread` worker immediately (no command-queue hop) → worker enqueues send → dispatch tick | ≤ 1s (avg ~0.5s) |
| Queued command (reminder-fired via analytics.py:213, on_ready inits main.py:116-119, music/diffusion chaining) | enqueue → ≤1s command tick → `create_task` + thread runs → worker enqueues send → ≤1s dispatch tick | ≤ 2s (avg ~1s) |

Two structural caveats, both visible in the code: (a) `process_command_queue` re-enqueues
`check_reminder` at the top of every tick (main.py:825) and pops it in the same tick — reminders
are checked every second, then their commands take the queued-command path above; (b) the dispatch
loop awaits each send SEQUENTIALLY inside the tick (main.py:834-854), so one slow Discord HTTP call
delays the sends behind it in that tick — that is Discord I/O time, not queue design time.

**The diagnostic implication (the point of the recipe):** the queues can contribute at most ~2
seconds. **Any "hang" over ~3 seconds is thread work (the worker function itself) or Discord I/O
(send/edit HTTP, or the fallback chain after a 15-minute interaction-token expiry — see
autocroissant-debugging-playbook), never queue scheduling.** Don't investigate the deques for a
30-second stall; time the worker.

**Conclusive result:** you state the expected latency for the exact path the command took (≤1s or
≤2s of queue overhead), compare with the observed delay, and the difference tells you which
component to profile next. This recipe never requires running the bot — it's arithmetic over
main.py:823-871.

---

## Recipe 6 — Pickle forensics without the bot

**When to reach for it:** any question about what's IN a pickle, how it changed across history, or
whether a snapshot is damaged — answered with standalone reads, never by running the bot.

**Mechanics.** `stats.pkl` values are `CardInfo` dataclass instances defined in
`commands/psd_analyzer.py`, so unpickling needs (a) the repo root on `sys.path` and (b) `config.py`
present (the import chain reads it and prints the token-presence noise line — hence the grep).
Load pattern used throughout:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from pickle import load
db = load(open('stats.pkl','rb'))
print(len(db))" 2>&1 | grep -v "Git token"
```

Aggregate with one-liners (Counter over `c.path.split('/')[0]`, max over `c.timestamp`, filter on
`c.problems` — e.g. today the ONLY *stored/pickled* problem card is `('20 Creature Types',
['MISSPELT TYPE: tornado'])`; full validation computes 5 — surface distinction in
autocroissant-validation-and-qa §2). For the packaged summary use `inspect_pickle.py`; for two-snapshot diffs use
`diff_stats.py` (both documented with golden outputs in autocroissant-diagnostics-and-tooling).

**Compare snapshots across git history** — pickles are committed, so `git show <hash>:<file>` is a
free time machine:

```bash
git show HEAD:stats.pkl > /tmp/stats_head.pkl
python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl
# old: 813 cards   new: 813 cards
# added: 0   removed: 0   modified: 0     (exit 0 — verified 2026-07-11)
```

**The motivating case — old_stats.pkl's size history** (bytes measured 2026-07-11 with
`git cat-file -s $(git rev-parse <hash>:old_stats.pkl)`; the four anchor commits only — the full
annotated trace, from the 66-byte first archive through today's 82,881, is owned by
**autocroissant-failure-archaeology** Entry 3):

| Commit | Date | Bytes | Note |
|---|---|---|---|
| e380cee | 2025-11-03 | 3,196 | pre-incident baseline |
| cca0aaf | 2025-11-10 | 12,041 | balloon #1 ("PICKLE") |
| eb9aa84 | 2025-11-10 | 3,196 | Revert "PICKLE" — restored exactly |
| 637698b | 2025-11-25 | 12,039 | balloon #2 ("PICKLE") — **never reverted** |

(Root-cause family — duplicate archiving of unchanged cards — and incident status:
autocroissant-failure-archaeology. The archival guard `should_update and not is_new` landed
e7befd5/fb47b5d; the save gate story is Recipe 3.)

**Worked example, actually run 2026-07-11 — quantify balloon #1.** Extract and load the ballooned
snapshot and its revert:

```bash
git show cca0aaf:old_stats.pkl > /tmp/old_cca0aaf.pkl
git show eb9aa84:old_stats.pkl > /tmp/old_eb9aa84.pkl   # 12041 and 3196 bytes on disk
python3 -c "
import sys; sys.path.insert(0, '.')
from pickle import load
a = load(open('/tmp/old_cca0aaf.pkl','rb')); b = load(open('/tmp/old_eb9aa84.pkl','rb'))
print(type(a).__name__, len(a), sum(len(v) for v in a.values()), max(len(v) for v in a.values()))
print(type(b).__name__, len(b), sum(len(v) for v in b.values()), max(len(v) for v in b.values()))
print('subset:', set(b) <= set(a)); print(sorted(set(a)-set(b))[:6])
" 2>&1 | grep -v "Git token"
```

What actually happened (report observations, not expectations):

- **Both snapshots LOADED CLEANLY on today's class definitions.** These are post-"Big massive
  refactor" (366c8d9, 2025-10-20) pickles, and dataclass field access (`name`, `path`,
  `card_type`, `timestamp`, `author`, `stars`, `types`) works on every entry tested. The schema
  coupling did NOT bite for Nov-2025 snapshots. It remains real for PRE-refactor pickles (plain
  dicts, different files — metadata.pkl-era) and for any future field rename; the requirement
  (repo on sys.path + config.py) always stands.
- The numbers overturned the naive mental model: cca0aaf = **39 names / 39 archived versions**,
  eb9aa84 = **13 / 13**, and **no name had >1 version in either**. The balloon was NOT many
  versions of the same card — it was **breadth**: 26 extra names archived in one sweep, eb9aa84's
  13 a strict subset of cca0aaf's 39. The extras (`10 Attacking2` — a rulebook page, `Asteroid
  Field`, `Bench`, `Cloud Soiree`, `Crystal Peak`, ...) are cards that had NOT been edited —
  i.e. unchanged cards were being archived as if updated, once per sweep. Growth compounds
  ACROSS sweeps, so within-snapshot duplicate counts stay at 1 while the file balloons.
- Lesson: "duplicate archiving" as a phrase suggests depth; the artifact proves breadth-per-sweep.
  Quantifying (names, versions, subset relation, WHICH names) is what turns a byte count into a
  mechanism — and it took four one-liners and zero bot runs.

**Conclusive result:** counts (entries, versions, per-name max), a subset/diff relation between
snapshots, and named example entries — enough that someone else could re-derive your mechanism
from the same two `git show` commands. "The file got bigger" is not a finding; "26 specific
unchanged cards were archived in one sweep" is.

---

## Provenance and maintenance

Everything above was verified on 2026-07-11 against working tree + git history (repo at 192
commits, branch `main`). Volatile facts and their one-line re-checks:

| Fact | Re-verify with |
|---|---|
| Scripts exist / arg handling (full paths needed) | `ls .claude/skills/autocroissant-diagnostics-and-tooling/scripts/` ; `grep -n 'expanduser().resolve()' .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py` |
| Lich King golden trace ([1]-[6] numbers) | rerun the `gap_trace.py` command in Recipe 1 |
| Gap regex `\s{3,}` :348; punctuation cleanup :538; leftover-append :631-635; NO ABILITY LAYER :422 | `grep -n '_gap_pattern = \|\[:;,\|Append remaining types\|NO ABILITY LAYER' commands/psd_analyzer.py` |
| removeprefix no-op line (LIVE BUG, still open) | `grep -n 'removeprefix' commands/psd_analyzer.py` (:1002 today; story in autocroissant-failure-archaeology) |
| stats.pkl = 813 entries, 0 `Users/` paths, newest 2026-06-18 | first two one-liners in Recipe 2, or `inspect_pickle.py` |
| Timestamp clustering (103 distinct; 769 shared; max cluster 64 @ 2026-02-08 04:11:26Z) | the Counter one-liner in Recipe 2 (drifts with every sweep — recompute, don't reuse) |
| Newest-card commit-time match | the `git -C ~/Desktop/TTSCardMaker log -1` + `stat -f '%Sm'` pair in Recipe 2 (mtime changes on every pull — only the COMMIT time is stable) |
| `-S` results (f7c915c; fed8a83/081b1fd) and commit dates | rerun the two `git log -S` commands in Recipe 3; `git log -1 --format='%h %ad %s' --date=short <hash>` |
| fb47b5d gate ineffective / 4e03190 real fix / 2ff52b2 cosmetic | `git show fb47b5d`, `git show 4e03190`, `git show 2ff52b2` (read increments, not names) |
| `num_new` gate at HEAD | `grep -n 'if (num_new > 0)' commands/psd_analyzer.py` (:1222 today) |
| PSD counts 904 / 91 excluded / 813 processed | the three `find` commands in Recipe 4 (drift EXPECTED as cards are added — recompute, don't reuse) |
| Per-PSD get_commits before update decision; rate-limit warning | `grep -n 'get_remote_timestamp\|get_commits\|5000 if GIT_TOKEN' commands/psd_analyzer.py` (:929, :1066, :859 today) |
| Queue loops drain whole deque each 1s tick | `grep -n 'tasks.loop\|while command_queue\|while dispatch_queue\|while edit_queue' main.py` (:823-871 today) |
| Slash default `use_local_repo=True` (why remote-mode ops rule exists) | `grep -n 'use_local_repo: Optional' main.py` (:361 today) |
| old_stats byte trace incl. 637698b=12,039 never reverted | `git log --format='%h %ad %s' --date=short main -- old_stats.pkl` + `git cat-file -s $(git rev-parse <hash>:old_stats.pkl)` |
| cca0aaf/eb9aa84 load + 39/13 counts | rerun the Recipe 6 extraction + one-liner |
| Problem card list (today: 20 Creature Types / tornado) | the `c.problems` one-liner in Recipe 6 |

Line numbers drift with every edit to `commands/psd_analyzer.py` and `main.py` — trust the greps
above over the numbers in this file. If `PSDParser._extract_from_layers` changes shape, gap_trace's
mirror must be updated (see the MAINTENANCE NOTE in gap_trace.py itself). When a recipe's worked
example stops reproducing, that is itself a finding: apply Recipe 3 to find out what changed, then
update this file.
