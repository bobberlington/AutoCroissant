---
name: autocroissant-analysis-toolkit
description: "First-principles analysis recipes for AutoCroissant — load this when you must PROVE something rather than pattern-match it. Triggers: \"root cause\" a mis-parsed card or wrong [type] injection; \"prove\" / \"which code path produced this data\" / forensic attribution of pickle contents; \"bisect\" / \"when did X change\" / \"which commit introduced\" (git log -S, git show time-travel); \"measure before running\" an API budget for /update_stats vs GitHub rate limits; \"why is the bot slow to reply\" queue-timing reasoning; \"pickle forensics\" / comparing stats.pkl or old_stats.pkl snapshots across git history without running the bot; any \"analysis method\" question where the debugging-playbook triage tables don't cover the symptom. Contains six worked recipes with real commands and real outputs from this repo's history: the gap_trace 6-step pipeline on Nebulous Precipice (The Lich King as the font-token case since 2026-10-22), the path-style experiment that root-caused the removeprefix bug, the \"am forehead\" diff-reading lesson, the /update_stats API budget arithmetic (1 request when nothing changed; range reads on vs off), the ≤0.2 s queue-latency derivation, and an actually-run cca0aaf-vs-eb9aa84 old_stats comparison."
---

# AutoCroissant analysis toolkit

Date-stamped 2026-07-11; Recipe 1's worked example, Recipe 4's table and the line numbers re-measured 2026-10-23 (main
2ada357). All commands assume cwd = repo root `/Users/michaelsrouji/Desktop/AutoCroissant`. The PSD-reading scripts need
psd-tools and pandas: run them with `~/.venvs/ac-py314-fork/bin/python` (the Mac's plain `python3` lacks both).

This skill is the project's method book: how to turn "this looks wrong" into a mechanism you can
defend. The house evidence bar (owned by autocroissant-research-methodology): **one mechanism must
explain ALL observations, including the negative ones, and your hypothesis should predict numbers
BEFORE you run the measurement.** Every recipe below follows that bar and ends with a worked example
that was actually executed against this repo on 2026-07-11 — the outputs shown are real, not
illustrative.

Why this matters here specifically: this repo has NO CI and only an offline pytest suite (`tests/`), the pickles ARE the database,
and history contains commits whose messages are jokes ("am forehead") and commits whose messages
look routine but ballooned the database ("PICKLE"). Pattern-matching on names, messages, or vibes
has burned this project repeatedly; the recipes below are the antidote.

## Ground rules for every recipe

- The repo is treated READ-ONLY for analysis: read-only git only (`log`/`show`/`diff` — never
  add/commit/checkout/reset), never run the bot (`python3 main.py`), never call `update_stats()`,
  never write any pickle in the repo. Snapshots you extract go to `/tmp` or a scratch dir.
- `config.py` (repo root) is SECRET (tokens). Import chains read it; never print its values.
- Since 2026-10-20 only main.py prints the token-presence line; importing the card modules prints
  nothing, so the `2>&1 | grep -v "Git token"` filters below are harmless leftovers.
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
cards through the GitHub API (**remote mode**). **Local mode** (walking the clone) was the other path
until it was archived on 2026-10-20 (now on branch `archive/local-update-stats`, not in main); the recipes below still mention it
because it explains old data and history.

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
   psd_parser.py:348), [4] below-midline icons, sort order, and the prune decision
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

**Worked example — Nebulous Precipice as a healthy trace** (all outputs real, 2026-10-23, TTSCardMaker 96d61368,
`~/.venvs/ac-py314-fork`):

`dump_psd_layers.py` on `~/Desktop/TTSCardMaker/"Field/3 Stars/Nebulous_Precipice.psd"` shows size `1200 x 1800`
(midline 900), the field's own type icon above the midline, one `Ability` text layer at (26, 1229) and a `darkmatter`
icon layer at (269, 1207) — below the midline, sitting in the gap after "Friendly".

`gap_trace.py` then prints exactly:

```
PSD 1200x1800; TYPE_REGION_RATIO=0.5 -> card_mid_y=900
[1] creature-types collected ABOVE midline: ['field']
[2] ability text layers found: 1
    at (x=26, y=1229): "'Friendly     gain +0.5 DEF and +1 SPD.\n(1GCD) Put a darkness counter ...'"
[3] Friendly<GAP:5>gain +0.5 DEF and +1 SPD. ...
[4] type icons BELOW midline (candidate inline types): 1
    darkmatter      at (x=269, y=1207)
    prune threshold = max(last_y//3=402, card_mid_y=900)
    kept after prune: ['darkmatter']   dropped: []
[5] gap count = 1 vs kept types = 1   (match)
[6] FINAL ability text ... Friendly [darkmatter] gain +0.5 DEF and +1 SPD.
```

`parse_one.py` confirms the injected text and `validator problems: NONE` — every number predicted by the trace.
(Canonical current expected values live in autocroissant-validation-and-qa §3; if the trace stops matching, check
there before assuming a regression.) Nebulous Precipice is due to get a `[darkmatter:glow]` font token once the glow
font lands (owner, 2026-10-23); then pick another card whose `psd_details.json` `icons` field says `images` or `mixed`.

**The token case — The Lich King** (the 2026-07-11 golden trace, re-run 2026-10-23). On 2026-07-11 its `Ability` layer at
(23, 1773, 1176, 2285) read "Minions inherit       ." and a visible `Undead` smartobject at (734, 2239) filled that
`<GAP:7>` (prune `max(746, 1200)`, 1 vs 1). Since the standardization batch (TTSCardMaker 1c24dd48, 2026-10-22) the layer
sits at (23, 1767, 1176, 2293) and reads "Minions inherit [undead]." as a stoogedings token, and the smartobject is
hidden: `[1]` ['ice', 'undead'], `[2]` one layer at y=1767, `[3]` no GAP, `[4]` 0 icons (kept [] dropped []),
`[5]` 0 vs 0 (match), `[6]` the same final text. It shows a token-typed card, not gap injection. (No certified golden
fills a gap from an icon layer any more; Tainted Lazarus, 16_The_Field and 20_Creature_Types exercise the leftover
append, Twin Emperors and 00_Table_of_Contents have unfilled gaps.)

**Failure signatures** — how each classic failure looks in the same trace:

| Signature | What you see | Mechanism it points to |
|---|---|---|
| No text layer | `[2] ability text layers found: 0`; parse_one shows `NO ABILITY LAYER` (appended at psd_parser.py:449) and/or `ABILITY TEXT NOT FOUND` (validator, psd_parser.py:1104) | Text layer missing, or not named `ability` on a non-rulebook card — check the dump for a `type`-kind layer with an unexpected name |
| Icon above midline | The icon appears in `[1]`, not `[4]`; the gap in `[3]` stays unfilled | Icon y < 1200 → it's classified as a creature type, never an injection candidate. TYPE_REGION_RATIO=0.5 is deliberate — do NOT "fix" by injecting top-half icons (fenced off in autocroissant-psd-extraction-campaign) |
| Gap/type mismatch, extra types | `[5] gap count = N vs kept types = M   <-- MISMATCH, leftover types get appended to last line!` | `_inject_type_names` appends leftovers as ` [x] [y]` onto the LAST line (psd_parser.py:1045, "Append remaining types") — the classic "types stuck at the end of the ability" symptom |
| Gap/type mismatch, extra gaps | `[5]` MISMATCH the other way; final text keeps raw multi-space runs — and a gap directly before punctuation silently VANISHES (cleanup `\s+([:;,\.\?!])` → `\1`, psd_parser.py:947) | Fewer icons survived than gaps exist — check `[4]`'s `dropped:` list (prune threshold ate them) before blaming the text |

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
   `update_stats`, repo_traverser.py:486 as of 2026-10-20 (:61, its traversal in card_data.py, since 2026-10-22); until that day it branched on `use_local_repo`,
   and since then the local code path lives on branch `archive/local-update-stats`, outside main).
2. **For each path, derive an observable fingerprint it MUST leave** in the artifact. Prefer
   fingerprints that differ per path and cannot be produced accidentally by the other path.
3. **Check the artifact** with standalone reads (Recipe 6 mechanics). Every fingerprint must agree;
   one disagreement kills the attribution.

**Worked example — which mode built today's stats.pkl?** This is the experiment that established
the removeprefix bug (introduced by f7c915c, fixed 2026-10-19; full incident owned by
autocroissant-failure-archaeology Entry 1) had NOT corrupted production data. The method still
applies after the fix: the TIMESTAMP fingerprint below is exactly how you would spot pickles from a
local-mode run (local mtimes instead of commit times; docs/BUGS_AND_IMPROVEMENTS.md B6; since 2026-10-20 only the
archived `update_stats_local()` can produce them), even though the path fingerprint no longer differs
between modes.

Candidate paths and their fingerprints:

| Path | Path-string fingerprint | Timestamp fingerprint |
|---|---|---|
| Local mode (`_process_local_files`; psd_analyzer.py:1002 on 2026-07-11) | Before 2026-10-19: `relative_path = full_path.removeprefix("TTSCardMaker").strip('/')` on an ABSOLUTE walk path was a no-op → stored paths start `Users/michaelsrouji/...`, classification sees top folder "Users" → mass UNKNOWN type. Since the `relpath` fix: repo-relative, same as remote | `getmtime` → local clone mtimes (whenever the clone was last checked out) — still distinct from remote |
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
   # 53db8ee Parser fixes, card authors file, type emojis, reminders   <- removed it (the 2026-10-19 fix)
   # f7c915c cythonized the code                <- introduced the local-path bug
   git log -S 'TYPE / WHITESPACE MISMATCH' --oneline
   # 081b1fd fix bug with types being improperly placed in abilities   <- removed it
   # fed8a83 output a problem if num types does not match num gaps in ability   <- added it
   ```

   Both real (verified 2026-07-11; the 53db8ee line 2026-10-20). The second one instantly answers "we used to WARN on gap/type
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
  content-changed cards, so the gate finally does its job. (Since the 2026-10-20 code-health pass the counter is
  `num_changed`; in `card_data.update`, commands/card_data.py since 2026-10-22, the prune is gated by `can_prune(build)`
  (card_data.py:968: whole tree seen, no 304, not truncated, no failed new-name PSD; called at :1146) and the save
  (:1158) by `num_changed > 0 or gone or backfilled or build.changed or line or again or db.stale_archives()`.
  `_process_files_from_response` still returns `num_changed + num_moved`, :692.)
- `git show 2ff52b2` ("Update psd_analyzer.py", 18:55) — despite the plausible message, it's ONLY
  type annotations and parenthesization. Attributing the gate to it would be wrong.
- Bonus name-lie inside the same loop (renamed 2026-10-20): until then `num_new` ALSO counted re-parsed existing
  cards (the `else: num_new += 1`, including `cards:`/`force_update` re-parses) — "new + re-parsed", not "new" — and
  `_process_files_from_response` returned `num_new + num_moved`. Only the increment sites told you that. The counter is
  now `num_changed` (commands/card_data.py since 2026-10-22).

**Conclusive result:** you can name the introducing (and, if applicable, removing) commit, quote
the exact changed line from `git show <hash>:<file>`, and your story survives reading the
neighboring commits from the same day. If two commits within minutes touch the same lines, the
LAST one is usually the real fix and the first one the attempt — check both.

---

## Recipe 4 — Measure an API budget before running (remote /update_stats)

**When to reach for it:** BEFORE any run that talks to the GitHub API in a loop — remote-mode
`update_stats`, or the archived local traversal with `use_local_timestamp=False`. Predict the request count first;
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

2. **Read the loop and tag every network call** (`commands/card_data.py` since 2026-10-22, re-pointed 2026-10-23 at 2ada357):
   - `traverse_remote` — one tree GET (`self._fetch_tree`, :269), sent with If-None-Match
     unless `force_update` or `cards:` is set. A 304 ends the run (:270). The ETag
     (`_tree_etags`) and the small-file cache (`_file_cache`) are module dicts in repo_traverser.py:31-32, in
     memory only, so the first sweep after a restart always gets a 200.
   - Small files (icon manifest, authors, edits, Card Creation Dates.json; `_repo_text`, :154) — one git/blobs
     request each, only when its blob sha changed or the bot restarted: **up to 4**, plus `.mailmap` once a run
     when a new card needs an author (2026-10-23).
   - Changed cards are found by comparing tree blob shas with `card.blob_sha` (`_psd_changed`,
     :237) — **no request per unchanged card**. The PyGithub repo is lazy (`_repo()`, :136): 0 requests.
   - Per CHANGED card: one commit page (`_commit_page`, :145-148; per_page=100, run in a 2-thread pool beside the
     reads) that holds the newest commit and, for a new or authorless card, the first; a file with 100+ commits adds
     one (`.reversed[0]`, :222). The PSD itself, **since 2026-10-23 with range reads on** (`RANGE_READS`,
     github_api.py:27, and a psd-tools with `range_reader`: the jetson and `~/.venvs/ac-py314-fork`): HTTP range
     requests to raw.githubusercontent at the tree's commit, which are **not REST API requests**; a run with changed
     cards adds one `branches/<CARD_BRANCH>` request to find that commit (`_tree_commit`, :702). With range reads off,
     or as the fallback (3 refusals turn them off for the run), the whole blob from api.github.com git/blobs
     (`_download`, :164): **2 REST requests per card**. Before 2026-10-20 it was 3 requests plus (commits − 1) for a
     new card.
   - Legacy cards with no `blob_sha` (one-time): `_committed_since_update` (:124) costs 1
     `get_repo` + the `get_commits(since=)` pages + 1 request per commit for `.files`.
   - Before 2026-10-20 (blob shas) every sweep called `get_commits(path=)` for every PSD: ≈815 requests even
     when nothing changed.

3. **Do the arithmetic against the limits.** The code prints its own reminder
   (github_api.py:39-42, `print_token_status`): 60 req/hr anonymous, 5000 req/hr with a token.

   | Sweep | REST requests, range reads on (2026-10-23) | Range reads off | Verdict |
   |---|---|---|---|
   | Nothing changed | 1 (a 304; free with a token) | 1 | Trivial |
   | First after a restart, nothing changed | 1 + ≤4 = 5 | 5 | Trivial |
   | k changed cards | 1 + ≤4 + 1 + k (+1 per new/authorless card with 100+ commits; +1 `.mailmap` with a new card) | 1 + ≤4 + 2k | Anonymous: about 50 (on) or 28 (off) cards per hour; with a token, no practical limit |
   | `force_update` | ≈ 813 + ~6 ≈ 820 | ≈ 2 × 813 + 5 ≈ 1,630 (≈ 2,450 before 2026-10-20) | About a sixth (on) or a third (off) of the hourly budget with a token; impossible without one |

**Conclusive result:** a predicted request count derived from the loop, checked against the input
count, BEFORE anything runs — and the prediction is falsifiable (if a sweep rate-limits earlier
than predicted, your reading of the loop is wrong; re-read it). The general form: fixed calls +
(per-item calls × items surviving filters) + (per-changed-item calls × expected changes).

---

## Recipe 5 — Reason about queue timing

**When to reach for it:** "the bot took N seconds to answer / the reminder fired late / is the
queue stuck?" — decide from first principles whether the delay can even BE queue delay before
digging anywhere else.

**The mechanism** (verify at main.py:831-979; the deques commands/utils.py:25-27, queue helpers :36-63): three module-level
deques (`command_queue`, `dispatch_queue`, `edit_queue`) each drained by a `@tasks.loop(seconds=0.1)`
coroutine (1 s until 2026-10-21). Each tick drains the WHOLE deque (`while q: popleft()`), so backlog does not add
one-tick-per-item delay. Worker code never awaits Discord; it appends via
`queue_message`/`queue_file`/`queue_edit`/`queue_command`.

**Derived latency budget:**

| Path | Hops | Queue-added latency |
|---|---|---|
| Slash command reply | wrapper awaits the `to_thread` worker immediately (no command-queue hop) → worker enqueues send → dispatch tick | ≤ 0.1 s (avg ~50 ms) |
| Queued command (reminder-fired via `queue_command(_run_reminder_command, ...)` in analytics.py `_fire`, on_ready's queued commands main.py:134-139 incl. sweep_temp_psds and refresh_history, music chaining) | enqueue → ≤0.1 s command tick → `create_task` + thread runs → worker enqueues send → ≤0.1 s dispatch tick | ≤ 0.2 s (avg ~0.1 s) |

Two structural caveats, both visible in the code: (a) `process_reminders`, a separate 1 s loop (main.py:926-928 as of
2026-10-23), enqueues `check_reminder` and the command loop pops it within a tick — reminders are still checked
every second, then their commands take the queued-command path above; (b) the dispatch
loop awaits each send SEQUENTIALLY inside the tick (main.py:945-954 as of 2026-10-23; the drains are :932/:945/:957), so one slow Discord HTTP call
delays the sends behind it in that tick — that is Discord I/O time, not queue design time.

**Measured 2026-10-20** (Mac, sandbox; `docs/Finished/optimization-pass-2026-10-20/profile.md`, bench scripts in
`docs/Finished/optimization-pass-2026-10-20/bench/`): `queue_message` to send 502 ms mean (p95 966 ms) against 3.9 ms of /query work, so
the tick is ~99% of /query's bot-side time; a reminder takes 2 hops (~1 s). An ability query queues one message per
result (85 per /query_ability on average, median 15, max 763), all through the one dispatch loop; idle cost of the
1 s loops was 0.33 ms CPU/s. **Changed 2026-10-21** (owner decisions, archaeology 10.30): the drains tick every 0.1 s, so
the queue bound is ≤ ~100 ms per hop and ≤ ~200 ms for a reminder's two hops. /query_ability listed cards 10 per
message from then until 2026-10-23, when it went back to one message per card (Discord previews only a few images per message). Idle cost of the four loops (`docs/Finished/optimization-pass-2026-10-20/bench/idle.py` on a snapshot, 10 s, no Discord): 1.29 ms CPU/s with 0
reminders, 1.53 with 200; 2 threads alive.

**The diagnostic implication (the point of the recipe):** the queues can contribute at most ~0.2 s.
**Any "hang" over ~1 second is thread work (the worker function itself) or Discord I/O
(send/edit HTTP, or the fallback chain after a 15-minute interaction-token expiry — see
autocroissant-debugging-playbook), never queue scheduling.** Don't investigate the deques for a
30-second stall; time the worker.

**Conclusive result:** you state the expected latency for the exact path the command took (≤0.1 s or
≤0.2 s of queue overhead), compare with the observed delay, and the difference tells you which
component to profile next. This recipe never requires running the bot — it's arithmetic over
main.py:831-979.

---

## Recipe 6 — Pickle forensics without the bot

**When to reach for it:** any question about what's IN a pickle, how it changed across history, or
whether a snapshot is damaged — answered with standalone reads, never by running the bot.

**Mechanics.** `stats.pkl` values are `CardInfo` dataclass instances pickled as
`commands.psd_analyzer.CardInfo` (defined in `commands/card_models.py` since 2026-10-20, `__module__` pinned). Since
2026-10-22 the forensics pattern is `commands.card_db.load_pickle` (its `SeedUnpickler` allow-list, card_db.py:275-284):
it loads stats.pkl / old_stats.pkl / aliases.pkl with neither config.py nor psd-tools, on the Mac's plain `python3`.
The raw `pickle.load` path still imports the `commands.psd_analyzer` chain (discord, pandas, psd-tools, config.py) and
the older one-liners below use it. Load pattern:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from commands.card_db import load_pickle
db = load_pickle('stats.pkl'); old = load_pickle('old_stats.pkl')
print(len(db), len(old), sum(len(v) for v in old.values()))"
# 818 354 446     (PICKLE a7c310e, 2026-10-23)
```

Aggregate with one-liners (Counter over `c.path.split('/')[0]`, max over `c.timestamp`, filter on
`c.problems` — e.g. on 2026-10-23 no card has a *stored/pickled* problem (0; on 2026-07-11 it was `('20 Creature
Types', ['MISSPELT TYPE: tornado'])`), and full validation (`CardValidator.validate` over stats.pkl) computes 1:
Computer Virus, `HP/DEF/ATK/SPD NOT FOUND` (5 on 2026-07-11, before the three WoW minions joined ABILITY_EXCLUSIONS in
e4e0a58 and tornado became wind) — surface distinction in autocroissant-validation-and-qa §2). For the packaged summary use `inspect_pickle.py`; for two-snapshot diffs use
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
annotated trace, from the 66-byte first archive through 82,881 on 2026-07-11, is owned by
**autocroissant-failure-archaeology** Entry 3; at PICKLE a7c310e, 2026-10-23, old_stats.pkl is 193,937 B, 446 versions,
and about 1,155 versions are expected after the next run's history recovery):

| Commit | Date | Bytes | Note |
|---|---|---|---|
| e380cee | 2025-11-03 | 3,196 | pre-incident baseline |
| cca0aaf | 2025-11-10 | 12,041 | balloon #1 ("PICKLE") |
| eb9aa84 | 2025-11-10 | 3,196 | Revert "PICKLE" — restored exactly |
| 637698b | 2025-11-25 | 12,039 | balloon #2 ("PICKLE") — **never reverted** |

(Root-cause family — duplicate archiving of unchanged cards — and incident status:
autocroissant-failure-archaeology. The archival guard `should_update and not is_new` landed
e7befd5/fb47b5d; since 2026-10-20 a card is archived only when its PSD blob changed or it moved
(`_history`, card_data.py), so `cards:`/`force_update` re-parses of unchanged PSDs add nothing; since 2026-10-22 only
when its data changed too (`_record_history`, `card_history.same_data`). Since 2026-10-23 an applied Card Edits or
Discord edit that changes data also keeps a version (`card_history.record_edits`, cause `edit`), and every
/update_stats runs `card_data.clean_history` (dedupe, normalize, recover from the bot's PICKLE commits, misreads,
unknown values, archive tags dated from git): the history rules are impossibility-cards-reference §6.
The save gate story is Recipe 3.)

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
commits, branch `main`). On 2026-10-20 the rows below that cite code were re-pointed to that day's
line numbers (HEAD 9fb3093), and again after that day's optimization pass; the worked examples' OUTPUTS are still the
2026-07-11 measurements. Volatile facts and their one-line re-checks:

| Fact | Re-verify with |
|---|---|
| Scripts exist / arg handling (full paths needed) | `ls .claude/skills/autocroissant-diagnostics-and-tooling/scripts/` ; `grep -n 'expanduser().resolve()' .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py` |
| Nebulous Precipice golden trace ([1]-[6] numbers); The Lich King's token case | rerun the `gap_trace.py` command in Recipe 1 (with `~/.venvs/ac-py314-fork/bin/python`) |
| Gap regex `\s{3,}` :348; punctuation cleanup :947; leftover-append :1045; NO ABILITY LAYER :449; ABILITY TEXT NOT FOUND :1104 (2026-10-23) | `grep -n '_gap_pattern = \|\[:;,\|Append remaining types\|NO ABILITY LAYER\|ABILITY TEXT NOT FOUND' commands/psd_parser.py` |
| removeprefix no-op line — FIXED 2026-10-19 | `grep -c 'relpath\|removeprefix' commands/psd_analyzer.py` (0; local mode archived 2026-10-20); story in autocroissant-failure-archaeology Entry 1 |
| stats.pkl = 813 entries, 0 `Users/` paths, newest 2026-06-18 | first two one-liners in Recipe 2, or `inspect_pickle.py` |
| Timestamp clustering (103 distinct; 769 shared; max cluster 64 @ 2026-02-08 04:11:26Z) | the Counter one-liner in Recipe 2 (drifts with every sweep — recompute, don't reuse) |
| Newest-card commit-time match | the `git -C ~/Desktop/TTSCardMaker log -1` + `stat -f '%Sm'` pair in Recipe 2 (mtime changes on every pull — only the COMMIT time is stable) |
| `-S` results (f7c915c; fed8a83/081b1fd) and commit dates | rerun the two `git log -S` commands in Recipe 3; `git log -1 --format='%h %ad %s' --date=short <hash>` |
| fb47b5d gate ineffective / 4e03190 real fix / 2ff52b2 cosmetic | `git show fb47b5d`, `git show 4e03190`, `git show 2ff52b2` (read increments, not names) |
| `num_changed` gates at HEAD | `grep -n 'can_prune(build)\|def can_prune\|db.save()' commands/card_data.py` (968, 1146, 1159) |
| Range reads and the branch-head request | `grep -n '^RANGE_READS' commands/github_api.py` (27); `grep -n 'def _tree_commit\|def _fetch_psd' commands/card_data.py` (702, 716) |
| PSD counts 904 / 91 excluded / 813 processed | the three `find` commands in Recipe 4 (drift EXPECTED as cards are added — recompute, don't reuse) |
| get_commits only for re-parsed cards; rate-limit warning | `grep -n '_get_remote_timestamp\|get_commits' commands/card_data.py; grep -n 'requests/hour' commands/github_api.py` (card_data.py since 2026-10-22, github_api.py since the 2026-10-20 split) |
| Queue loops drain the whole deque each 0.1 s tick; reminders on a 1 s loop | `grep -n 'tasks.loop\|while command_queue\|while dispatch_queue\|while edit_queue' main.py` (926 at 1 s; 932/945/957 at 0.1 s on 2026-10-23) |
| Slash option `use_local_repo` (default True through 2026-10-19, which is why the remote-mode ops rule existed; False on 2026-10-20 because of local timestamps, docs/BUGS_AND_IMPROVEMENTS.md B6; removed the same day when local mode was archived) | `grep -rn 'use_local' main.py commands/` (only `RETIRED_OPTIONS` in analytics.py); `git cat-file -e archive/local-update-stats:archive/local_update_stats.py && echo archived` |
| old_stats byte trace incl. 637698b=12,039 never reverted | `git log --format='%h %ad %s' --date=short main -- old_stats.pkl` + `git cat-file -s $(git rev-parse <hash>:old_stats.pkl)` |
| cca0aaf/eb9aa84 load + 39/13 counts | rerun the Recipe 6 extraction + one-liner |
| Problem card list (2026-10-23: none stored; full validation 1, Computer Virus) | the `c.problems` one-liner in Recipe 6; `CardValidator.validate` over `load_pickle('stats.pkl')` with the fork venv |

Line numbers drift with every edit to the card modules (`psd_parser.py`, `card_data.py`, `repo_traverser.py`, `card_models.py`, `psd_analyzer.py`) and `main.py` — trust the greps
above over the numbers in this file. If `PSDParser._extract_from_layers` changes shape, gap_trace's
mirror must be updated (see the MAINTENANCE NOTE in gap_trace.py itself). When a recipe's worked
example stops reproducing, that is itself a finding: apply Recipe 3 to find out what changed, then
update this file.

2026-10-20 (later): the profile and bench scripts cited in Recipe 5 moved from `~/Documents/ac-optimization/` to branch
`docs/Finished/optimization-pass-2026-10-20/`.

2026-10-21: Recipe 5 re-derived for the 0.1 s tick (≤0.2 s bound) and the separate 1 s reminder loop; main.py line numbers re-pointed. Review: bench path spelled out.

2026-10-21 (review): github_api.py citation re-pointed (:26-29; the module gained two imports above it).

2026-10-21 (titles): psd_parser.py and repo_traverser.py citations re-pointed (the parser reads the printed title into `display_name`; PARSER_VERSION 2).

2026-10-21 (review): psd_parser.py citations re-pointed after the WX-78 title fix; repo_traverser.py `update_stats` :472, history append :371-372.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-22 (data build): the traversal loop, its network calls and the save gates moved to the uncompiled commands/card_data.py (legacy mode for /update_stats); the recipe citations and greps re-pointed there. The API arithmetic is unchanged; a failed commit lookup now fails only its card.

2026-10-23 (docs drift): Recipe 1's worked example re-run on Nebulous Precipice (The Lich King became a font-token card
in TTSCardMaker 1c24dd48: its numbers kept as the token case); psd_parser.py citations re-pointed; Recipe 3's gates
(`can_prune`, the save gate with the History lines and stale archives); Recipe 4 gains range reads (REST cost about one
commit page per changed card; force_update about 820) and four small files plus .mailmap, with card_data.py and
github_api.py lines re-pointed; Recipe 5's main.py lines; Recipe 6 loads through `card_db.load_pickle`, problem
baselines 0 stored / 1 full (Computer Virus), old_stats 193,937 B at a7c310e, and points to the history rules.

2026-10-23 (unbatch): Recipe 5 says /query_ability posts one message per card again.
