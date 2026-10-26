# Next session plan (2026-10-26 start)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `owner-decisions-2026-10-24-night` (its 10-25
lines), `lean-tests-and-verification`, `no-guides-csv-only`, `edits-two-kinds`, `jetson-server`,
`persistent-work-paths`, `agent-model-preference`. What landed on 10-25: `docs/Finished/session-2026-10-25/README.md`
(the Evening section covers 17:10 to midnight). Designs still queued: `docs/Future/`.

First: check the `+ability` workflow's result (In flight, below).

## Standing rules

- Work unattended unless the owner says "ask me questions". Agents on Opus or lower.
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution. Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill: push main and deploy in one sitting.
- Tests: `~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests` (618 collected on 10-25; `-n auto` = 6
  workers). Targeted files unless a full run is needed.
- Resource limits (memory `resource-limits`):
  - never a partial (`blob:none`) clone of the local card repo; read history blobs with one `cat-file --batch`;
  - xdist at most 6; one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`);
  - at most 2 workflows; process pools at most 4;
  - the git watchdog (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) under Monitor during long unattended work;
  - no repack or gc of the big clones while agents run.
- `gh run list -c` needs the FULL sha (short shas made wait loops hang): `git rev-parse` first.
- zsh doesn't split unquoted variables: use arrays or `${=var}`.
- Send only GitHub tokens to GitHub. Never print config.py values or tokens.
- No guides or checklists for the owner: problem lists are a CSV plus its `.md` (memory `no-guides-csv-only`).

## State (2026-10-25 ~22:05 PDT)

- AutoCroissant main 4b29d55. The jetson runs 8b0ce02 (later main commits are tests and skills only).
- TTSCardMaker main 3853b688. The Desktop clone is at 8f2b3367, 32 behind, clean.
- Data branch 1cf84da1: parser 4 (builder 8b0ce02), 820 cards, details v2; no `db/verdicts.json` yet.
- psd-tools fork pin `cards-20261025b` (`requirements-cards.txt`).
- Card sync chains leftover work, with one safety cron at :17. Both proven on 10-25: run 40262751459 was dispatched by
  github-actions[bot] after 40262608396; the :17 cron fired at 21:19, 00:25 and 06:18 UTC (other hours' runs didn't
  come). The guards' window is 30 minutes from the push.
- OJWikibot main 8924db4. Its deployed commit is unknown (5135280 in the morning; a70877d, 8ea1c7c..174f62b and
  cdf6f03..8924db4 landed since): check it on the jetson and deploy with the `jetson-deploy` skill if behind.
- `psd_edit.ADD_ABILITY_LIVE = False`.

## In flight

**`+ability` placement and font** (AC worktree `~/Documents/Tools/wt/ac-plusplace`, branch `work/plusplace` at
c422934; scratch `~/Documents/Tools/wt/scratch/plusability2/`). One placement rule: the box starts at the top of the
card's free text area (below the stats, or the top of the empty panel), flows down, never off the card. The card's own
font when it exists; the template's only as the fallback. Measured leave-one-out on real cards. New test files go to
`~/std-work-1025/plusability/<card>/plus_ability_2/`. The branch adds `commands/ability_place.py` and
`commands/ability_index.json` (the other cards' facts, remade by hand: `python -m commands.ability_place
~/Desktop/TTSCardMaker`) and touches `psd_edit.py` and `psd_regions.py`.

What to do with its result:

1. Read its report; check the test files exist and nothing overwrote the first rehearsal files.
2. Land the branch on main with `ADD_ABILITY_LIVE` still False (targeted tests: `tests/test_psd_edit.py`,
   `tests/test_card_data.py`, `tests/test_card_guards.py`), then delete the worktree and branch.
3. Tell the owner the files are ready. Flip `ADD_ABILITY_LIVE` only after the owner approves them in Photoshop.
   Before the flip, both new files join the sparse lists: `card_data.SPARSE_FILES` and `card_guards.SPARSE_FILES`, and
   TTSCardMaker's card-sync.yml and card-guards.yml sparse-checkout lists. The flip goes live at the next Card sync run
   (it checks out AC main at `builder_ref`), not at a jetson deploy. Corrections from the owner go into a
   `plus_ability_3/` round.

## Open items

### Owner

- [ ] Rotate the Discord TOKEN and HF_TOKEN (sent to api.github.com by mistake on 10-25; GitHub answered 401).
- [ ] Open `plus_ability_2` when it lands; approve or correct (then flip `ADD_ABILITY_LIVE`). Possible option: the
      card family's own centring and colours.
- [ ] Photoshop re-saves: 11 badge cards and 7 apostrophe cards (session README), and the 54 problem cards
      (`~/Downloads/Problem cards 2026-10-25 (after fixes).csv` and `.md`; `docs/Future/PROBLEM_CARDS_2026-10-25b.*`).
- [ ] Safari checks: the grid flash, the many-boxes edge scroll.
- [ ] Mark PNG-only duplicates on the website.
- [ ] Data problems from the live problem list (`PROBLEM_CARDS_2026-10-25b.md`): Computer Virus has no stats (no
      template stat bars; Photoshop); Twin Emperors' Vek'lor has DEF 10 on the website and 4 on the card (an editor
      checks which is right).
- [ ] Pull the Desktop TTSCardMaker clone (32 commits behind, clean) and the Desktop OJWikibot clone (174f62b, 9
      behind).
- [ ] `site/profiles/MichaelJSr.json`: the series "Don't Starve Gourmet Expansion" isn't a known series. "Gourmet
      Expansion" is a subseries of Don't Starve (1 card, Grapes): probably series "Don't Starve" plus that subseries.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers); the team's rulings draft; developers resetting clones after the 10-24 rewrite and reinstalling
  stoogedings. Layout support still refuses Grimora, Skeleton Mate, Slugcat's spd, three smart-filter star layouts,
  PoD04 Zebra Bloon +1, Living Bread.

### Waiting on events

- [ ] Verdicts migration step 6 (`docs/Future/CARD_SYNC_VERDICTS.md`; runbook
      `~/Documents/Tools/wt/scratch/verdicts/MIGRATION.md` section B): one `_psd` block left in Card Edits.json
      (PoD11ZOMG's). Waits for the first live bake or refusal to write a verdict to the data; drop the `_psd`
      fallback one cycle later.
- [x] The joined splice (PARTS_EDITOR §8): already deleted on 10-25 (AC ab132b2, 0 legacy entries). Only the
      `_joined_edit` refusal for legacy joined edits remains.

### Outside

- [ ] OJWikibot's wiki API answers HTTP 500 since 10-23 (wiki/host side); O-6 waits for it.

### Developer

- [ ] Stubborn Queen and Gust of Trees are stuck under "Not parsed yet": PNG-only copies (`dup` set) that
      build_site.py's "card PNGs without a PSD" loop (site/build/build_site.py ~3486-3503) lists image only, with
      `parsed: false`. Teach the site these copies. Stubborn Queen's series also shows Other instead of Sort the Court!.

### Small known

- [ ] The font filter's included sidebar chip: a click removes it (no off state).
- [ ] Headless checks never ran Safari.
- [ ] The 8 first-screen tiles drop their href once adopted: middle-click on them opens no tab.
- [ ] 16 cards whose star layers the counter misreads keep a reworded flag (they print the right count).

## Queue (`docs/Future/`)

| File | What is left |
|---|---|
| `PARTS_EDITOR.md` | `+ability` live (In flight); the rest landed 10-25 |
| `CARD_SYNC_VERDICTS.md` | Step 6, the `_psd` migration (waiting on events) |
| `PROBLEM_CARDS_2026-10-25b.csv` / `.md` | The live problem-cards list (54 cards with an edit that can't bake); `PROBLEM_CARDS_2026-10-25.csv` is the morning's |

## Not chosen

Optimization questions answered no or not now: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b) (S-5, S-6, B-4, B-5,
P-3, P-4, P-6, K-4, K-6, O-7, O-8, O-9). Parked ideas: `docs/Finished/session-2026-10-24/README.md`.
