# docs

Start a session at `docs/NEXT_SESSION_PLAN.md`. Everything here is prose; procedures and code conventions live in the skills.

## Index

| Document | What it is | Read it when |
|---|---|---|
| `docs/NEXT_SESSION_PLAN.md` | The plan: the state now, what is in progress, the owner's queue, what waits on the owner and on developers, ideas parked | Starting a session; before picking work |
| `docs/Future/` | One short design per decided, unbuilt item, in queue order: `PARTS_EDITOR.md`, `TEST_SUITE_CUT.md`, `SMALLER_PNG_PUSHES.md`, `WEBSITE_SMALL_FIXES.md`, `LAYOUT_SUPPORT.md`, `CARD_SYNC_VERDICTS.md`, `GAME_TXT.md`; and `PROBLEM_CARDS_2026-10-25.csv`, the census. `README.md` there indexes them | Before building a queue item; before planning bake or layout work |
| `docs/EDIT_PIPELINE_DESIGN.md` | The PSD edit pipeline (Task 3 steps 4-7): two kinds of edits, per-card gates, delta merge, the bake in Card sync, "PSD pending", acceptance | Before implementing or changing how website edits reach the PSDs |
| `docs/BUGS_AND_IMPROVEMENTS.md` | Open bugs (B/D/V rows), retired rows, candidate improvements; each row names its owner skill | Before fixing or proposing anything; after a fix (move the story to failure-archaeology) |
| `docs/Campaigns/SIMULATOR_CAMPAIGN.md` | Play tab: hand test stats, sandbox table, rules automation, ability scripts, batch simulation  | Designing or estimating the Play tab |
| `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` | Game reports from Tabletop Simulator and the browser; home of the report schema, §4 | Anything that reads or writes a game report |
| `docs/Campaigns/AI_CARDS_CAMPAIGN.md` | Generated cards: isolation, text and stats, template fill, art, gallery, Discord | Before any AI card work (in a separate AI sandbox repo with its own website and bot, AI_CARDS D1-D3 as revised 2026-10-21) |
| `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` | Rulebook contradictions, undefined tags, hidden text; checker, decision log, fixes | Before touching rulebook pages or rulings |
| `docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md` | The ledger of PSD edits (batches done 10-22/23; the owner standardizes PSDs himself since) and §5, the card standards the push guards check | Before changing a standard or a batch PSD write |
| `docs/Campaigns/DATA_BUILD_ACTION.md` | The Card sync Action: card data built on GitHub (orphan `data` branch), the bot as a reader, the migration steps 1-10 with their status (cut over 2026-10-24; two lanes since 10-25; step 10 in progress) | Anything about Card sync or the data branch |
| `docs/Campaigns/PUSH_GUARDS.md` | The push guards (card-guards.yml): stale-PNG export, the card standards, posts to the standards channel; §15 the owner's open points | Before changing `CARD_GUARDS` or a guard rule |
| `docs/Campaigns/WEB_EDITOR_CAMPAIGN.md` | Card as data: one data file per card as the truth, PSD and PNG as build outputs, tiered migration with gates, browser preview options, the extended editor, sync guarantees | Before extending the website editor past the parts editor, or before anything writes card data in a second place |
| `docs/Finished/optimization-pass-2026-10-20/` | Task 4's audits: `README.md`, `profile.md`, `network-robustness.md`, `network-speed-results.md`, `code-health.md`, `docs.md`, `refs_table.md`, `bench/` | Looking up a number measured in that pass |
| `docs/Finished/session-2026-10-2*/` | Session reports: what landed, numbers, owner rulings, incidents, lessons (2026-10-21, 2026-10-22, 2026-10-22b with the plan it worked from, 2026-10-23 with the archived resume list, 2026-10-24 with the 10-23 plan, the unbakeable list and the standardization rehearsal) | Looking up when or why something landed |
| `docs/Finished/optimization-pass-2-2026-10-21/` | Pass 2 over the edit pipeline, font check, site badges and commit paths: `README.md` (numbers before/after, what changed, what was skipped), `perf.md`, `code-health.md`, `tests.md`, `bench/` | Looking up a bake's or font check's cost, or why a finding was skipped |

## Conventions

- **Campaigns** are designed, not scheduled. Each opens with `Status: designed <date>, not scheduled` (`idea <date>` for a
  stub) and is listed under the plan's "Ideas parked"; its open questions sit in the doc. Rule citations are rulebook pages as the
  rules skill names them; nothing invents a rule. When a campaign ships, its status moves to the plan and its report to Finished.
- **Future** docs are decided and queued: Status, goal, the owner's rulings, design, lean verification, real open
  questions only. Short prose. Shipped ones move to Finished.
- **Finished** holds reports of completed passes, one dated folder each with a README. Numbers there are frozen; the summary
  lives in the bugs doc or the plan.
- **The plan** is the entry point for a session: State, In progress, Queue (the owner's order, each pointing at its
  Future doc), Waiting on the owner, Waiting on developers, Smaller open items, Ideas parked, Not chosen; every open
  task has a status. Terse prose, game vocabulary (creatures, minions, items, cards), paths from the repo root.

## Where other knowledge lives

- `.claude/skills/` (17 skills): how the bot works, operations, debugging, change control, the card data, the game rules,
  TTSCardMaker. Start with `autocroissant-architecture-contract`; `autocroissant-docs-and-style` owns the skill conventions.
- `tests/`: the offline suite (parser goldens in `tests/goldens/`); `~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests`
  (the fork venv: on stock psd-tools the bake tests skip).
