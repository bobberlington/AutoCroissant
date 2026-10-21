# docs

Start a session at `docs/NEXT_SESSION_PLAN.md`. Everything here is prose; procedures and code conventions live in the skills.

## Index

| Document | What it is | Read it when |
|---|---|---|
| `docs/NEXT_SESSION_PLAN.md` | The plan: state, decisions, tasks with status, Queued, Ideas 5-23, Order | Starting a session; before picking work |
| `docs/EDIT_PIPELINE_DESIGN.md` | The PSD edit pipeline (Task 3 steps 4-7): two kinds of edits, per-card gates, delta merge, Actions runner with jetson fallback, "PSD pending", acceptance | Before implementing or changing how website edits reach the PSDs |
| `docs/BUGS_AND_IMPROVEMENTS.md` | Open bugs (B/D/V rows), retired rows, candidate improvements; each row names its owner skill | Before fixing or proposing anything; after a fix (move the story to failure-archaeology) |
| `docs/Campaigns/SIMULATOR_CAMPAIGN.md` | Play tab: hand test stats, sandbox table, rules automation, ability scripts, batch simulation (item 16) | Designing or estimating the Play tab |
| `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` | Game reports from Tabletop Simulator and the browser; home of the report schema, §4 (item 17) | Anything that reads or writes a game report |
| `docs/Campaigns/AI_CARDS_CAMPAIGN.md` | Generated cards: isolation, text and stats, template fill, art, gallery, Discord (item 18) | Before any AI card work (on the proposed branch `ai/cards`, AI_CARDS D2) |
| `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` | Rulebook contradictions, undefined tags, hidden text; checker, decision log, fixes (item 19) | Before touching rulebook pages or rulings |
| `docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md` | The ledger of every intended PSD edit: token migration, stat rows, digit effects, Ability/Name layers, status-word FX, title reprints, the guard (item 21; on hold until the user lifts it, 2026-10-20) | Before any PSD or PNG write in TTSCardMaker; when the edit pipeline meets an odd PSD |
| `docs/Campaigns/WEB_EDITOR_CAMPAIGN.md` | Card as data: one data file per card as the truth, PSD and PNG as build outputs, tiered migration with gates, browser preview options, the extended editor, sync guarantees (item 22) | Before extending the website editor past item 20, or before anything writes card data in a second place |
| `docs/Finished/optimization-pass-2026-10-20/` | Task 4's audits: `README.md`, `profile.md`, `network-robustness.md`, `network-speed-results.md`, `code-health.md`, `docs.md`, `refs_table.md`, `bench/` | Looking up a number measured in that pass |
| `docs/Finished/optimization-pass-2-2026-10-21/` | Pass 2 over the edit pipeline, font check, site badges and commit paths: `README.md` (numbers before/after, what changed, what was skipped), `perf.md`, `code-health.md`, `tests.md`, `bench/` | Looking up a bake's or font check's cost, or why a finding was skipped |

## Conventions

- **Campaigns** are designed, not scheduled. Each opens with `Status: designed <date>, not scheduled; plan item N` (`idea <date>`
  for a stub) and a link to the plan, and has a section of open questions for the user. Rule citations are rulebook pages as the
  rules skill names them; nothing invents a rule. When a campaign ships, its status moves to the plan and its report to Finished.
- **Finished** holds reports of completed passes, one dated folder each with a README. Numbers there are frozen; the summary
  lives in the bugs doc or the plan.
- **The plan** is the entry point for a session: state, decisions, tasks each with a status line, Queued, Ideas, Order. Terse
  prose, game vocabulary (creatures, minions, items, cards), paths that resolve from the repo root.

## Where other knowledge lives

- `.claude/skills/` (17 skills): how the bot works, operations, debugging, change control, the card data, the game rules,
  TTSCardMaker. Start with `autocroissant-architecture-contract`; `autocroissant-docs-and-style` owns the skill conventions.
- `tests/`: the offline suite (parser goldens in `tests/goldens/`); `~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests`.
