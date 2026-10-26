# Next session plan (2026-10-25 evening, after the unattended run)

Start in `~/Desktop/AutoCroissant`. Read memories `owner-decisions-2026-10-24-night`, `resource-limits`,
`lean-tests-and-verification`, `no-guides-csv-only`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`.
What landed on 10-25: `docs/Finished/session-2026-10-25/README.md` (10-24: `docs/Finished/session-2026-10-24/`). Designs still queued: `docs/Future/`.

**Standing rules:** work unattended unless the owner says "ask me questions". Agents on Opus or lower. Commit to main;
`work/*` only while in flight. Persistent paths only. Deploy with the `jetson-deploy` skill. Tests:
`~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests` (about 570 tests, about a minute; `-n auto` = 6 workers).
**Resource limits (memory `resource-limits`):** one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`),
at most 2 workflows, pools of 4, never a partial clone of the local card repo, the git watchdog under Monitor.

## State

AutoCroissant main be39f86 (deployed); TTSCardMaker main a4d137bf; data branch at parser 3 with details v2 (812 records
with parts). Fork pin `cards-20261025b`. The website editor is in box-per-layer mode on every card; the bake applies
per-layer edits (splice, delete, sha skip, variants, star folder moves, first type icon). The guards silently fix
Single-line Composer, curly apostrophes, badge size/centring and old stoogedings glyphs; sweeps landed (badges 101
cards, apostrophes 34). reminder.json is backed up to `backup/reminder.json` on change (SSH from
`~/Documents/AutoCroissant-backup`).

## In flight / prepared, not landed

- [ ] **Card sync verdicts, step 6**: migrate today's one `_psd` block after the first live bake or refusal shows
      `db/verdicts.json` and `db/bake-records.json` on data (`~/Documents/Tools/wt/scratch/verdicts/MIGRATION.md`); drop
      the `_psd` fallback one cycle later.
- [ ] **Variant tags** for the 5 multi-picture PSDs: commit on TTSCardMaker `work/varianttags`
      (`~/Documents/Tools/wt/tc-varianttags`); the plan to show the owner first:
      `~/Documents/Tools/wt/scratch/parts-ab/VARIANT_TAGS_PLAN.md`. Land after his look (then the numbered-PNG
      duplicate scan goes, PARTS_EDITOR §10 step 5).
- [ ] **Badges resized anyway** (owner 10-25: rewraps allowed): AutoCroissant `work/resize58` (the guards resize even when
      the text rewraps; an FX layer on moved words is cut again; a line pushed out of its box refuses), TTSCardMaker `work/resize58` 32039569 (43 cards, 13
      rewrapped; the other 11 are the cards below that need a Photoshop re-save). Land AutoCroissant first, then the card
      commit once `python -m commands.card_sweep moved --plan ~/Documents/Tools/wt/scratch/resize58/card-sweep-badges.json
      --clone ~/Desktop/TTSCardMaker` prints nothing.
- [ ] **OJWikibot optimization lane 3** (opt PLAN §(a)): in flight at the end of the night (`work/opt3`).

## Questions for the owner (recommended answers in bold in the source)

- Optimization pass: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b): S-1..S-6 (site bundle, first screen in HTML,
  lazy modules, WOFF2 preview fonts, CDN caching), B-1..B-5 (bot queue wake-ups, /query_ability grouping, card list at
  start, pandas), P-1..P-6 (guards window from the push, one main lane per guarded push, thumbs off Pages, **P-5 the
  Card sync fallback: the cron fires ~8% of the time**), T-1..T-5, K-1..K-6 (dirty-box gate 1.6x stat bakes, runner
  parallelism, edit_copy), O-1..O-9 (OJWikibot tests opt-in, throttle 2 s, outage responsiveness, retry storms...).
- Michael's Blessing: the badge sweep shrank its badge table (the "->" arrows no longer line up). Keep or revert that
  card from 218a122d?
- KungFuPandaRegular-xYjV.ttf (the one old font copy kept): no PSD asks for its PostScript name `Kung Fu Panda`; deleted on TC work/tooling (bcaca99f).

## Found, needs the owner

- **OJWikibot's wiki API answers HTTP 500 since 10-23** (last good login 10-22 01:19): wiki/host side.
- 11 cards' badges and 7 apostrophe cards could not be fixed glyph by glyph (stored ink outside the layout): listed in
  the night log; a Photoshop re-save fixes them.

## Queue

1. [ ] Re-run the problem-cards census (`docs/Future/PROBLEM_CARDS_2026-10-25.csv`) now that the parts editor and layout
       support landed; the CSV is the only deliverable (memory `no-guides-csv-only`).
2. [ ] `+ability` (PARTS_EDITOR §10 step 6, after a Photoshop proof of `import_layer`) and the §8 joined-splice deletion
       after one Card sync cycle with no legacy entries.
3. [ ] The parked research ideas below.

## Owner, eventually (never blockers)

Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page Number layers); the team's
rulings draft; developers resetting clones after the 10-24 rewrite and reinstalling stoogedings.

## Ideas parked / not chosen

Unchanged from `docs/Finished/session-2026-10-24/README.md`.
