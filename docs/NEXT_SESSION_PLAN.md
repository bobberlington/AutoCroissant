# Next session plan (2026-10-25 night, after the afternoon lanes)

Start in `~/Desktop/AutoCroissant`. Read memories `owner-decisions-2026-10-24-night`, `resource-limits`,
`lean-tests-and-verification`, `no-guides-csv-only`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`.
What landed on 10-25: `docs/Finished/session-2026-10-25/README.md` (its Afternoon section for after 13:30; 10-24:
`docs/Finished/session-2026-10-24/`). Designs still queued: `docs/Future/`.

**Standing rules:** work unattended unless the owner says "ask me questions". Agents on Opus or lower. Commit to main;
`work/*` only while in flight. Persistent paths only. Deploy with the `jetson-deploy` skill. Tests:
`~/.venvs/ac-py314-fork/bin/python -m pytest -n auto -q tests` (about 595 tests, 1-2 minutes; `-n auto` = 6 workers).
**Resource limits (memory `resource-limits`):** one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`),
at most 2 workflows, pools of 4, never a partial clone of the local card repo, the git watchdog under Monitor.

## State

AutoCroissant main 4a4ad4c (jetson last known at be39f86: B-1's main.py and later bot code need a deploy); TTSCardMaker
main 54395592; data d4dd2172 (parser 3, details v2). Fork pin `cards-20261025b`. Card sync chains leftover work and
has one safety cron at :17 (none fired at 22:17 or 23:17 UTC; no Card sync run at all since 21:24 UTC). The guards'
window is 30 minutes from the push (`card-guards-quiet` timer 25, set). Reply queues wake on enqueue. The site ships
one esbuild bundle (editor, Stats and People load on first use) and WOFF2 mock-up fonts; the first screen in
index.html was removed for Safari's flash.

## In flight at 17:10 (each lane reports its own leftovers)

AC/TC `work/grid` (B-2: /query_ability as grid images), `work/cardlist` (B-3a: saved tree at startup),
`work/mainlane` (P-2), `work/bakecopy` (K-3 + K-5), `work/tooling` (T-1, T-3, T-4, T-5; Kung Fu Panda copy deleted,
TC bcaca99f), `work/resize58` (the 58 badges, resized with rewraps allowed), `work/pintests` (T-2), `work/partsfollow`,
`work/bakebugs`, OJWikibot `work/ojres` (O-1, O-4, edit conflicts), the problem-cards census re-run
(`~/Documents/Tools/wt/scratch/census-1025b`).

## Left

- [ ] **Safari flash:** the owner still sees it; the first screen is gone since TC 54395592 (17:07): check on Safari again.
- [ ] **Card sync verdicts, step 6:** migrate the one `_psd` block after the first live bake or refusal puts
      `db/verdicts.json` and `db/bake-records.json` on data (not there at d4dd2172;
      `~/Documents/Tools/wt/scratch/verdicts/MIGRATION.md`); drop the `_psd` fallback one cycle later.
- [ ] **`+ability`** (PARTS_EDITOR §10 step 6): waits on the owner's Photoshop check of the fork's `import_layer`.
- [ ] The numbered-PNG duplicate scan's deletion now that the 5 PSDs are tagged (PARTS_EDITOR §10 step 5), unless
      `work/partsfollow` did it; the §8 joined-splice deletion after one Card sync cycle with no legacy entries.
- [ ] Deliver the problem-cards CSV from the census re-run (memory `no-guides-csv-only`).
- [ ] Watch the first chained run and the first :17 safety run.

## Owner's open items

- OJWikibot's wiki API answers HTTP 500 since 10-23 (wiki/host side); O-6 waits for it.
- A Photoshop re-save for the 11 badge and 7 apostrophe cards that couldn't be fixed glyph by glyph (session README).
- Layout support still refuses Grimora, Skeleton Mate, Slugcat's spd, three smart-filter star layouts, PoD04 Zebra
  Bloon +1, Living Bread.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers); the team's rulings draft; developers resetting clones after the 10-24 rewrite
  and reinstalling stoogedings.

## Optimization questions answered no / not now

From `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b), not chosen on 10-25: S-5 (CDN caching), S-6 (small site items),
B-4 (pandas out), B-5 (changes() from JSON), P-3 (thumbs off Pages), P-4 (site inside Card sync), P-6 (deploy-pages as
a step), K-2 (runner parallelism), K-4 (smart-filter pair check), K-6 (row-limited decode, dirty box for stars/types,
prange, zlib level), O-7, O-8, O-9. O-2, O-5 and O-6: as `work/ojres` reports (O-6 needs the wiki back).

## Ideas parked / not chosen

Unchanged from `docs/Finished/session-2026-10-24/README.md`.
