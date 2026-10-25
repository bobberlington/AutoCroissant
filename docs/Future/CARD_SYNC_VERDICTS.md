# Card sync verdicts on the data branch, bake and guard events in the changelog

**Status:** built and landed 2026-10-25 (TTSCardMaker a4d137bf, AutoCroissant be39f86; steps 1-5, see "As built"). Left: step 6, the migration of today's `_psd` blocks (one: PoD11ZOMG), after the first live bake or refusal shows `db/verdicts.json` and `db/bake-records.json` on data: runbook `~/Documents/Tools/wt/scratch/verdicts/MIGRATION.md`; then drop the `_psd` fallback one cycle later.

**Goal.** `site/Card Edits.json` on main holds only people's edits; the bake's verdicts live on `data`, so a refusal
makes no main commit. Bakes and guard fixes show on the Changes page.

**Rulings that bind it**
- Verdicts to `db/verdicts.json` on `data`: `{"schema": 1, "cards": {key: verdict}}`, one card per line. A refusal makes
  no main commit; a bake commits to main only the PSD, the PNG and the edit fields it drops as printed (§16.7).
- Events `baked` (fields, PNG changed), `bake-refused` (reason), `guard-png` (a PNG the guards re-made or recompressed),
  `guard-fix` (a fix the guards landed). The Discord post stays one post per push (owner, 10-22).
- The guards' Single-line Composer fix is silent: no event, post or comment (owner, 10-25 ~04:10).
- Discord gets the bake's events (baked, bake-refused) in the push's one post, never the guards': guard-png and
  guard-fix show only on the Changes page and in the commit messages (owner, 10-25 ~08:30). The pixel fixes
  (apostrophes, badges, old glyphs) and PNG re-exports and recompressions are guard events.
- Only main writes wait 30 minutes; the data lane runs at once (owner, 10-25 ~03:20).

## Design

1. **Bake** (`commands/card_bake.py`): the `_psd` blocks it writes today (`verdict`, `entry`, `final()`, `--pending`)
   go to `$RUNNER_TEMP/verdicts.json` instead of Card Edits.json; `final()` and `--pending` read the verdicts from the
   data checkout (`db/verdicts.json`). The main lane's commit M carries only PSD, PNG and dropped fields.
2. **card-sync.yml:** the `main` job uploads `verdicts.json` and `bake-events.json` as an artifact; `after` (group
   `card-data`) runs when the bake baked **or** wrote a verdict, merges both into `data` (`card_data --verdicts`,
   `--events`) in its one data commit. The `data` job keeps its verdicts untouched.
3. **Guards** (`commands/card_guards.py`, `post` job): write `guard-events.json` for re-made and recompressed PNGs and
   landed fixes (Composer fixes excluded); the next data job reads it from the guard run's artifact by push SHA
   (`card_data --guard-events`) and adds the events to that push's changelog group.
4. **Changelog** (`commands/changelog.py` `KINDS`): add the four kinds; each rides its push's post as one line
   ("baked: HP 5 -> 8, PNG re-made"). Changes page: the existing tabs show them (TTSCardMaker `site/assets/changes.js`).
5. **Readers:** `build_site.py` (PSD pending, refused) reads `db/verdicts.json` instead of `_psd`; the bot's `/query`
   facts read the card data's verdicts; card-sync's due gate (`card_bake.due`) likewise.
6. **Migration:** one data commit with today's `_psd` blocks as `db/verdicts.json`, then one main commit removing them
   from Card Edits.json (after the readers above are live). `CardEdits` keeps reading `_psd` for one Card sync cycle,
   then drops it.

## As built (2026-10-25, `work/verdicts`)

1. **Bake.** `card_bake` writes no `_psd`. A run's verdicts and events go to the work dir: `verdicts.json`
   `{"schema", "cards": refusals (no main commit), "landed": the verdicts of the cards in B, None to remove}` and
   `bake-events.json` `{"events": bake-refused, "landed": baked}` (`records()`). A refusal whose verdict equals the data's
   is not recorded again, so a dispatch's re-check tells nothing twice. `--verdicts db/verdicts.json` feeds `final()`,
   `--pending` and `due()`; the entry's old `_psd` is the fallback (step 6's cycle). Entries B rewrites lose their
   `_psd`. A refusal-only run makes no B.
2. **card-sync.yml.** The main job's Bake records step keeps the landed part (and gives the baked events M's sha) only
   when M landed, uploads `bake-records` (a dry run's is `bake-records-dry-run`, never merged); `after` runs when M baked
   or records exist (`AFTER` = M or empty). Every data job (data, after) merges the records no data job merged yet
   (review 2026-10-25: a pending after job in `card-data` is replaced by a newer data job, and its records were lost):
   its Earlier bake records step lists the repo's unexpired `bake-records` artifacts, skips those in `db/bake-records.json`
   (`{"merged": {artifact id: run id}}`), folds at most 20, oldest first (a later verdict wins), and passes `--verdicts`,
   `--events` to the build; a build that exits 0 adds their ids to the file (still listed ones kept), which rides in the
   data commit. A gate refusal, a failed job or a lost one leaves them for the next data job. `card_db.merge_verdicts`
   writes `db/verdicts.json` (one card per line, not in the manifest's hashed files). The main job reads the same
   unmerged records into its data checkout before Pending edits and the bake (Unmerged verdicts, local only), so a
   card refused a run ago is not baked and told again while that run's after job waits.
3. **Guards.** `card_guards.land` lists each included write's event (`events_of`: no event for a Composer-only PSD or a
   Composer-only rewrap PNG); `finish` writes `guard-events.json` for a landed commit; card-guards.yml uploads it as
   `guard-events` before Follow up. Card sync's Guard events step finds each guards commit since the last build, its
   Guarded-push trailer's card-guards run, and that artifact (matched by the commit it names; one not found warns). The
   build adds them by that commit; a guards commit's push is `Card guards`, its `art` events are dropped (its own events
   say them; a Composer switch says nothing).
4. **Changelog.** `KINDS` gains the four; `SILENT` (guard-png, guard-fix) and every event of a `Card guards` push never
   post: a group of them alone is a message without a body (the Send step moves the cursor past it). The Changes page
   shows all four (sections, counts, chips).
5. **Readers.** build_site reads `CARD_DATA_DIR/verdicts.json` over the entries' `_psd` (`add_verdicts`); card_bake as
   in 1. The bot's `/query` reads no verdict today: nothing to change.

Known gaps: two runs' records merged out of order (an older artifact read only after a newer one was merged, after a
failed download) let the older verdict win for a card both touched; a stale verdict counts for nothing in `final()`, so
at worst the card is looked at again. Records older than the artifacts' 14 days are lost (the cron runs a data job every
30 minutes). The owner may still prefer the other fix (the after job in its own group, retried on G6).

Conservative choices (unattended): refusal events are push-less (they post with their build, "Card sync, <time>");
the guards' events are lost to the Changes page when their artifact cannot be found (a warning); a guards commit's other
events (changed) still show on the Changes page but never post.

## Verification (lean)

A dry Card sync dispatch on a refused card (no main commit, a verdict on `data`) and on a baked card (M without `_psd`,
a `baked` event); the migration's two commits diffed by hand. Tests: change the existing `_psd` tests to the file; add
one for the event kinds.
