# Card sync verdicts on the data branch, bake and guard events in the changelog

**Status:** answered by the owner 2026-10-23 ~18:30 (`docs/Campaigns/DATA_BUILD_ACTION.md` §16 item 7); designed
2026-10-25 for the two-lane Card sync; not built. Queue item 6; best after step 10 (`work/pickles`) lands, since both touch `card_data`.

**Goal.** `site/Card Edits.json` on main holds only people's edits; the bake's verdicts live on `data`, so a refusal
makes no main commit. Bakes and guard fixes show on the Changes page.

**Rulings that bind it**
- Verdicts to `db/verdicts.json` on `data`: `{"schema": 1, "cards": {key: verdict}}`, one card per line. A refusal makes
  no main commit; a bake commits to main only the PSD, the PNG and the edit fields it drops as printed (§16.7).
- Events `baked` (fields, PNG changed), `bake-refused` (reason), `guard-png` (a PNG the guards re-made or recompressed),
  `guard-fix` (a fix the guards landed). The Discord post stays one post per push (owner, 10-22).
- The guards' Single-line Composer fix is silent: no event, post or comment (owner, 10-25 ~04:10).
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

## Verification (lean)

A dry Card sync dispatch on a refused card (no main commit, a verdict on `data`) and on a baked card (M without `_psd`,
a `baked` event); the migration's two commits diffed by hand. Tests: change the existing `_psd` tests to the file; add
one for the event kinds.
