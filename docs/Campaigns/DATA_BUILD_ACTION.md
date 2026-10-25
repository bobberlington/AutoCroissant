# Data build Action: the card data built on GitHub (2026-10-22)

**Status:** designed 2026-10-22, revised the same day for the owner's later decisions (below) and again for what was built
(the status note below; `docs/NEXT_SESSION_PLAN.md` tracks the landing). The owner's goal, 2026-10-22: "let's work on the
data build action". The
website and the TTS save must not depend on the jetson; a GitHub Action builds the card data. This design also covers:
- the changelog in Discord (one post per push) and on the website (a Changelog tab and a Database edits tab on the Changes
  page);
- balance analytics (the Stats page and each card's Details);
- `/query as_of:<date>`;
- old_stats refinements (version dates from git, no duplicate versions);
- the archive of deleted cards: **merged and deployed** (AutoCroissant main dfdf953; step 2 is done).

**The owner's decisions after the first draft (2026-10-22).**
- The changelog posts **one post per push**, not a daily digest. Failure alerts stay in the same channel.
- **Batching:** "since the data build makes new commits, let's just update the save on pushes too - but could we batch edits
  together for data builds and the save, perhaps every 30 minutes or an hour, to reduce commit churn". So one scheduled run,
  **Card sync**, every 30 minutes (hourly if the owner prefers, decision 5) does everything in order: bake the pending website
  edits, build the data, update the TTS save, post the changelog. It makes at most one commit on main and one on `data`.
  It replaces the per-push build trigger and the hourly catch-up of the first draft, and it absorbs `tts-save.yml` and
  `card-edits.yml`.
- **The PSD hold is lifted** (2026-10-22 ~02:40), so the bake runs live in Card sync after a fresh dry run (step 8).

Numbers were measured read-only on 2026-10-22 at AutoCroissant main cae1f2f (last PICKLE 3888392, 00:23) and TTSCardMaker main
5bdda4ad, with push counts re-measured at e0e26b2f (§2). Anything not checked is marked "to confirm". They predate the build:
what is built is in the status note below and §14.

**The choice.** Two designs were judged, "minimal" (the Action pushes PICKLE commits to AutoCroissant) and "data-with-cards"
(the data lives in TTSCardMaker). The robustness judge scored data-with-cards 8.4 and minimal 6.4. The simplicity judge scored
them 5.8 and 5.6 and asked for a hybrid. **data-with-cards wins**: it is the only design with no cross-repo credential and
exactly one writer. Grafted onto it:
- minimal's reuse of the bot's own traverser code, moved rather than rewritten;
- minimal's cutover discipline (parity dry runs with predicted numbers, the jetson reconcile, the Discord-side writers);
- the hybrid's order, which ships value before the move.

§3 lists every graft.

**Status 2026-10-23.** Steps 1-6 and 9 are on main and deployed: the data build, the Discord writers and the alias
migration landed with AutoCroissant main 8861696 (2026-10-22 20:50) and TTSCardMaker c0b6f2d7; the bot's cutover side
(work/cutover-bot, step 8's code with `CARD_SYNC = False`) merged as 7b99438 and deployed with main 2ada357 on 2026-10-23
10:36. **Step 7 started:** TTSCardMaker main a7d1fe56 turned on card-sync.yml's schedule (cron `4,34 * * * *`, `MAIN_COMMIT`
and `POSTS` off). At 11:39 no scheduled run had fired (GitHub's scheduler ran 3.4-5.6 h late for this repo on 2026-10-22) and the card repo had no `data` branch, so the first scheduled run
creates it from AutoCroissant's latest PICKLE: check it is one pushed after the history-recovery run (step 7's
precondition) and re-seed if not. The shadow runs to 2026-10-26 10:36 or 10 card pushes, whichever is later; then step 8
(the owner swaps the reminders; `ACTIONS_TOKEN` is in the jetson's config.py since 2026-10-22). The open list is
`docs/NEXT_SESSION_PLAN.md`.

**Status 2026-10-22 (as built; superseded by the note above).**
- **Steps 1-4 done and gated.** The builder is AutoCroissant work/data-build (ff47ad6). It passed both parity gates: legacy
  mode writes byte-identical pickles, and a full re-parse on the Mac shows 0 data-field differences.
- **Steps 5, 6 and 9 built and reviewed, not landed.** AutoCroissant work/discord-writers (on work/data-build): /update_metadata,
  /mass_replace and /alias commit to the card repo. TTSCardMaker work/alias-migration (the 60 aliases into `Card Edits.json`
  `aliases` plus `aliases_moved`; build_site honours it), work/card-sync (card-sync.yml, dispatch only, `dry_run` true by
  default; §6 describes it as built) and work/changes-tabs (the Changelog and Database edits tabs).
- **Step 7, the shadow, is next.** Step 8, the cutover, waits on `ACTIONS_TOKEN` in the jetson's config.py and the owner's
  reminder swap. Step 10 follows within a day of clean runs.
- **Step 8's bot side built, switched off** (AutoCroissant work/cutover-bot, on work/history-fixes): the `card_db.sync()`
  reader, /update_stats as a sync plus Card sync dispatches, the /push reply, /pull, /update, /fonts from the data and
  admin-only `force_update`/`cards`, all behind one constant, `CARD_SYNC` in global_config.py (§11, "As built"). With it off
  the bot is byte-identical to before; the cutover commit is that constant set to `True`.
- **Owner decision: 624 database-only series values.** They were typed in Discord (`/update_metadata series`, `/mass_replace
  series`, 2025-10-29 to 2026-06-20; 86 distinct values) and no card-repo file holds them. The parser keeps them from the
  database, so no re-parse loses them, but a build from scratch without a seed would. 28 sit under a `Card Edits.json` series
  edit. work/alias-migration carries the other 596 into `Card Edits.json` (§14 step 5); the owner confirms before it lands
  (§16, decision 2).

## 1. Goal and non-goals

**Goal.** One scheduled GitHub Action in TTSCardMaker, Card sync, is the only writer of the card database and the TTS save.
Every 30 minutes, when its card inputs changed since its last successful run, it bakes, builds, updates the save and posts,
and commits plain JSON to a data branch of the same repo. Everything else only reads that data: the website, the TTS save
(from the run's own data), the bot, the tests and the diagnostics. The jetson can be off for a week and the website, the TTS
save and the changelog keep working.

**Non-goals.**
- No change to how cards are parsed: same `psd_parser`, same psd-tools pin, same validator.
- No new PSD or PNG writer: the bake (`card_bake`, the edit pipeline) stays the only one. Card sync runs it; the build
  itself writes no card files.
- No server and no new secret outside TTSCardMaker.
- No second build path left behind: the traverser's data code moves out of `repo_traverser.py`, it is not copied.
- No AI.

## 2. What was measured (2026-10-22)

| Fact | Value |
|---|---|
| stats.pkl at 3888392 | 819 entries (6 duplicates), every one at `parser_version` 2 |
| old_stats.pkl | 363 names, 469 versions; 291 versions have no `blob_sha`; 1 name not in stats: `ZOEY!!!!!` |
| Duplicate versions (all data fields and stats equal to the next version or the current card) | 10, re-counted with the dedupe code itself (step 4): 4 differ only in timestamp (Monkey D Vour, The Lich King v0, Anubisath Guardian, Qiraji Soldier), 6 only in blob_sha or parser_version (or with the timestamp); the first count here said 8 |
| aliases.pkl | 60 aliases |
| psd_details.json | 813 records, 0 with a null `supplied` flag (the jetson's fonts cache, B11, filled them 2026-10-21) |
| Blobless bare clone of TTSCardMaker | 1.1 MB, 615 commits, 2 s |
| `git log --raw --no-renames -- '*.psd'` on it | 0.07 s, no blob fetched (`GIT_NO_LAZY_FETCH=1`) |
| Tree at HEAD | 904 PSDs, 1003 PNGs, no upper-case extensions |
| Stored timestamps vs the newest commit at the card's path | equal for 812 of 813 non-duplicate cards |
| Re-dating to "the commit that first added this blob" | 31 of the 813 non-duplicate cards change date (their duplicates follow); 3 of the 178 versions that have a blob |
| ZOEY!!!!! deletion | `Creatures/Other/5 Stars/ZOEY!!!!!.psd`, deleted 2026-06-18 in 35adb383 by Zomoroko |
| JSON, one record per line (psd_details' style) | stats.json 604 KB (114 KB gzipped); old_stats.json 268 KB (47 KB) |
| Commits per day, Oct 20 / 21 / 22 | all: 21 / 47 / 28; outside TTS/: 20 / 41 / 17; card inputs (PSD, PNG, Types/, Card Edits.json, Card Authors.txt, manifest): 2 / 8 / 1 |
| Pushes to main per day, Oct 20 / 21 / 22 (activity API, at e0e26b2f) | 10 / 20 / 22 |
| Card-input commits, including `TTS/decks.json`, Oct 20 / 21 / 22 | 2 / 8 / 4, in 2 / 5 / 2 half-hour windows; PNG commits in 0 / 2 / 1 half-hour windows |
| Bot commits on main (AutoCroissant, github-actions), Oct 20 / 21 / 22 | 1 / 1 / 4 |
| `GET repos/…/activity?ref=refs/heads/main&activity_type=push` | one call lists the last 100 pushes with actor, `before` and `after`; it includes the bot's API commits and GITHUB_TOKEN pushes |
| TTSCardMaker collaborators with push | 8; rulesets 0; branches: main only |
| Creatures and minions with all four stats | 387 (creature 2★ 45, 3★ 102, 4★ 93, 5★ 56; minion 1★ 15, 2★ 31, 3★ 27, 4★ 11) |

## 3. Grafts

| Idea | From | Where |
|---|---|---|
| Data in TTSCardMaker, pushed with that repo's own GITHUB_TOKEN; no PAT, no deploy key | data-with-cards | §5, §9 |
| A local `git ls-tree` and `git log --raw` instead of the tree and commits APIs | both | §7.2 |
| Build main's head and diff against the stored blobs, never against a push's `before` | data-with-cards | §6.2 |
| A request budget with continuation for full re-parses | data-with-cards | §7.3 |
| Gate G4 (mass re-parses of unchanged blobs need approval), the `restore` input, the manifest | data-with-cards | §8, §13 |
| A version only when data changes; art-only changes become `art` events; from/to dates and a cause | data-with-cards | §7.4 |
| The traverser's own code (moved into a Discord-free module), the three hooks (git dates, `last_run`-style counters, per-PSD failure isolation) | minimal | §7 |
| Per-PSD failure isolation landed on main first, so the jetson stops aborting whole runs | minimal (judge 1) | step 1 |
| A dry run with predicted numbers, the jetson reconcile | minimal | §14 |
| /update_metadata and /mass_replace write `site/Card Edits.json` (with `Edit-by`, `_by`/`_at`) and apply the edit in memory at once | minimal | §11 |
| A changelog `seq` that never reuses a number | minimal | §8 |
| An immediate failure alert; a staleness warning; a pinnable builder ref | judge 1 | §6.3, §10, §6.4 |
| Analytics and `/query as_of` ship first, from today's data | judge 2 | step 3 |
| The site keeps its own push trigger, so website edits and new art publish as fast as today | judge 2 (from minimal) | §10 |
| One scheduled run every 30 minutes with a change gate; bake, build, save and post in one run; at most one main commit | the owner, 2026-10-22 | §6 |
| One changelog post per push | the owner, 2026-10-22 | §12.1 |

**Not taken.**
- *Keep writing pickles in the first phase* (judge 2). Writing a pickle needs `commands.psd_analyzer` importable, which means
  discord and pandas on the runner, or a fake module. Instead the JSON is proven against the pickles twice: a round trip of all
  819 cards and 469 versions, and the site's `cards.json` built both ways must be byte-identical (step 7). The pickles stay
  reachable for rollback through `python -m commands.card_db export-pickles`.
- *A jetson fallback writer* (minimal's `local:True`). That would put two parsers (Cython and pure Python) on the same files.
  The fallback is the same builder run by hand on the Mac (§13).
- *A second clone of AutoCroissant on the jetson* (minimal). The bot reads two to four files over HTTPS (§11).
- *A build on every push* (the first draft). The owner chose batching to cut commit churn; a card push now waits up to one
  interval (§15).

## 4. Architecture

```
 people: the website's Edit button | Discord (/alias, /update_metadata, /mass_replace) | git pushes (PSDs, PNGs, decks.json)
        |  commits                         |  commits (git_commit: API, else SSH)          |
        v                                  v                                               v
 MichaelJSr/TTSCardMaker  main    PSDs, PNGs, Types/, MDW/Fonts/, stoogedings.json, site/Card Edits.json,
        |                         site/Card Authors.txt, TTS/decks.json, TTS/Impossibility Simulator.json
        |
        |  every 30 minutes (decision 5) | dispatch: the bot, by hand
        v
 .github/workflows/card-sync.yml  "Card sync"                       runner, GITHUB_TOKEN only, one job
   gate   main head, builder, pending work vs db/manifest.json -> go, or end in ~15 s
   bake   pending website edits (card_bake) -> local commit B on main's head, nothing pushed yet
   build  B (or main's head), blobless + sparse -> git ls-tree, git log --raw   (tree, dates, authors, deletions, renames)
          data, depth 1 -> db/ before
          bobberlington/AutoCroissant @ builder ref, sparse card_data.SPARSE_FILES
          changed PSD blobs by sha (local first, else REST, budgeted) -> psd_parser, psd_details
          history pass, changelog events, gates G1-G6
   save   tts_save on B with names from the new db/
   main   one commit M = B + the save, fast-forward push to main (only when the bake or the save changed something)
   post   one Discord message per push with events (DISCORD_CHANGELOG_WEBHOOK)
   data   one commit: db/ and the posted cursor -> branch data (never forced);  gh workflow run site.yml
        |
        v
 MichaelJSr/TTSCardMaker  data    orphan branch, db/ only, at most one commit per run
        |                              |                                       |
        | checkout, depth 1            | 304 check, then raw files at the sha  | the next run's "before"
        v                              v
 site.yml -> GitHub Pages             the bot on the jetson: reads JSON, never writes card data,
   cards.json, Changes, Stats,        dispatches runs (ACTIONS_TOKEN), commits Discord edits to main
   Details, analytics.json
```

**Who writes what afterwards**

| Data | Only writer afterwards | Today |
|---|---|---|
| `db/stats.json`, `db/old_stats.json`, `db/psd_details.json`, `db/manifest.json`, `db/changelog/*.jsonl` (branch `data`) | card-sync.yml, stage `build` | /update_stats on the jetson, then /push to AutoCroissant main |
| `db/changelog/posted.json` | card-sync.yml, stage `post` | none |
| `TTS/Impossibility Simulator.json` | card-sync.yml, stage `save`; `/update_tts_save` as the hand fallback | tts-save.yml (daily), `/update_tts_save` |
| Baked PSDs and PNGs, the bake's verdicts in `Card Edits.json` | card-sync.yml, stage `bake`; `/bake_cards` as the hand fallback | card-edits.yml (dispatch, dry run), `/bake_cards` |
| `site/Card Edits.json` | the website, the bake (verdicts), by hand, **and the Discord commands** | the website, the bake, by hand |
| `site/Card Authors.txt` | by hand, **and `/update_metadata author:` and `/mass_replace author`** | by hand |
| Aliases | `Card Edits.json` field `aliases` (decision 3); `/alias` commits there | aliases.pkl via /alias and /push |
| AutoCroissant `stats.pkl`, `old_stats.pkl`, `aliases.pkl`, `psd_details.json` | nobody after the cutover (frozen; deleted at step 10, decision 6) | the jetson |
| The bot's `card_data/` (a local copy of db/), `reminder.json`, `type_emojis.json`, `repo_history.json`, `fonts_cache/` | the bot (all gitignored) | the same, without `card_data/` |

## 5. Where the data lives (decision 1)

**Recommended: an orphan branch `data` in MichaelJSr/TTSCardMaker holding only `db/`.**
- **Main stays for people.** A data commit never lands on main. Main gets a Card sync commit only when a bake or the save
  changed something; a PSD-only push, an edit that changes no name and a re-parse make no main commit at all.
- **Its history is the database's history.** There is at most one commit per run, the per-card diffs are readable, a rollback
  is one commit, and a reader fetches the branch alone at depth 1.
- **No loops.** A GITHUB_TOKEN push triggers no workflow, and every workflow listens to `main` only. build_site's
  `git_history()` and the save's change check read main only, so data commits never count as contributions.
- **The cost is small.** `fetch-depth: 0` checkouts in site.yml also fetch the branch's commits and trees: a few KB a day.

**Conflict with a memory rule.** The rule "commit to main; delete spare branches" applies. `data` is machine-written state, not a
work branch, so the owner must approve the exception.

**Fallback: `data/db/` on main.** Batching makes this cheaper than in the first draft: the data rides in the run's one main
commit, so there is no extra writer and no extra "pull first" moment. The changes:
- the main commit M also holds `data/db/`, so a run commits to main whenever the data changed, not only for a bake or the save
  (recently 2 / 5 / 2 runs a day instead of about 0 / 2 / 1, §2);
- the change gate ignores `data/**`, and `card_commit` names M's parent plus the run's own bake;
- card makers' pulls carry the JSON deltas (a few KB to tens of KB per commit).

**Who can write `data`.** All 8 collaborators can push to any branch, and there are no rulesets today. Two consequences:
- The bot and the site read only JSON from `data`, never a pickle. A forged commit can at worst forge card data, which the
  same people can already do through `Card Edits.json`. It cannot run code on the jetson.
- A ruleset that lets only GitHub Actions update `data` is to confirm: it depends on whether a personal repo's ruleset can list
  Actions as a bypass actor.

## 6. The workflow: TTSCardMaker `.github/workflows/card-sync.yml` ("Card sync")

It replaces `tts-save.yml` and `card-edits.yml`, which are deleted at the cutover (step 8).

### 6.1 Triggers

```yaml
run-name: "Card sync", or "Card sync (dry run)" for a dry dispatch   # the alert's streak check skips dry runs
on:
  schedule:                  # added at step 7; as built (step 6) the workflow is dispatch only
    - cron: "4,34 * * * *"   # every 30 minutes; "4 * * * *" for hourly (decision 5)
  workflow_dispatch:
    inputs:                  # all optional; all 10 of GitHub's 10
      force: boolean         # run every stage even when the gate finds nothing
      cards: string          # re-parse these cards (comma separated, matched like Card Authors.txt)
      full: boolean          # re-parse every card; also the approval gate G4 asks for
      dry_run: boolean       # run and report, commit and post nothing (default true until step 8, then false)
      bake: string           # "all" (default), one card's name (card-edits.yml's `card`), or "none"
      allow_archive: boolean # lift gate G2 for this run
      restore: string        # a data commit: write its db/ back as a new commit
      seed: string           # an AutoCroissant commit: start from its pickles (first run; re-seed after a rollback)
      builder_ref: string    # AutoCroissant ref for this run (else vars.CARD_DATA_BUILDER_REF, else main)
      retry: string          # set by Card sync itself when main moved: the rejected run's event (§6.3, stage 4)
env: {POSTS: "off", MAIN_COMMIT: "off"}            # step 8 turns both on (§14)
permissions: {contents: write, actions: write}     # push to main and data; gh workflow run site.yml; gh run list
concurrency: {group: card-sync, cancel-in-progress: false}
```

There is no push trigger. Minutes 4 and 34 avoid the top of the hour, when GitHub delays scheduled runs most.
- A dispatch runs dry unless `dry_run` is off; a scheduled run is live. Two file-level switches say what a live run writes:
  `MAIN_COMMIT` (the bake and the main commit M; while off, only a dry run bakes) and `POSTS` (sending to Discord; while off,
  `discord.json` goes to the artifact only). With both off, a live run writes only `data`.
- No input slot is left: a new option must replace one or ride in an existing one.

### 6.2 Batching and the change gate

- **The schedule is the batch.** Every push in the interval between two runs is handled by the next run, together. A run
  builds the head of main at its start, not `github.sha`, and diffs against what was last built (the blob shas stored in db/
  and `manifest.card_commit`). No push is missed and none is built twice.
- **Concurrency** holds one run going and one pending. A long run (a budgeted full re-parse) delays the next scheduled one; a
  newer pending run replaces the older one; nothing is cancelled midway.
- **The gate** (the job's first step, about 15 s) takes a blobless clone of main (1.1 MB) and reads `db/manifest.json` from
  `data` (one contents read). The run goes on when any of these holds, else it ends green:
  - (a) `git diff --quiet <manifest.card_commit> <main> -- <card inputs>` finds a change. The card inputs are `*.psd`, `*.png`,
    `Types/`, `MDW/Fonts/*`, `stoogedings.json`, `site/Card Edits.json`, `site/Card Authors.txt`, `TTS/decks.json` and the save
    file. `card_commit` only advances on a successful run, so a failed run is retried by the next one;
  - (b) `manifest.pending.stale` is not empty: a budgeted re-parse is still continuing;
  - (c) `PARSER_VERSION` at the builder ref differs from `manifest.builder.parser_version`. That version is one line of
    `commands/psd_parser.py`, read raw from AutoCroissant at no API cost, so a parser change shows up within one interval (gate
    G4 stops it until it is approved);
  - (d) `manifest.pending.repin` is not empty: images baked by the last run still need their save links (§6.3, stage 4);
  - (e) events are unposted: the last event's `seq` is past `posted.seq` (a post failed);
  - (f) a dispatch with `force`, `cards`, `full`, `restore` or `seed`.
- **Failed cards are not retried every run.** A broken PSD stays broken until someone pushes a fix, and that push changes its
  blob, so the build tries it again then; `cards` retries it by hand.
- No `sleep`, no polling.

### 6.3 The job `sync`

One job (ubuntu-latest; timeout 90 min). Each stage after the gate runs only when the gate said go.

**Setup.**
1. **Check out main:** `fetch-depth: 0`, `filter: blob:none`, sparse `/site/Card Edits.json`, `/site/Card Authors.txt`,
   `/MDW/Fonts/stoogedings-src/stoogedings.json` and `/TTS/`.
2. **Check out the data:** `ref: data`, `path: carddata`, `fetch-depth: 1`. On the first run, with no branch yet, the build
   seeds instead (§7.1).
3. **Check out the builder:** `bobberlington/AutoCroissant` at `inputs.builder_ref || vars.CARD_DATA_BUILDER_REF || 'main'`,
   `path: bot`, sparse to `card_data.SPARSE_FILES` (§7), which covers the bake and the save as well. `tests/test_card_data.py`
   runs `--help` on the three CLIs with exactly that list, as test_card_check does for card-check.yml.
4. **Python:** setup-python 3.14, the Mac test venv's and the jetson's version (the goldens pass there), and since the
   owner's ruling of 2026-10-24 every Action's. **One environment** (`$RUNNER_TEMP/venv`, the step "Python environment")
   for every stage: the bake, the build, the save and the posts. Until 2026-10-24 the bake had its own on 3.12
   (card-edits.yml's then) beside the builder's 3.14, with a cache key naming both Pythons. It installs
   `bot/requirements-cards.txt` (the psd-tools fork at the one tag, read there; fonttools, requests, pyoxipng, built from
   its sdist on Linux x86_64 3.14) offline from the card workflows' one wheel cache entry (the file's hash, the runner, the
   Python), built on a miss and saved once installed. `--help` of card_data, tts_save and card_bake proves the import
   closure before any stage. TTSCardMaker `TTS/checks/card_sync_steps.py` runs the steps with that venv's path.
5. **Fonts:** the cache key is `card-fonts-$(git rev-parse HEAD:MDW/Fonts)`, the key card-edits.yml uses today, so the
   entries carry over. On a miss, run `git sparse-checkout add '/MDW/Fonts/*' '!/MDW/Fonts/stoogedings-src/'`. This step runs
   without `GIT_NO_LAZY_FETCH`.

**Stages, in order.**
1. **Bake** (unless `bake: none`). `python -m commands.card_bake --repo "$GITHUB_REPOSITORY" --head <main> --local "$GITHUB_WORKSPACE"
   [--card <name>] [--dry-run]`. The new `--local` mode writes every baked PSD and PNG and the rewritten `Card Edits.json`
   (verdicts) into the clone's object store as one local commit B on main's head, and pushes nothing. The bake's checks are
   unchanged. A scheduled run (or its retry) bakes with `--pending`. A card's error fails the step but keeps B for the cards
   that baked (card-edits.yml's per-card rule); with no B the run goes on from main's head, and the report says so.
   Since 2026-10-23 (owner) a pixel miss is no refusal: a box past the parity bar bakes, and B's card line and the report
   say `name baked with a pixel difference: bad 1.97%, mean 1.60, max 217`; only past 25% bad, when the render shows no
   change, or when the re-parse does not read the edit back, is a card refused (EDIT_PIPELINE_DESIGN.md §5 step 3). M's
   body carries those card lines.
2. **Build** on B, or main's head without a bake, with env `GIT_NO_LAZY_FETCH=1`, `GITHUB_TOKEN` and
   `PSD_TOOLS_FONT_DIRS=$GITHUB_WORKSPACE/MDW/Fonts`:
   ```
   cd bot && python -m commands.card_data --repo "$GITHUB_REPOSITORY" --checkout "$GITHUB_WORKSPACE" --head <B or main> \
     --db "$GITHUB_WORKSPACE/carddata/db" --budget 400 --run "$RUN_URL" --summary "$GITHUB_STEP_SUMMARY" \
     --message "$RUNNER_TEMP/message.txt" [--full] [--cards "$CARDS"] [--allow-archive] [--restore SHA] [--seed SHA]
   ```
   - The baked blobs are in the local object store, so they cost no API call.
   - **Exit codes:** 0 = written, or nothing to do; 2 = a gate refused (red, nothing committed); 1 = an error.
   - **Job summary:** the counts, a diff_stats-style per-card diff, and the validator notes on re-parsed cards.
3. **Save.** `python -m commands.tts_save "TTS/Impossibility Simulator.json" --names carddata/db/stats.json`, with
   `AUTOCROISSANT_LOCAL_DIR` at the checkout on B, as tts-save.yml runs it today. The new `--names` reads the card names from
   the run's own data (`Card Edits.json` first, as today) instead of the live site's `cards.json`, which is one run behind.
   - **The run's own baked images keep their old links** in this save: a save cannot pin a link to the commit it is in. The
     build lists them in `manifest.pending.repin`, and the next run pins them (gate d). Those cards show their old art in the
     save for one interval.
   - If the build failed, the save still runs, with names from the last good data.
4. **Commit main** (skipped on `dry_run`, or when neither the bake nor the save changed anything). One commit M on main's head
   with the bake's files and the save, as `AutoCroissant <autocroissant@gmail.com>` (git_commit's identity), then
   `git push origin M:main`: fast-forward only, never forced.
   - Subject: `Card sync: 2 cards baked, TTS save updated`. Body: the bake report's card lines and the save's message.
   - M holds the same card files as B, so the builder's references to B (version dates, events) are rewritten to M before
     the data is written.
   - **A save-only M** (no bake) sets `manifest.main_commit` to M, and `card_commit` too when it was main's head.
   - **Main moved during the run** (the push is rejected): M, the posts and the data are all dropped; `card_commit` and
     `pending.repin` do not advance, so the next run's gate finds the work again. If `git push` failed but `ls-remote`
     shows main at M, M counts as pushed.
     - The run then dispatches one retry of itself on the new head: `dry_run=false`, `force`, `retry=<its event>`, the
       builder's sha as `builder_ref`, and its `bake`, `cards`, `seed` and `restore`. It never carries `full` or
       `allow_archive`, which approved the old head only (a warning says so).
     - It dispatches nothing when a Card sync run is already queued or pending (`gh run list`), since that run does the work
       and a new dispatch would replace it. A run with `retry` set only warns on a second rejection.
   - **Deadline.** M is pushed only before minute 60 of the job's 90. The 30-minute reserve is the sum of the caps of this
     step (10) and every later one (posts, send and data commit 3 each; artifact 2 + 5; site build 2; alert 2), and a check
     in `TTS/checks/card_sync_steps.py` holds it there. Past minute 60 the step fails red, so M and the data are skipped
     together and the next run does both.
5. **Post** (unless `dry_run`; sending also needs `POSTS` on, a successful build and an M that was not rejected, so the
   cursor never moves without its data commit). `python -m commands.changelog post --db carddata/db --site
   https://michaeljsr.github.io/TTSCardMaker --out discord.json`, then curl, following card-check's pattern. One message per push
   with events, oldest first (§12.1). The card ids come from the live site's `cards.json`. A missing secret or a failed call is
   a `::warning::`; the URL is never printed. `posted.json` advances after each accepted message.
6. **Commit data** (skipped on `dry_run`, or when db/ and the cursor are unchanged), in `carddata/`:
   - `git add -A db`, committed as `AutoCroissant <autocroissant@gmail.com>` with `-F message.txt`, then
     `git push origin HEAD:data`, never forced. A rejected push fails the run, and the next run rebuilds on the new tip.
   - Subject: `Card data: 2 changed, 1 new, 1 archived (main 1a2b3c4d)`.
   - Body: one line per card, then `Builder: bobberlington/AutoCroissant@<sha>` and `Build: <run url>`.
   - The cursor is in the same commit, so a run makes one data commit at most. The cost: if this push fails after the posts
     went out, the next run posts the same pushes again (only possible when another writer moved `data`, gate G6).
   The manifest is rewritten only when the build actually changed `db/`, so an unchanged build makes no data commit.
7. **Upload an artifact** (whenever the gate says go, live runs included, since the shadow reads `discord.json` from it):
   `db/`, the bake's crops and report, the save report, the diff, the posts and the commit messages. On a gate refusal
   (exit 2) the builder stops before writing, so the artifact holds the previous `db/` and the summary, not the would-be
   `db/` (a G3 refusal found after writing is the exception); that needs a builder change.
8. **Dispatch the site build** (`if: !cancelled()`): `gh workflow run site.yml` after M was pushed, or after a data commit
   with `MAIN_COMMIT` on (in the shadow the site does not read the data yet).
9. **Failure alert** (`if: failure()`, `POSTS` on, not a dry run): one Discord message to `DISCORD_CHANGELOG_WEBHOOK`, for
   example "Card sync failed (build): <run>; the website and the bot keep the last good data". It names the stages that
   failed. Any failed live run alerts (scheduled, a retry, the bot's or the owner's dispatch). It is sent only when the
   previous completed live run did not fail; dry runs, cancelled and skipped runs are left out of that streak (by run name).

### 6.4 The builder ref

The builder follows AutoCroissant main by default. Gate G4 stops a parser change from rewriting the database unannounced (§8).
To roll back a broken parser push without touching AutoCroissant, set the repo variable `CARD_DATA_BUILDER_REF` (Settings →
Variables; not a secret) to the last good commit, then clear it once main is fixed. `manifest.builder.commit` records the exact
commit of every build. The pin also pins the bake and the save, which come from the same checkout. A ref from before
requirements-cards.txt (2026-10-23) fails at the Python environment step: the workflow installs that file.

## 7. The builder (AutoCroissant, Discord-free)

**Modules (all new unless noted; none join setup.py `INCLUDE_FILES`, so the jetson runs the same pure Python as the runner):**

| Path | What |
|---|---|
| `commands/card_data.py` | The build and its CLI (`python -m commands.card_data`). The traverser's data path is **moved** here from `repo_traverser.py`: the changed-PSD loop, `_details`, authors, edits, duplicates and archive. It reads a **source**: a tree (in the API's shape), a `text(path)` and a `download(path, sha)`. It also holds the gates and `SPARSE_FILES`. |
| `commands/card_history.py` | The git index from one `git log --raw`, the version rules, re-dating, dedupe, archive dates and renames |
| `commands/card_db.py` | JSON records (`record(card)`, `card(record)`), reading and writing `db/` files atomically one record per line, the manifest with sha256 per file, the one-time seed from pickles (an allow-list unpickler mapping `commands.psd_analyzer`/`commands.card_models` CardInfo and CardStats to card_models), `export-pickles`, and the bot's `sync()` |
| `commands/changelog.py` | Events from before/after plus the index, each with its push; the CLI `post` (Discord payloads, one per push; the cursor) |
| `commands/card_bake.py` (changed) | Gains `--local`: all baked cards and the verdicts as one local commit, nothing pushed (§6.3, stage 1). The bot's `/bake_cards` keeps its one commit per run. |
| `commands/tts_save.py` (changed) | Gains `--names <stats.json>`: names from a `db/` instead of the website's `cards.json`, with the same rule build_site uses for `cards.json` names |
| `commands/repo_traverser.py` (changed) | Becomes the bot's /update_stats. Until step 8 it calls `card_data.build` with the GitHub-API source and writes the root pickles as today (legacy mode, for the jetson). From step 8 it is sync and dispatch (§11). |
| `commands/card_models.py` (changed) | Gains `version` beside `archived` (plain builtins). As built (step 4) it is a class attribute `version = None`, not a dataclass field, so the pickles stay byte-identical; dataclass `fields()`, `asdict`, `eq` and `repr` skip it, and card_db carries it through `EXTRA_KEYS`. Nothing else changes. `created` (2026-10-23) is a second such attribute: `site/Card Creation Dates.json`'s `{date, from[, provisional]}` (or the card's first upload, step 7), no data field (no version, no event). |
| `requirements-cards.txt` | The card code's packages, the bot's and every card Action's (`requirements-card-data.txt` until 2026-10-23, stock `psd-tools==1.25.0`): the psd-tools fork's one pin, `fonttools`, `requests`, `pyoxipng` on Linux x86_64 at any Python (3.14 built from its sdist, cached) and where it has a wheel up to 3.13, so not on the jetson or the Mac (3.14) |

`SPARSE_FILES`: card_data.py, card_history.py, card_db.py, changelog.py, card_models.py, card_edits.py, psd_parser.py,
psd_details.py, psd_regions.py, font_check.py, github_api.py, global_config.py, requirements-cards.txt, and for the other
two stages card_bake.py, psd_edit.py, git_commit.py, repo_history.py and tts_save.py (today's card-edits.yml and tts-save.yml
lists). The list never includes psd_analyzer, query_card, utils or type_emojis (discord, pandas). There is no config.py:
github_api reads it inside a try/except and takes the token from the environment.

### 7.1 One build, in order

1. **Before.** Read `db/*.json` through card_db. When `db/` is empty, or `seed` is given, read AutoCroissant's pickles at that
   commit through the allow-list unpickler and record `manifest.seed`.
2. **The tree.** `git ls-tree -r -z <head>` (B or main's head) is shaped like the API tree (path, sha, mode, type), so every
   traverser helper and `psd_details.font_index` work unchanged. It is never truncated and costs no API call.
   **Every blob is read by its sha:** from a local object store when it is there, else by REST.
   - Local means `git cat-file` under `GIT_NO_LAZY_FETCH=1`, so a missing blob is never downloaded by git.
   - On the runner, the sparse checkout supplies the small files (`Card Edits.json`, `Card Authors.txt`, `Card Creation Dates.json`, `stoogedings.json`),
     and the bake supplies its own PSDs.
   - On the Mac, `--blobs` supplies every PSD as well.
3. **The git index** (card_history). One `git log <head> --format='C%x09%H%x09%ct%x09%an' --raw --no-abbrev --no-renames --
   '*.psd' '*.png'`, under `GIT_NO_LAZY_FETCH=1`, so a blob fetch fails loudly instead of downloading. It took 0.07 s on
   2026-10-22 (§2). It gives:
   - per path: the newest commit, the first author, the last `D` (its deletion) and the PNG commits;
   - per blob: its intro, the oldest commit that added it at any path;
   - exact renames: a `D` of A and an `A` of B with the same blob in one commit. This is build_site's `image_versions` rule and
     needs no git rename machinery.
   `--no-renames` and no `-M`: inexact rename detection reads blobs.
4. **The changed set.** The traverser's rules, unchanged: new, moved, an older `parser_version`, a different blob. Add the
   names in `cards`, or everything with `full`. Archived names whose PSD is back are restored (`StatsDatabase.restore()`).
5. **Downloads.**
   - Changed PSD blobs by sha (`github_api.fetch_blob`) over 4 streams, after the local object store.
   - The run stops at `--budget` (400) downloads, or when `X-RateLimit-Remaining` drops below 150. The rest go to
     `pending.stale`.
   - Hand run on the Mac: the blobs come from `--blobs`, so no API is used.
6. **Parse and details,** with hook (c): an exception in `parse` or `_gather_details` for one PSD counts that card as failed
   (`pending.failed`), keeps its old data, and the run goes on.
7. **Authors, edits, duplicates.** As today, except one input: `_duplicate_name` reads the duplicate's `Card Edits.json`
   aliases. Until step 5 it reads aliases.pkl from AutoCroissant main, only while `Card Edits.json` has no
   `aliases_moved` (2026-10-23; read before the PSDs, so a missing file stops the build early), so deleting the file at
   step 10 breaks no build. Then every card's `created` from
   `site/Card Creation Dates.json` (`CardDates`, 2026-10-23; written by `python -m commands.discord_edits created` from
   the Discord archive), by the website's rule (build_site.py `apply_created`): the file's date, unless the card's first
   upload is earlier and the entry isn't a hand one; a card the file misses (one added since the last `created` run)
   gets its first upload. The bot's first upload is the earliest version the database holds (`_first_upload`; the
   website's is the PNG's first commit): equal for every card first uploaded since 2026-02 (11 of 11), and the file
   covers the older ones, where the bot's history is cut short (770 of 770 cards dated as the website on 2026-10-23).
   Rulebook pages stay undated. A missing or unreadable file leaves the dates as they are.
8. **Archive** (merged on main, dfdf953): `prune_clean_cards(at)` runs only when `_can_prune` (no failed new-name PSD, at
   least one card). Each tag's `at` and `commit` come from the deleting commit in the index, not from the run's clock or a
   commits-API call. `renamed_to` comes from an exact rename, or from a new card with the same blob in this run.
9. **History** (§7.4).
10. **Changelog events** (§12.1), each with the push that brought its commit.
11. **Gates** (§13).
12. **Write** `db/`, the manifest (with `pending.repin` for the PNGs baked in B), the summary and the commit message.

### 7.2 Dates and authors

- **Card timestamp.** A card's `timestamp` becomes the time of the commit that first added its current blob. A move, a mode
  change or a re-commit of the same bytes no longer re-dates a card, and the website's "Recently updated" stops showing moves.
  Predicted at the cutover: 31 current cards re-dated, 0 data fields changed (§2). A card baked in a run is dated by M.
- **First author.** It comes from the oldest commit at the path, as the commits API gives today. `site/Card Authors.txt` still
  wins.
- **No commits API.** The commits-page lookups (`_commit_page`, `LOOKUP_WORKERS`, `_committed_since_update`) leave the Action's
  path. Legacy mode on the jetson keeps them until step 10, because the jetson's git 2.25 must never run `log --raw` in a
  blobless clone (memory `jetson-server`).

### 7.3 Incremental and full builds

- **Incremental** is the default. It downloads only the changed PSDs that are not already local, plus nothing else.
- **Full** (`full`, or a `PARSER_VERSION` bump approved through G4) means 813 downloads (about 16.4 GB, streamed through temp
  files). That is three budgeted runs, continued by the gate (b): about an hour at 30 minutes, two hours at hourly. The budget
  is 400 so that two runs in one hour stay under the token's 1,000 requests an hour with room for card-check.
  - Between the runs the data mixes parser versions. `manifest.pending.stale` says so, and the site footer and
    `/update_stats` show "re-parse in progress: N cards still on parser version 2".
  - The approval (`pending.full = {parser_version, approved_run}`) carries over to the continuation runs.
- **Caches:** fonts (carried over from card-edits.yml) and pip. The data lives in git and needs no cache.

### 7.4 History rules (old_stats refinements)

- **A version only when data changes.** A new version is recorded only when a card's data fields change: card_type,
  display_name, ability, stars, subtype, series, subseries, types, stats, creatures, author, note, hidden, duplicate_of.
- **Each version gets `version`:** `{from, from_commit, to, to_commit, cause, png_commit, legacy}`.
  - `cause` is one of `psd` (the blob changed), `reparse` (same blob, newer parser), `moved` (the same blob in another
    folder, whose folder-derived data changed: stars, card type, subtype or series), `edit` (`Card Edits.json`), `authors`
    (`Card Authors.txt`) or `legacy`. A move that changes no data field is no version (below). A re-read of the same
    PSD replaces its reading and keeps its version record whole, cause too (so a re-read card can still show `psd`).
  - `from` is the blob's intro (for psd and reparse; after a revert, the revert: `GitIndex.blob_since`) or the commit of
    the edits or authors file. `to` is the next version's `from`, or the deleting commit.
  - `png_commit` is the newest commit of the card's PNG at `to`, so `/query as_of` can link the image of that day with no git on
    the jetson.
- **Art-only changes are not versions.** A blob change with identical data becomes an `art` changelog event and updates the
  current card's `blob_sha`.
- **The same rules in legacy mode (2026-10-22, owner: "clean our history of dupes").** `/update_stats` archives the card
  before only when its data changed too (`_record_history`, after the authors and edits); with the same data the card
  takes the new blob, path and date and keeps its start in a `version` record (`card_history.carry`: cause `legacy`, no
  git), which `/query as_of` reads (`from` before `timestamp`) and the seed pass dates again from git, keeping the earlier
  start. Each `/update_stats` runs `card_data.clean_history` first: one line, "History: N duplicate versions removed (M
  cards), K brought to today's series and type names, archive tags from git: ...", the first time; nothing after.
- **What counts as the same data (2026-10-23, owner: "fix the dupes and other issues with our history").** `same_data`
  compares one form of each field: a title equal to the name is none, an empty ability is none, a retired type icon is
  its name now (`TYPE_RENAMES`: `offense` became `Negation.png`, the same bytes, in 9649b4d). A field a reading could not
  know is no difference (`unknown`): the title of a version read before parser 2 (it read no titles), the stars of a
  `NO STARS FOUND` parse, the types of a `MISSPELT TYPE` one, the ability of `NO ABILITY LAYER`. A series the 2026-10-20
  split moved under a parent (the same subseries-or-series) is no difference. The changelog's steps
  (`changelog.field_changes`) use the same forms but run both ways: a field either reading could not know is left out
  of the step, so a step to a reading that knew less lists no change in it. The history's rule only runs one way
  (`knows(a, b)`: a knew every field b knew, with b's value): the reading that knew more absorbs the one that knew less,
  never the reverse. So a commit with a broken ability, star or type layer keeps the card before as a version
  (`_record_history`, `record_versions`, the untag step), and the cleanup never folds a version into a reading that
  knew less or links two versions whose known values differ.
- **The one-time cleanup** (`clean_history`, also the seed pass's first step):
  - with the history clone and the bot's own full clone (`BOT_CLONE`, default the checkout holding global_config.py: the
    jetson's ~/Documents/AutoCroissant; skipped quietly without either; the seed pass reads the builder's checkout or
    `refine --bot-git`): `recover` (2026-10-23, owner: "fix the dupes and other issues with our history"). Every
    stats.pkl/old_stats.pkl the bot committed (`bot_pickles`: one `git cat-file --batch` process, one snapshot
    unpickled at a time with the allow-list unpickler and folded into the readings before the next, a reverted PICKLE
    left out; kept for the process, so a new PICKLE commit (each /push) is the only one read then: 45 MB peak on the
    Mac for 32 commits, 114 MB before) gives readings, one per name, path and timestamp; a name one PICKLE held at two
    paths (the 2025 ones are keyed by path: Auxiliary/Minions/Auto_Croissant.psd, now Mini Auto Croissant, beside the
    creature) keeps its stored versions' path's readings (2 left out). Each is dated from git: the commit at its path
    with its timestamp, else (the April 2025 PICKLEs hold file mtimes) the one blob its path held from then to the
    PICKLE, dated by the commit that put that blob at that path (a reading never stands for a time its file sat at
    another path: Stone Spikes' bytes in Items/Offense before 2025-03-24) since the path last held other bytes (a
    revert dates from the revert, as `blob_since`: King Dedede's 9fa2e2df reading of 2025-04-09 starts at 243a357,
    2025-04-03, not at a0fcf23, before the 'xvs' edits; `GitIndex.arrival(blob, path, at)`); undatable readings are
    skipped. Readings of one blob at one path are one (the newer parse, the earlier start) within one such stretch
    (back after other bytes: two, and a run holding both starts at the return, `_first`); at another path they are
    another reading (the
    same bytes print another card type there: Air, Earth, Fire, Water Totem and Crystal Bits minions until 2025-08-29,
    Mettaton EX/NEO, Igor Roballtowski and Shadow Queen minions with their stars until 2025-09-01). A reading of a
    stored version's blob at its path only moves that version's start earlier (parser-only), never across other bytes
    at that path (`GitIndex.held`). The others, between the
    stored versions: one
    printing what the version before it prints (ability words in any case without icon tokens or punctuation, types,
    stars, each stat, card type; a field it could not know (a stat read as -1 alone: Uumuu's 2025 DEF), an
    impossible value or a type list alone that differs while the PNG stayed aside) is nothing; a run printing the same
    goes into the version after it when that prints it too and knew all it knew (its earlier start carried: format-only), else becomes a legacy version (the old parse's
    PSD fields, the next version's series, title, author, note, hidden and creatures, `version.restored_from` the
    PICKLE, the start's commit and that commit's blob: a pair git had). A name in neither pickle gets its readings as
    versions, the last tagged `archived` (`backfill`) from its path's deleting commit (none in git: nothing) with the
    bytes git deleted (Chestnut's 4155a8f and Chestnutter's 9894185, edited twice after the PICKLEs' last reading; the
    record keeps the reading's blob), `renamed_to` along an exact rename chain to a listed card (`GitIndex.renamed`),
    `moved_to` into MDW. **One time only** (2026-10-23 review): `recover` does nothing once the database holds a
    restored version (`recovered_before`; the bot then reads no PICKLE at all) or any PICKLE commit did, a reverted one
    included (`Pickles.done`, the bytes `restored_from` in the commit's pickles). So a restored version removed later on
    purpose (a fold of the types-only ones below, dupes-5) stays removed, and undoing the recovery is a pickle-only
    revert of its PICKLE (no code revert). Idempotent besides (`once=False`): a second pass finds every reading held;
  - `unknown_values`: an impossible history value (stars outside 0..7; a stat above 10 without an allowance, INT-4's
    rule; with one, a stat of three digits) of a version read before parser 2 (`unchecked`: no `parser_version`, the
    validator had no range check) becomes unknown (stats -1, stars None), the old value kept in an `OUT OF RANGE: ...`
    problem; the build runs it after its versions and G3 refuses any such left (the range check), the seed report
    counts it. A parser-2 reading's value is kept (its problem showed when it was read; an allow_high_stats entry added
    later allows it) and the build lists the names ("History values out of range kept ...", `kept_values`). An
    allowance (`allows_high`): the card's allow_high_stats, a name on the retired EXCESSIVE_STAT_EXCLUSIONS (Royal
    Eradicator Main Cannon, Pix, Tainted Lazarus, Sonic, Twin Emperors), or no current card to tell (archived); the
    values it keeps above 10 are listed for the owner (`high_values`, below);
  - with the history clone: `misreads`: consecutive versions whose PSD blobs differ while the card's PNG blob is the
    same when the older one's bytes came to its path and at the newer one's start (+6 h each, at least 6 h apart: a
    push's PNG export can trail its PSD) and whose data differ in the type list alone are one version read twice: the
    older goes into the newer (which knew all else it did), its start carried; the list is printed. Run before or
    after `dedupe` (main's or this cleanup's), in the bot or the seed, the result is the same (rehearsed: 1081 versions
    each way);
  - `normalize`: a stored version's series is filled from the version after it when it had none (an item's series is typed
    in later; a creature's comes from its folder, never filled), or moved under the parent the split gave it; retired type
    icons renamed; a version's copy of the card's creation date dropped (the card and an archived record keep it);
  - `collapse`: an earlier reading of the same commit goes first (two versions at one start, no other blob: the older
    parser's or the pre-edit reading, when the later one knew all it did; nothing carried), never the card before a
    version an edit or authors change made (cause `edit`, `authors`: `record_edits` and `record_versions` start it no
    earlier than the version it ends, so at that start when the edits file's newest commit is older, as an edits commit
    merged after a PSD push), then runs of the same data. A
    run needs a member that knew all the others did (`_knower`), so an unknown title never joins two known ones; a run of
    exactly equal versions keeps its first, a run that is only the same because a reading knew less keeps the reading
    that knew the most; the card after it (or an archived record) takes a run only when it knew all of it;
  - with the bot's history clone (`HISTORY_CLONE`; skipped without one): `date_archive`, so an archive tag made by the
    run's clock takes its deleting commit's date, an exact rename's `renamed_to` (else its PNG's exact rename), or
    `moved_to` for a PSD moved into MDW (retired, not deleted); a tag linked along a chain is dated by the step that
    reached its end, the old name's last day and where the new name starts (`intro_of`): Azamoth the Possessed King at
    d64fc8d (the rename to Vessel, 2025-09-02 00:05), not 72e33d3 (a move within its name 1.5 h before); Pandiety at
    b3d1e7c (the move into MDW/Unfinished), not 1a036c2 (its rename to Pandeity, a name no record holds, 44 min before);
  - with the history clone: `commits_of`: a start carried from a version stored without its commit gets the one commit
    at the version's path with that time (`blob_at`, as the seed does), so the bot's records name it too (Stone Spikes
    from 2025-08-30: 4c32fcb8).
  The step runs at every /update_stats on a copy of the cards and history, put into the database only when it ran
  through (`card_data.cleaned`, 0.03 s for the jetson's copy): an unexpected error in it is one line ("History cleanup
  skipped (...)."), never a failed run and never half a cleanup saved.
  Measured on the jetson's 15:41 pickles at TTSCardMaker 6e50c3ee: 755 -> 446 versions, 448 -> 354 names (309 dropped on
  293 cards, 63 normalized), 289 version records; ZOEY!!!!! (renamed to Z!!!!! in 35adb383) and White Palace (moved to
  MDW/The Morgue in b6330730) dated from git. With the recovery (work/history-recovery f12d223, merging main e837f68
  and work/history-fixes 93e3458; the bot clone at main's 32 PICKLEs; rehearsal `wt/scratch/recovery/runs/I_*`):
  755 -> 1089 versions, 448 -> 655 names: 698 versions recovered on 553 names (by the PICKLE whose reading knew most:
  3be93f3 4, ad4b6b4 16, d1840e6 33, d77fb6a 278, e5f5393 367; 622 on current cards, 76 on archived names), 171
  starts moved earlier by a reading, 64 cards gone before archiving archived (38 with renamed_to, 25 moved_to into
  MDW, with ZOEY!!!!! and White Palace; 3 deleted: Amy Rose's Hammer, Sacrifice, Tainted Esau; 17 tags hold bytes git
  deleted after the last reading), 55 type misreads merged, 12 impossible values unknown (7 names), 309 duplicates
  dropped, 101 normalized, 306 carried starts given their commit (3 left without one: no commit at that time); 0
  cards modified; the next run writes nothing (the recovery is done: no PICKLE read); the forced re-parse adds no
  version (13 current cards' data change: the 7 printed-star ones and the 6 DATES2-3 abilities); every one of the 755
  stored versions' data is still held by its name, no name lost, no earliest start later (603 earlier). Restored
  versions: 0 whose blob and from commit git never paired, 0 at a path that did not hold their blob at their start,
  6 consecutive pairs sharing a blob (the minion -> item/creature moves above). The 2026-10-23 review's fixes changed:
  105 versions (77 restored, 28 stored with a carried start) started before the 2025-03-29/04-02 'xvs' edits with the
  same bytes back at the 2025-04-03 'revert text 1' commits (61 with another PNG between); 0 now in the bot's run, the
  forced re-parse and both seed passes (0 without the recovery too), so 11 fewer starts move (King Dedede, The Freezer,
  Balloon Boy, Old Freddy, HR-D3, Toriel, Necrodeus... start 2025-04-03 again); Uumuu's DEF -1 reading is no version
  (698, not 699); 3 archive tags moved to the step that reached the chain's end (Azamoth the Possessed King and
  Militron No Armor to d64fc8d, where their new names start; Pandiety to b3d1e7c). The seed pass after it and the seed
  pass of the raw 15:41 pickles give the same result (0 differences; each idempotent; after the bot's run the seed
  skips the recovery as done); it re-dates 18 cards and 3 versions by the bytes' first commit (below), so the bot's
  and the website's earliest starts differ on 22 names until the cutover.
- **Open history decisions (2026-10-23; owner).** Findings in `wt/scratch/history-1023/all-findings.json`. Decided the
  same day from the owner's rules (archive, never drop; skip identical versions, keep the earliest date; only printed
  text counts) and done by work/history-recovery: all 64 cards dropped before archiving existed are archived with
  their history, renames linked by `renamed_to` (archive-1, INT-1, D4); the lost pre-2026-02 and 2025 history is
  recovered from the PICKLEs (D1, C1, DATES2-1); the types-only steps with an unchanged PNG are misreads (55 folded;
  the 3 `offense` ones were already; Miracle Matter's is an ability step); INT-4's values are unknown; Spawn of the
  Forbidden One's start is 2026-06-19 (d8e94877; its June reading printed what the card prints, the 10 stars aside).
  Still open:
  - **Side files**: the 4 cards still credited to their renamer in Card Authors.txt (The Abyss Watchers, Pontiff
    Sulyvahn: hdckighfkvhvgmk; Firebomb, Maxim Tomato: MichaelJSr), and the lines that match no card (White Palace in
    Card Authors.txt; Gust of Trees, Stubborn Queen in Card Creation Dates.json until TTSCardMaker's work/history-dupof
    lands their dup_of entries, which make them cards). `/update_stats` prints them each run.
  - **dupes-5**: the 4 rulebook pages (09, 11, 12 Reactions2, 22 Negation) keep versions that change nothing printed
    (hidden text read by the old parser) until their old blobs are re-read once; there is no idempotent marker for that
    inside `/update_stats`, so it is a one-off on copies (after the recovery ran, so no PICKLE reading brings the
    versions back: the recovery is one time only).
  - **Landing order**: land work/history-fixes before main's first post-deploy `/update_stats`. Otherwise main's dedupe
    keeps Tainted Bethany's blob-less pre-parser-2 reading of 2026-10-19 and drops the c532847b parser-2 one; the data
    shown is the same, only the blob and parser record differ (nothing can bring the dropped record back).
  - **Seed dates by the bytes' first commit at any path** (`GitIndex.blob_since`): a version's or card's start is its
    blob's first commit at any path, unless its path (or the path an exact rename brought the bytes from) held other
    bytes after that: then the commit that put them back (2026-10-23; the 2025-04-04 revert of 112 PSDs, as legacy
    dates them). Not across a name the database holds (`GitIndex.intro_of`, 2026-10-23): a renamed card starts at the
    rename, since the old name's archive (`renamed_to`) holds the time before (Blasphemous Blade from 2025-09-02, not
    Blashemous Blade's 2023-11-05; the seed listed 24 of the 38 renamed names twice before, 14 with one blob under
    both); a build dates a new name so too. A start carried from a version the bot's cleanup folded is dated by that
    version's bytes like the version would be, so the seed after the bot equals the seed of the raw pickles (You are
    going DOWN you Sputnik! 2025-08-30, 08 Global Turn 2025-03-28 both ways). Restored versions keep the start where
    their PICKLE read them (`version.restored_from` with its commit: no re-dating), so 50 names' earliest start is
    later than the first recovery rehearsal's (the Auxillary/ items 2025-02-10, the Items/Offense -> Counter spikes
    2025-03-24, the 9_ -> 09_ rulebook pages 2025-03-29), still never later than the stored history's; if the owner keeps
    the any-path rule, restored versions should take it too (`_locate`, `_dated`), if he picks "a name's own path",
    `start_of` should. On the cleaned 15:41 copy at 6e50c3ee the seed and the build no longer date 9 versions and 3 cards
    (Balloon Boy, Old Freddy, Dark Mind Eye; King Dedede's 9fa2e2df version, Holo Defense API's renamed after its
    revert) across other bytes (12 before, 0 now; idempotent); with the recovery too since its 2026-10-23 review fix
    (105 at d5a904e, 0 now: a PICKLE reading dates from the revert as well). With the recovery, 22 names' earliest
    start moves earlier than the bot's, to when their PSD's bytes first appeared under another path or a name the database never
    held (Shell Shield, Dragon Mount, Unbreakable Strength 2025-02-10 -> 2023-11, the misspelt `Auxillary/` folder; 12
    rulebook pages first committed as 3_ ... 8_ and the 2 covers; 4 status effects moved within minutes; Sputnik). A move or rename keeps
    the version going, so this is by the rule; say so if the owner wants a name's history to start at its own path.
  - **History stats above 10** (INT-4's rule, 2026-10-23 review): kept by an allowance and listed here: Sonic 7/4/4/23
    (2025-03-09 and 2025-08-28; it prints 20 now, allow_high_stats), Pix 12/20/25/24 and Twin Emperors 3/7/12/13
    (retired exclusion list; Twin Emperors has no allow_high_stats now), Royal Eradicator Main Cannon 9/8/12/3, and the
    archived 6-star Bill Cipher 16/6/6/10 and Oryx The Mad God 16/12/8/4 (no card to tell). Set unknown: Tainted
    Lazarus 110/110/110/110 (4 versions; its blobs re-parse to 7/4/5/5), God Tamer 7/9/10/15 and Blank Card 55/55 (no
    allowance), stars 10/15/17/36 on Spawn of the Forbidden One (3), Chestnutter, Michaels Blessing, Indoctrination.
    Say which kept ones are misreads, or which unknown ones were printed (an allow_high_stats entry keeps them).
  - **26 restored versions differ from their neighbour in types alone while the PNG changed** (so no misread by the
    rule; the PNG blob differs between the two starts in all 26, checked again by the review): 19 Status Effects2,
    Abigail, Barry The Barrel, Blazing, Burning, Chilled, Chomp, Cortez, Drawcia Soul, Engorged, Flurrie, Galacta Knight,
    Grimora, Hive Knight, Living Bread, Lost Kin, Mantis Lords, Mosling, Paintbrush ([dark, darkmatter, light, machine]
    -> [nme, support]), Poison, Reverse Mushroom, Smoldering, Stuffed, Tainted Lazarus, The Bull, Warcraftian Shaman.
    Many look like old-parser readings (fire/panda/machine before the debuff badges) rather than printed changes; their
    old PSD blobs are not in the local clones (tc-full.git is blob:none), so this can't be checked offline. Say whether
    to fold them like misreads: the recovery is one time only, so a fold (a new cleanup rule) sticks.
  - **Kept by the rules** (no change; say so to change them): dupes-3/INT-2, the 6 versions dated before their PSD's
    first commit (the Ahn'Qiraj cards' author times of rebased commits 48a9d42 and d5d8a04; Z!!!!! v0 of 2026-04-15
    matches no commit): the earliest date stays, and Z!!!!! v0's data differs from the version after it. dupes-4, the
    ZOEY!!!!! -> Z!!!!! exact rename: each name keeps its own history, the old one archived with `renamed_to`, so
    ZOEY v1 and Z!!!!! v1 (the same data) stay one under each name.
  - **Done on other branches** (the AutoCroissant ones merge this one; read them with it): work/history-recovery:
    D1/C1/DATES2-1, the lost pre-2026-02 and 2025 history of current cards (397 versions, the highest-severity finding),
    recovered from the bot's PICKLE commits, with archive-1/INT-1/D4, the types-only misreads, INT-4 and DATES3-1;
    archive-r2-dup-dropped (an alt-art duplicate archived like any card); INT3-1 (space-named card files take
    underscore lookups, e4cd227); INT-R2-2 (Auxiliary/Creatures stars from the PSD, then a `cards:` re-read); INT-R2-5
    (the bake's redundancy check first, bb183d0). TTSCardMaker work/history-dupof: INT-3, Stubborn Queen and Gust of
    Trees as `dup_of` duplicates (no database entry until then; their Card Creation Dates.json lines match no card).
    TTSCardMaker work/mario-series: INT-R2-3 (41 Mario cards to series Mario plus a subseries). The website,
    TTSCardMaker work/site-history with AutoCroissant work/site-history's site tests: C2 and the viewer's carried start
    (cards.json `since`), C5 (committer time), C3-site-local-day (Pacific dates), C3-dup-history-split (a duplicate's
    History). Still open, no branch: INT-R2-4, Computer Virus has no stat rows (a PSD or Card Edits `creatures` fix;
    BUGS_AND_IMPROVEMENTS D1).
- **Moves are not versions** either (the path is not data); they become a `moved` event. `_update_old_stats_paths` (rewriting
  every old path to the current one) is removed from both modes (legacy too, 2026-10-23), so each version keeps the path it
  was parsed at. Both readers key history by
  name: build_site reads only `ability` and `timestamp` of old versions (build_site.py:1485), and `/view_old_metadata` shows the
  version's own path.
- **Duplicates** (`duplicate_of`) get no versions; they are copies of their original. A duplicate is a numbered PNG beside
  its original's PSD, or (2026-10-23) a card PNG with no PSD whose Card Edits.json entry has `dup_of` (build_site's rule;
  Stubborn Queen, Gust of Trees). One whose PNG or original leaves is archived like any card (its tag at its own PNG,
  dated like its original when the PNG stayed; never counted by G2), and back when found again. Their `version` is a copy of the
  original's, made after the original's own is set (2026-10-23): before, a duplicate copied a re-parsed original's empty
  record, so a full re-parse and an incremental build disagreed and the next run rewrote 6 duplicates with no input change.
- **Seed pass** (once; idempotent afterwards, so a later run changes nothing):
  - drop the duplicates (`card_history.collapse`, never one tagged `archived`): of each run of consecutive versions with
    the same data the first stays; a run with the current card's data (or ending in the archived record) goes into that
    card. The run's earliest `from` moves to the version that holds its data; that version keeps its own cause and
    `legacy` flag (−15 on the 2026-10-22 pickles: 5 repeats of the version before, 10 equal to the current card);
  - re-date the versions that have a blob (3 of 178 change);
  - give a blob-less version the blob of the one commit at its path with its stored time (`GitIndex.blob_at`; then it
    is dated like any version), and a carried start that commit; mark the rest `legacy: true` with their stored dates;
  - date ZOEY!!!!!'s archive tag: `at` 2026-06-18 and `commit` 35adb383. The merged archive code backfills the tag on the
    jetson's first /update_stats after the deploy (no PICKLE since dfdf953 yet); the seed pass then only adds the commit and
    date, or tags it itself when seeded from an older PICKLE.

## 8. Data formats (`db/` on branch `data`)

| File | Content | Readers |
|---|---|---|
| `db/manifest.json` | What was built from what (below) | everyone: freshness, staleness, the gate, the bot's change check |
| `db/stats.json` | `{"schema": 1, "cards": {name: record}}`, one card per line, sorted by name; current cards including duplicates | site, the save, bot, tests, diagnostics |
| `db/old_stats.json` | `{"schema": 1, "cards": {name: [record, ...]}}`, one name per line: history plus archived cards | site (history), bot (`/view_old_metadata`, `as_of`) |
| `db/psd_details.json` | Unchanged: schema v1 `{version, cards: {blob sha: record}}`, one record per line (`psd_details.text_of`) | site Details, bot `/fonts` |
| `db/changelog/YYYY-MM.jsonl` | One event per line, appended by each build | site Changes tabs, the post stage |
| `db/changelog/posted.json` | `{"seq": 120, "push": "<after sha>"}`: the last posted event and its push | post stage, gate (e) |

**Record.**
- **Keys are CardInfo's attribute names;** `stats` and `creatures` hold CardStats' (`hp`, `defense`, `attack`, `speed`). For
  example:
  `"The Freezer": {"ability": "…", "author": "…", "blob_sha": "…", "card_type": "creature", "name": "The Freezer", "parser_version": 2, "path": "…", "stats": {"attack": …, "defense": …, "hp": …, "speed": …}, "timestamp": …, …}`.
- **Unknown and missing keys.** A record key CardInfo lacks is ignored, with one warning per key. A missing key takes the
  field's default.
- **Written** with `json.dumps(…, ensure_ascii=False, separators=(",", ":"), sort_keys=True)` per record (floats round-trip
  exactly), and atomically: temp file, fsync, replace, as `save_pickle` does.
- **The round trip is a test:** `record(card(record(c))) == record(c)` and every field equal through `getattr`, for all 819 cards
  and 469 versions of the live pickles.
- **The archive tag** is main's `{at, path, blob, commit, renamed_to?, moved_to?, backfill?}`, with `at` and `commit` from
  git (§7.1, step 8); `moved_to` is the MDW path of a PSD moved there (retired, not deleted).

**Manifest.**
```json
{"schema": 1, "built_at": "2026-10-22T18:04:11Z", "run": "https://github.com/MichaelJSr/TTSCardMaker/actions/runs/<id>",
 "card_repo": "MichaelJSr/TTSCardMaker", "card_commit": "<main sha built: M, or main's head>", "card_commit_at": 1792700000,
 "main_commit": "<M, or null when the run made no main commit>",
 "builder": {"repo": "bobberlington/AutoCroissant", "ref": "main", "commit": "<sha>", "parser_version": 2, "psd_tools": "1.25.0", "python": "3.14"},
 "seed": {"repo": "bobberlington/AutoCroissant", "commit": "<PICKLE sha>"},
 "inputs": {"edits_blob": "…", "authors_blob": "…", "created_blob": "…", "manifest_blob": "…", "decks_blob": "…", "types_tree": "…", "fonts_tree": "…"},
 "counts": {"cards": 819, "versions": 461, "archived": 1, "details": 813, "changed": 3, "new": 1, "failed": 0, "baked": 0},
 "pending": {"stale": [], "failed": [{"path": "…", "blob": "…", "error": "…"}], "full": null, "repin": []},
 "problems": [{"card": "…", "path": "…", "notes": ["HP NOT FOUND"]}],
 "gates": {"G2": "override by run 123"}, "restored_from": null,
 "files": {"stats.json": "<sha256>", "old_stats.json": "<sha256>", "psd_details.json": "<sha256>"}}
```

**Changelog event.**
```json
{"seq": 121, "at": 1792700000, "build": "<run id>", "card_commit": "<main sha built>", "commit": "<sha that caused it>",
 "push": {"after": "<sha>", "before": "<sha>", "by": "<pusher's login>", "at": 1792699000},
 "by": "<its git author>", "edit_by": "<_by of the edit>", "kind": "new|changed|art|moved|renamed|archived|restored|edited|authors|reparse|restore",
 "card": "<name>", "title": "<display name>", "card_type": "creature", "stars": 3, "path": "…", "from_path": "…",
 "fields": {"hp": [5, 6], "ability": ["old text", "new text"]}, "blob": ["<old>", "<new>"], "renamed_to": null}
```
- `seq = max(the largest seq, posted.seq) + 1`, so a restore never reuses a number.
- **`push`** comes from one activity API call per run (`repos/…/activity?ref=refs/heads/main&activity_type=push`, §2): the push
  whose `before..after` holds the event's commit. The run's own commit M is a push of its own (by `Card sync`). Without the
  activity list (an API error, or a push older than the 100 listed), consecutive commits by one author count as one push.
- The seed build writes no events: the re-dates, dedupes and backfill are not card changes.
- Growth is a few KB a day; one file per month keeps each file small.

## 9. Credentials (names only)

| Where | Name | Use |
|---|---|---|
| TTSCardMaker Actions | `GITHUB_TOKEN` (built in) | Reads the card repo and the activity list; pushes main (`contents: write`: the bake and the save, as card-edits.yml and tts-save.yml do today) and `data`; `gh workflow run site.yml` and `gh run list` (`actions: write`) |
| TTSCardMaker Actions | `DISCORD_CHANGELOG_WEBHOOK` (set 2026-10-22) | The per-push posts and the failure alert |
| TTSCardMaker variables | `CARD_DATA_BUILDER_REF` (optional, not secret) | Pins the builder (§6.4) |
| The bot's config.py (jetson) | `GIT_TOKEN` (existing) | The change check on `data` and the commits of Discord edits (the API, else SSH through the history clone, as tts_save does) |
| The bot's config.py (jetson) | `ACTIONS_TOKEN` (new, optional; decision 4) | A fine-grained token on TTSCardMaker only, Actions read and write: `/update_stats` with `force_update` or `cards:` dispatches Card sync |

- **AutoCroissant needs none.** It has no workflows and nothing writes to it any more.
- **Not needed:** a PAT, a deploy key, or minimal's `AUTOCROISSANT_DEPLOY_KEY` or `AUTOCROISSANT_PUSH_TOKEN`.
- **The webhook URL.** The owner pasted it in chat on 2026-10-22. It lives only in the secret. Never write it into a file in
  either repo, since both are public. If that transcript is ever shared, regenerate the webhook in Discord and update the secret.

## 10. Consumers and how each changes

- **site.yml and build_site.py (TTSCardMaker):**
  - **The data checkout.** A checkout of `ref: data, path: carddata, fetch-depth: 1`, and env `CARD_DATA_DIR=carddata/db`. When
    the variable is set, `load_bot_pickle` and `load_bot_json` give way to `load_card_data()`, which reads the JSON into the
    same `_Obj` stand-ins, so `bot_field_warnings` and the rest of the build are unchanged. When it is unset, the build reads
    AutoCroissant as today, which is the rollback.
  - `archived` and `version` join `CARD_FIELDS_OPTIONAL` (`archived` already did, e0e26b2f).
  - **Version dates.** analytics dates a version by `version['from']` when it has one (`analytics.when`, TTSCardMaker
    work/version-dates), else its timestamp. After the seed pass every version has a record, so every website date moves to
    the blob's intro, and a `moved` version to the move commit (as the bot's changelog does).
  - **Triggers.** The push trigger stays as today (paths-ignore `TTS/**`), so website edits and new art publish at once on
    the last data. Card sync dispatches it after each commit, with the new data. The cron goes from every 6 hours to daily
    (`17 12 * * *`), as a safety net.
  - **New outputs:** `data/changelog.json` (the last 180 days, names turned into card ids) and `data/analytics.json` (§12.3).
  - **The footer** reads "Card data built <time> from TTSCardMaker@<sha> (run)". When `manifest.card_commit` is more than 3
    hours behind the head on card paths, a staleness warning goes in the log and the footer.
  - **Texts:**
    - "Not parsed by the bot yet … Run /update_stats" (app.js:691) becomes "added by the next card sync, within 30 minutes
      (link)".
    - "Re-parse pending" (app.js:68) becomes "the data follows the next card sync".
    - edit.js `ABOUT` and its "the bot reads it on /update_stats" lines become "the next card sync applies it (every 30
      minutes)".
- **tts-save.yml:** folded into Card sync (stage 3) and deleted at the cutover. Its daily gate becomes gate (a), which already
  covers its inputs (`*.png`, `TTS/decks.json`, `Card Edits.json`, the save); its card-browser names check goes, because the
  names come from the run's own data. tts_save.py gains `--names` (§7).
- **card-edits.yml:** folded into Card sync (stage 1) and deleted at the cutover. Its `card` and `dry_run` inputs become
  `bake` and `dry_run`; its commit becomes part of the run's one main commit.
- **card-check.yml, font-check.yml:** unchanged. card-check keeps reporting on each push's PSDs at once, so card makers still
  hear about a broken PSD in minutes; merging it into Card sync is a later, separate idea.
- **The bot:** §11.
- **Tests (AutoCroissant `tests/`):**
  - **New:**
    - `test_card_db.py`: the live round trip (skipped without the pickles), the record format, the manifest hashes, the seed
      unpickler refusing a non-allow-listed global.
    - `test_card_history.py`: a fixture `git log --raw` text (newest, first author, blob intro, deletions, exact renames,
      version meta, dedupe that keeps archived tags, legacy marking).
    - `test_card_data.py`: `--help` on `SPARSE_FILES` in a clean subprocess for the three CLIs, the gates G1-G6, the exit codes,
      an offline end-to-end build on a temporary git repo of fixture PSDs, and the B-to-M rewrite.
    - `test_changelog.py`: every kind, a `seq` never reused, events grouped by push (activity list and the fallback), the
      Discord payload within limits (§12.1), the cursor moving only after each accepted message.
    - `test_metadata_edits.py`: Discord edits produce exactly `dump_edits` text and Card Authors.txt lines; `/mass_replace` makes
      one commit; the alias migration mapping.
    - `test_query_as_of.py`.
  - **Changed:** `test_update_stats.py`, for hook (c) and legacy-mode parity (the moved code writes the same pickles as before on
    the fixtures); `test_card_bake.py` for `--local` (one commit, the same files as the per-card commits); `test_tts_save.py`
    for `--names` (the same names as `cards.json` on a fixture).
  - `conftest.real_stats` keeps reading the root stats.pkl until step 10, then reads `db/stats.json` from the Mac clone's
    `origin/data`.
- **Diagnostics:**
  - `diff_stats.py` and `inspect_pickle.py` (`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`) also read
    a `db/` folder, a `stats.json` or a data commit (`--git`, e.g. `origin/data` in a TTSCardMaker clone) through
    `card_db.read_cards`, with no config.py (built 2026-10-23; the diagnostics skill's Workflow 4 is the shadow's daily
    check). So the pre-push gate becomes a review of the data commit. The job summary already holds the same diff.
  - `export-pickles` turns any `db/` back into pickles for the older tooling.

## 11. The bot afterwards

**A reader of card data.**
- **The change check.** `card_db.sync()` sends one conditional request,
  `GET repos/MichaelJSr/TTSCardMaker/commits/data` with the sha accept header and `If-None-Match`. A 304 is free.
- **On a new sha:**
  - it fetches `db/manifest.json` and then only the files whose sha256 changed, from `raw.githubusercontent.com/<repo>/<sha>/db/…`
    (immutable at a sha, no lag), and checks each against the manifest;
  - it writes them atomically into `card_data/` (gitignored) and loads `stats_db` from JSON;
  - it queues today's follow-ups: `populate_files` (its own tree fetch, an ETag 304 when unchanged), `prep_dataframes`,
    `sync_changed_emojis` and `refresh_history`.
- **It never unpickles anything from the network.**
- **When it syncs:** on_ready, `/update_stats`, and a reminder at Card sync's interval, a few minutes after each run
  (`/update_stats` every 30 minutes, or hourly; decision 5). The 05:00 reminder d2664752 becomes redundant and is removed with
  054b5933.

**Commands.**
- **`/update_stats`:**
  - **No options:** sync, then reply "Card data built 12 min ago from TTSCardMaker@abc1234 (run): 3 changed, 1 new, 1
    archived; reloaded", or "already current".
    - If main is ahead on card paths, the reply says the next card sync (at :04 or :34) picks it up, or links the run when one
      is going (`actions/runs?status=in_progress`, readable on a public repo).
    - If the last run failed, it links that run.
  - **`force_update` or `cards:`** (admin only from step 8, since they cost API budget): dispatch Card sync with `full` or
    `cards` through `ACTIONS_TOKEN`, poll the run with progress edits for at most 30 minutes, then sync. Without the token, it
    replies with the Actions link and `gh workflow run card-sync.yml -R MichaelJSr/TTSCardMaker -f full=true`.
  - **`output_problematic_cards`** shows `manifest.problems`.
- **`/alias`** (decision 3) commits `site/Card Edits.json` with `aliases` on the card the PNG belongs to. The commit is `Discord:
  /alias breakfast -> A Snack2`, with an `Edit-by` trailer and `_by`/`_at` on the entry. The bot's lookups already merge
  `Card Edits.json` aliases (`CardEdits.names`, card_edits.py:190), so the alias works at once. `/alias` with no value removes
  it the same way.
- **`/update_metadata`** (decision 2) makes one commit to the card repo through `card_bake.land` (it retries when main moved) and
  `dump_edits`, which moves to card_edits.py and stays re-exported from card_bake. The subject is, for example, `Discord:
  /update_metadata Tainted Bethany (hp, ability)`, with an `Edit-by` trailer and `_by`/`_at` on the entry. Fields map like this:
  - `author` → a `[cards]` line in `site/Card Authors.txt`;
  - `ability`, `stars`, `types`, `hp`/`def`/`atk`/`spd` → `Card Edits.json` fields of the "psd" kind. They persist, show
    "PSD pending" like a website edit, and the next Card sync bakes them into the PSD. Stars and types too since
    2026-10-23 (the bake prints them; a layout it can't follow is refused for that field alone, the card's other edits
    still bake);
  - `subtype`, `series`, `card_type` → `subtype`, `series`, `type` (the "folder" kind);
  - `subseries` → `subseries`;
  - `del_entry` → `hidden: true`.

  A value the head's entry (or `[cards]` line) already has is no change: no op, no restamp of `_by`, no commit. An entry
  holding PSD fields keeps its `_by`/`_at` when a Discord edit sets none of them (an alias, a folder field): the bake
  credits the entry's `_by`, and a `discord:` one gets no `Edit-by` trailer, so the PSD field's editor keeps the credit.
  The bot applies the change in memory at once (`CardEdits.apply`), so `/query` shows it before the run lands; it
  saves no stats.pkl (a restart loads card_data/, which has the edit once a run built it). A card with only
  a PNG gets the entry in the file (the website shows it) but no database row.
- **`/mass_replace`:**
  - `author` → right-hand sides rewritten in Card Authors.txt plus a `[people] old = new` line;
  - `series`, `subseries`, `subtype`, `card_type` → one `Card Edits.json` entry per matching card, all in one commit.
- **`/update_tts_save` and `/bake_cards`:** they stay as hand fallbacks that commit directly, as today. With `ACTIONS_TOKEN`
  they could dispatch Card sync instead, keeping one writer; that is a later, small change.
- **`/push`:**
  - until step 10 it replies "Card data is built by the Card sync Action: <link>";
  - at step 10 it is removed (`help.py`, `main.py`; the change-control slash-command checklist);
  - `/update` keeps its pull, Cython rebuild caveat and restart, minus the pickle push.
- **`/pull` (update_bot.git_pull):** the "keep our pickles" conflict rule goes, because the bot no longer writes those files.
- **`/fonts`:** it reports from `card_data/psd_details.json` and no longer refills it, because the builder owns that file and
  fills `supplied` with the fonts present.
- **Reminders:** remove 054b5933 (the 05:30 /push) and d2664752 (05:00 /update_stats); add the interval sync above; keep
  134e3b24.
- **Unchanged:** emojis and the history clone.

**As built (2026-10-22, AutoCroissant work/cutover-bot; commands/card_sync.py, `card_db.sync`; run-and-operate §12).**
- **The switch.** `CARD_SYNC` in global_config.py, `False` until the cutover commit sets it `True`: nothing to remember and
  no runtime flag. Not the data branch's presence: step 7 creates `data` while the jetson still builds. Every switched path
  reads `global_config.CARD_SYNC` at call time; the slash texts are picked at import (the deploy restarts). With it off the
  bot is unchanged: the full suite, the rehearsal harness (byte-identical pickles and logs against work/history-fixes,
  twice) and a diff of every slash text, /help category and /website reply.
- **The reader** is as designed, plus `card_data/sync.json` (the data commit held and the request's ETag, so the first sync
  after a restart is a 304 too). Each downloaded file is parsed before any is written, the manifest and `sync.json` last;
  a copy that no longer matches its manifest is fetched again without `If-None-Match`. Startup loads `card_data/` first,
  then syncs; with neither (GitHub down at the first boot), the frozen root pickles.
- **The reply** counts changed, new and archived cards against the database the bot held (several runs between two syncs
  count together; duplicates left out), and `output_problematic_cards` shows `manifest.problems` only when new data loaded
  (the problems of the last run only). The status lines cost two requests (the run list, a compare of `card_commit` with
  main, on card input paths only).
- **Reminders are quiet:** a reminder's plain `/update_stats` replies to the console only, unless the sync fails or new data
  has problem cards, so the 30-minute reminder posts nothing on a normal day.
- **Dispatches** pass `dry_run=false` explicitly (so they are live before and after card-sync.yml's default flips) and
  `full=true` or `cards`. The run is taken from the dispatch reply (`return_run_details`; a 422 naming it is sent again
  without it) or else is the newest dispatched run not listed before. A cancelled run is dispatched once more only when
  the newer pending run is a scheduled one (ours does its work too) or nothing newer is pending; when another dispatch is
  pending it is said (dispatching again would cancel that one), and one seen running, or with nothing newer at all, was
  cancelled by hand. Card sync's own retry (main moved during the run: a green run with its data dropped, then a
  dispatch by github-actions[bot]) is followed too; it never carries `full`, so the reply says to ask again. One
  followed run at a time. Progress edits stop after 14 minutes (the interaction's 15); a reminder's run gets none.
- **Discord edits** keep committing and applying in memory, but no longer save stats.pkl. A sync of a run that started
  before the edit's commit shows the card without it until the next run (30 minutes at most), as does a restart.
- **/fonts** reports the fonts the build marked `supplied: false` in `card_data/psd_details.json`; a PSD without a record or
  with fonts not judged is an error line. Nothing is downloaded.
- **Not built:** the reminder swap and `ACTIONS_TOKEN` (the owner's, step 8); step 10's deletions; the TTSCardMaker side
  (site.yml's data checkout, card-sync.yml's step 8 switches).

## 12. Changelog, Database edits, analytics, `/query as_of`, archive

### 12.1 Changelog

**Kinds** (from before/after and the index):

| Kind | When |
|---|---|
| `new` | a name not seen before |
| `restored` | an archived name whose PSD is back |
| `changed` | data fields changed and the blob changed: the PSD's own (its parse with `parsed_values` put back) |
| `art` | the blob changed, the data didn't |
| `reparse` | same blob, newer parser, data changed; summarised as one line |
| `moved` | only the path changed |
| `renamed` | archived A plus new B with the same blob |
| `archived` | the PSD left the card repo |
| `edited` | a `Card Edits.json` entry changed (`edit_by` from `_by`); also beside a re-read PSD's `art` or `changed` in the same build (2026-10-23: Metal General and Masked Dedede had posted as PSD changes) |
| `authors` | only the author changed through Card Authors.txt |

A duplicate's change is listed under its original.

**One post per push.**
- **Which pushes.** Each push in the run's batch that brought at least one event gets one message, oldest first. A push with
  no card events (site code, the save alone, `Types/` art with no data change) gets none. The run's own commit M is a push:
  baked edits post as "Website edits baked into the PSDs". A reparse posts as one message of its own.
- **The message.** Titled with who pushed and when, for example "Zomoroko, Oct 22 14:05: 1 new, 2 changed", linked to the
  push's compare view (`before...after`). It uses the website's card-kind words (`KIND_LABEL` in `site/assets/app.js`: Creature,
  Minion, Item, Field card, N.M.E. item, Debuff, Buff…), never invented ones (memory `game-terminology`).
- **Its parts:**
  - New cards: "Name (Creature, 3★) by <author>".
  - Changed: "Name: HP 5 → 6; ability changed".
  - New art.
  - Moved and renamed: "Old is now New".
  - Archived: "Name (PSD deleted)".
  - Restored.
  - Database edits: "Name: subseries → … (by …, on the website / via Discord)".
  - One line for parser re-reads.
- **Links.** Each card links to `SITE_URL/c/<slug>/`, the slug found by the site's `key` (the database name; `name` is the
  shown title); when the card has no slug yet (a new card the site has not built), the link falls back to `#/cards?q=`.
  A rulebook page (not in `cards.json`) links the rulebook searched for it, `#/rules?q=` (2026-10-23).
- **Limits.** Discord allows at most 10 embeds, 4,096 characters per description and 6,000 characters per message in all.
  Anything past that in one push becomes "N more on the website" (`#/changes/changelog`). At most 20 messages per run, about
  one a second (webhooks allow roughly 5 per 2 s); the rest wait for the next run, through the cursor.
- `allowed_mentions: {"parse": []}`.
- **Failure alerts** go to the same channel (§6.3, step 9).

**Website, the Changes page tabs:**
- **Overview:** today's panels.
- **Changelog:** events grouped by day and push for the last 180 days, field diffs shown (`diffWords` for ability text),
  filterable by kind. It also lists archived cards.
- **Database edits:**
  - the `edited` and `authors` events;
  - the website publish batches build_site already computes from the `Card Edits.json` and `Card Authors.txt` history;
  - commits whose subject starts "Discord:", marked "via Discord".

  The "Edited on the website" panel moves here.

### 12.2 `/query as_of:<YYYY-MM-DD>`

- **The time.** The end of that day in `TIMEZONE`.
- **Which version.** Pick the version with `from ≤ t < to`, or the current card when `t ≥` the last version's `to`.
  - Before the first stored version: "no stored version is that old; the oldest is from <date>" (old_stats may not hold the
    card's first version, so "not made yet" would overclaim).
  - When archived and `t ≥ archived.at`: "archived on <date> (PSD deleted in <commit>)", showing the last version. A tag
    with no `at` counts as archived after its last version only.
- **The image** of that day (2026-10-23): the newest commit of the card's PNG before the end of the day, from the bot's
  history clone (`repo_history.commit_before`: one `rev-list` over trees, no blob). The current card posts it pinned to
  that commit with "Image: as of <date> (it has changed since)" when the PNG changed later (a carried start reaches back
  before today's re-render); an older version adds an "Image (<date>)" button. Without the clone, a day before the
  current card's own commit says "Image: today's (that day's is not known)". `version.png_commit` (the PNG at `to`)
  stays for the website.
- **Legacy versions** say "(date approximate)".
- **It ships first (step 3) on today's old_stats.** There, `from` is the stored timestamp and `to` is the next version's.
  Legacy `/update_stats` and a Discord edit's in-memory refresh record `edit` versions since 2026-10-23
  (`card_history.record_edits`: an applied edit that changed a data field keeps the card before, the card's version
  starting at the edits file's newest commit, else the run's time), so the reply says "Database edits (website,
  /update_metadata) are versions of their own only since late October 2026: an older version shows those it had when it
  was replaced." An older version shows its stored data and, with the history clone, an Image button.
- **The live check card** is Tainted Bethany: 4 old versions dated 2026-01-30, 02-07, 10-19, 10-19 in `TIMEZONE` (01-31,
  02-08, 10-19, 10-19 in UTC). Only the fourth has a blob, so the first three say "(date approximate)". Versions 3 and 4 share
  a timestamp, so version 3 is never picked until the seed pass drops duplicates.
- **Archived cards** are found only through their tag. On today's pickles ZOEY!!!!! is unlisted but untagged, so
  `/query ZOEY!!!!! as_of:…` fuzzy-matches another card until the jetson's next /update_stats backfills the tag.

### 12.3 Balance analytics (the Stats page and Details)

- **Who computes them.** `site/build/analytics.py` (new), called by build_site. The site is the only consumer, so there is one
  place to compute and nothing for the bot to keep in step. The build only has to supply correctly dated versions.
- **Cohorts:** creatures and minions by stars (387 cards with all four stats, §2). A cohort under 5 cards is folded into its
  kind's "all stars" cohort. The 7 auxiliary creatures (`creatures`) count per creature.
- **`data/analytics.json`:**
  ```
  {cohorts: {"creature|3": {n, hp/def/atk/spd/total: {p10, p50, p90}}},
   cards: {<id>: {cohort, pct: {hp, def, atk, spd, total}, z_total, flags, trend: [{ts, hp, def, atk, spd, total}]}}}
  ```
- **The Stats page** gets a Balance section: one quantile strip per cohort and an outliers table (top and bottom 10% of total).
- **A card's Details** get a Balance line ("total in the top 10% of 3★ creatures") and a sparkline of the trend from its
  versions.
- **Ships first (step 3)** from the pickles the site reads today. Load the `dataviz` skill before writing the charts.

### 12.4 Archive

- **Done.** `work/archive-deleted` merged into main (dfdf953, with the review fixes 9ae19f9 and 2167d81) and is deployed. The
  builder runs its code as is: `prune_clean_cards(at)`, `restore()`, `archived()`, old_stats saved before stats, and
  psd_details keeping archived blobs.
- **What the build adds:**
  - git dates and commits on every tag;
  - `renamed_to` from exact renames across runs;
  - gate G2, which caps archives per run.

## 13. Gates, failure modes, rollback

**Gates (in card_data; a refusal exits 2, commits nothing to `data` and uploads the would-be db/):**

| Gate | Refuses when | Lifted by |
|---|---|---|
| G1 floor | fewer than 100 cards (build_site's `MIN_CARDS`), or fewer than 95% of before's | nothing; fix the cause |
| G2 archive ceiling | more than max(5, 1% of cards) archived in one run | `allow_archive` |
| G3 round trip | any record changes through record → CardInfo → record; psd_details doesn't parse; a written file's sha256 differs from the manifest | nothing |
| G4 same-blob re-parses | more than 25 cards re-parsed with an unchanged blob | `full` (approval kept in `pending.full`) or naming them in `cards` |
| G5 quality | UNKNOWN card_type up by more than 3; more than 3 cards lost their ability text; more than 3 lost every stat; psd_details records down by more than 5% | after review: `cards` for the listed cards, or `full` |
| G6 freshness | the `data` tip moved since checkout (another writer) | the next run, automatically |

A gate refusal also skips the main commit when the run baked, so a bake is never committed without the data that reads it;
the save then waits too. Without a bake, the save still commits on the last good data.

**Failure modes**

| Case | What happens | Recovery |
|---|---|---|
| Build error (exception, API 5xx, a rate limit early in the run) | Red; no data commit; the last good data keeps being served; the save still runs on it; one failure alert per streak | The next run (gate a), or a dispatch |
| Bake error | A card's error keeps B for the cards that baked; with no B the run goes on from main's head; the report and crops are in the artifact; alert | Fix, then the next run bakes |
| One unreadable PSD | Hook (c): counted in `pending.failed`; old data kept; no archiving that run if it was a new name; card-check already pinged the card makers | Fix the PSD; the next run builds it |
| Gate refusal | Red; artifact holds the would-be db/ and the diff; alert | Review, then dispatch with the gate's input, or fix |
| Wrong text that passes the gates | Visible in the commit's per-card lines and the job summary | `restore` (below) |
| A parser regression pushed to AutoCroissant | G4 blocks a mass re-parse; per-card damage on newly pushed PSDs is possible | Pin `CARD_DATA_BUILDER_REF`; `restore`; dispatch `cards` after the fix |
| The builder's import closure broken (a new top-level import) | Red at `--help` time; nothing committed | `test_card_data.py` catches it before the push; pin the ref meanwhile |
| Main moved during the run | M, the posts and the data are dropped; one retry is dispatched unless a run is pending (§6.3, stage 4); a second rejection only warns | The retry, else the next run |
| Push to `data` rejected | Red; posts already sent may repeat on the next run | The next run rebuilds on the new tip |
| Webhook down | Warning; cursor kept | The next run posts the waiting pushes |
| A scheduled run late or skipped (GitHub delays crons under load) | The next run does the work; nothing is lost because the gate diffs against the last success | None needed; a dispatch if urgent |
| Actions outage | The bot serves `card_data/`; the site can't deploy either (Pages builds in Actions) | The same builder by hand on the Mac (below) |
| `ACTIONS_TOKEN` expired | `/update_stats force_update` replies with the link and the gh command | Rotate it |
| Tampered `data` commit | JSON only, so no code runs; the next build carries the forged values forward | `restore`; the ruleset (to confirm) |

**Rollback of data.**
1. Dispatch `restore=<good data commit>`. The run writes that commit's `stats.json`, `old_stats.json`, `psd_details.json` and
   manifest (`restored_from`) as a new commit, never forced. It keeps `changelog/` and appends a `restore` event, and the
   posted cursor never goes back.
2. **Before that, stop the cause.** Restored data has the old blob shas, so the next run would re-parse the same PSDs with the
   same code. Pin the builder ref, or run `gh workflow disable card-sync.yml -R MichaelJSr/TTSCardMaker` while fixing (that
   also stops the bake and the save; `/update_tts_save` covers the save meanwhile).
3. Re-enable, then run `/update_stats` on the bot. The site is dispatched by the restore run.

**By hand on the Mac** (an Actions outage, or a free full re-parse):
```
git clone -q --bare --filter=blob:none https://github.com/MichaelJSr/TTSCardMaker /tmp/tc.git      # 1.1 MB, 2 s
~/.venvs/ac-py314-psd125/bin/python -m commands.card_data --repo MichaelJSr/TTSCardMaker --checkout /tmp/tc.git \
  --blobs ~/Desktop/TTSCardMaker --db /tmp/carddata/db [--full] --summary /tmp/card-data.md
```
- `--checkout` gives the tree and the index, and must not be shallow: the builder refuses a shallow clone, as build_site does.
  The Mac's working clone is shallow (checked 2026-10-22), so it serves only as `--blobs`, the object store tried first for
  every blob by sha. The same gates apply.
- Publishing: a plain `git push` of `/tmp/carddata` to `data`, with the owner's own credentials. The next scheduled run sees
  G6 only if it was running at that moment.

**Rollback of the whole move (until step 10):**
1. Revert the site.yml cutover commit. The site reads AutoCroissant again.
2. On the Mac, run `python -m commands.card_db export-pickles <data tip> . --git <a checkout of data>`, commit the three
   files it writes (`stats.pkl`, `old_stats.pkl`, `psd_details.json`; `aliases.pkl` is not in db/ and stays) as `PICKLE` to
   AutoCroissant and push. The jetson then resumes from the Action's latest data, so nothing is lost.
3. Revert the bot cutover commit (`CARD_SYNC` back to `False`), deploy it (pull, `build_ext`, restart), and re-add the 05:00
   /update_stats and 05:30 /push reminders. Before the revert, `python -m commands.card_db export-pickles card_data .` on the
   jetson gives the same pickles step 2 commits (its `card_data/` is the data tip's db/).
4. Disable card-sync.yml and restore tts-save.yml and card-edits.yml from history (one revert of the cutover's TTSCardMaker
   commit).

## 14. Migration

Each step lands on main on its own, keeps the full suite green, and is recorded in the plan. [owner] marks the owner's part.

1. **Per-PSD failure isolation (hook c) in `repo_traverser.py`.**
   - *What:* an exception in `parser.parse` or `_gather_details` for one PSD prints a line, counts the card as failed
     (`failed_new` for a new name), keeps its data, and skips saving the ETag.
   - *Verify:* a new test (an unreadable PSD among two good ones: 2 parsed, `failed == 1`, nothing archived); the jetson deploy
     (pull, `build_ext`, restart); the next 05:00 run is normal in `journalctl -u autocroissant`.
   - *Undo:* revert the commit and redeploy.
2. **Merge `work/archive-deleted` into main. Done** (dfdf953, 2026-10-22, deployed).
   - *Still to check,* on the jetson's next /update_stats: "Archived 1 card deleted before this update: ZOEY!!!!!" and nothing
     else archived; before that /push, diff_stats shows 0 cards removed from stats and inspect_pickle counts 1 archived.
   - *Undo:* revert the merge. Revert the PICKLE it produced, a pickle-only revert, which is the change-control exception, after
     diff_stats.
3. **Ship value from today's data** (independent; any order).
   - *What:* the Balance analytics (§12.3), then `/query as_of` (§12.2).
   - *Verify, analytics:* the site build is green; the cohort sizes in `analytics.json` equal §2's counts; three cards checked by
     hand.
   - *Verify, as_of:* `test_query_as_of.py`; live `/query Tainted Bethany as_of:2026-02-01` shows version 1 "from 2026-01-30
     (date approximate)" (Pacific; 2026-01-31 in UTC). An archived-card check (`/query ZOEY!!!!! as_of:2026-07-01`) waits
     for the next /update_stats, which backfills its tag.
   - *Undo:* revert each commit.
4. **Move the data path into `commands/card_data.py`, and add `card_history.py`, `card_db.py` and `changelog.py`;** add
   `card_bake --local` and `tts_save --names`. `repo_traverser.update_stats` calls `card_data.build` in legacy mode (GitHub
   source, root pickles, the bake lock, Discord messages).
   - *Verify:*
     - the full suite, plus the new tests in §10;
     - `--help` of the three CLIs in a clean venv holding only `requirements-card-data.txt` (the bake's with its fork pin;
       since 2026-10-23 both hold `requirements-cards.txt`);
     - on the Mac, in a sandbox copy, an incremental legacy run on fixture pickles writes byte-identical pickles to the old code;
     - `tts_save --names` on a copy of the save with a copy of today's data gives the same Nicknames as the live `cards.json`;
     - jetson deploy; the next 05:00 run reports as before;
     - its PICKLE passes diff_stats with 0 unexplained differences.
   - *Undo:* revert (code only) and redeploy.
5. **Discord writers and the aliases** [owner: decisions 2 and 3].
   - *First, the parity re-parse.* On the Mac, a full re-parse by hand with step 4's builder (§13, "By hand on the Mac"; local
     blobs, no API) is compared by diff_stats (JSON input) with main's stats.pkl.
     - Predicted: data-field differences only where /update_metadata or /mass_replace set a value after the card's last re-parse
       (a re-parse would lose those values today), plus any PSD pushed since the jetson's last run.
     - Any other difference (pure Python vs Cython, psd-tools) stops the migration until it is explained.
   - *What:*
     - `/update_metadata`, `/mass_replace` and `/alias` commit to the card repo (§11);
     - one migration commit moves the 60 aliases into `Card Edits.json` `aliases`, and carries the Discord-only values the
       re-parse listed, so nothing typed in Discord is lost. `python -m commands.discord_edits migrate --clone <full clone>
       --ref <main> --aliases aliases.pkl --stats stats.pkl --out-dir DIR` writes that commit's files (it has no alias-only
       mode): the aliases, the values a re-parse keeps from the database (series, subseries, author) that no card-repo input
       gives (a field `Card Edits.json` already sets is not carried), and the top-level key `aliases_moved`, after which the
       bot ignores aliases.pkl. Measured 2026-10-22 on the parity seed at af826714: 60 aliases onto 53 cards, 596 series
       values (the parity's 624 minus the 28 that `Card Edits.json` already overrides), 0 authors.
     - *Consumers* (TTSCardMaker), in the same sitting as the migration commit; the `discord:<user>` display (the third
       item) lands before, or with, the jetson deploy of the writers, which stamp `_by: discord:<user>` from the first
       `/update_metadata` or `/alias` on:
       - build_site skips aliases.pkl when `aliases_moved` is set; otherwise the site keeps the pickle's 60 aliases (the
         bot never rewrites aliases.pkl) and drifts with the first `/alias` move (TTSCardMaker `work/alias-migration`
         4c728f15 on 5dd2a408 has this, with the migration commit itself);
       - the site's "edited" marks leave out every `_by: AutoCroissant` entry, not only alias-only ones (the 596 carried
         series values are no website edits; nor are the two `allow_high_stats` entries of 2026-10-21). The migration
         commit is authored `AutoCroissant <autocroissant@gmail.com>` (`migrate` prints the `git commit --author` to use),
         so build_site's history counts it for nobody. A carried value merged into an entry a website editor made (14 entries at
         5dd2a408: 12 series, 5 alias lists) keeps that entry's `_by`/`_at` (an entry has one stamp; the bake credits
         it) and is listed in the entry's `_bot`, so the site credits it to nobody. An edit that sets or unsets a field
         takes it out of `_bot`, and an editor's edit to a `_by: AutoCroissant` entry first lists that entry's fields
         in `_bot` (edit.js `editsFile`, `discord_edits.edits_file`), so the editor gets only what they change;
       - a `_by` or `Edit-by` of `discord:<user>` is shown "via Discord", never linked as a person (`#/u/discord%3A…`;
         app.js's "Edited on the website" panel and Edited tip, and build_site's shallow-clone contributions fallback,
         which calls `rec_of(key_of(by))`). `work/changes-tabs` does it for the Database edits tab.

     This works with the jetson's pipeline, which already applies `Card Edits.json`.
   - *At landing,* regenerate the migration commit: `migrate` against the TTSCardMaker head of that moment and the jetson's
     pickles, in the same sitting as the deploy. A field `Card Edits.json` already sets is not carried, and a live `/alias` or
     `/mass_replace` since the branch's commit would otherwise be missed or clobbered.
   - *Verify:*
     - `test_metadata_edits.py`;
     - every one of the 60 aliases resolves to the same card in `/query` and in the site's `cards.json` aliases as before;
     - the duplicates' display names are unchanged in the next PICKLE (diff_stats);
     - a second Mac re-parse shows 0 data-field differences;
     - one live `/alias` makes one commit with its `Edit-by` trailer.
   - *Undo:* revert the code. The `Card Edits.json` entries are valid data either way; revert that commit too for a full undo.
6. **card-sync.yml, dispatch only, `dry_run` defaulting to true** (no schedule yet). tts-save.yml and card-edits.yml keep
   running beside it.
   - *Runs:*
     - (a) an incremental dry run seeded from AutoCroissant's latest PICKLE, all stages;
     - (b) `cards: Mini Doomer, The Freezer, The Lich King` (the certified goldens), to prove the runner parses as the Mac does;
     - (c) a bake dry run (the fresh dry run the lifted hold asks for), compared with card-edits.yml's dry run at the same head.
   - *Predicted before running:*
     - (a) 0 PSDs to re-parse when the jetson is current; seed pass −10 duplicate versions at 3888392 (−15 on PICKLE 077337a, which holds 475 versions), 31 cards re-dated, 3
       versions re-dated, 291 marked legacy, archive tags dated; 0 data-field differences; 0 `supplied` changes; the save
       report equal to tts-save.yml's at the same head; `discord.json` empty, because the seed build writes no events (§8).
     - (b) the three cards equal their goldens field for field.
     - (c) the same cards, verdicts and crops as card-edits.yml's dry run.
   - *Verify:* each prediction against the job summary and `diff_stats.py` (JSON input) vs main's stats.pkl. Any other difference
     stops the migration until it is explained.
   - *Undo:* delete the workflow file.
7. **Shadow** [owner: decisions 1 and 5].
   - *Precondition (history recovery):* the seed PICKLE is one the jetson pushed after its first `/update_stats` with
     work/history-recovery (its line "History: N versions recovered from the bot's PICKLE commits ..."). The runners'
     checkout of AutoCroissant is shallow (actions/checkout's fetch-depth 1), so the seed pass there never recovers: its
     report says "Recovery from the bot's PICKLE commits: skipped (no clone of the bot's repo)", and the website gets the
     recovered history only through that PICKLE. Seeding from an older PICKLE loses it until a re-seed.
   - *What:* create the `data` branch with one fresh live dispatch (`dry_run=false`; with no branch yet the workflow starts an
     empty orphan and seeds from AutoCroissant's latest PICKLE, or `seed`), then add the schedule. The workflow cannot seed
     from an artifact, so the live run's `db/` is checked against run (a)'s artifact instead (same seed, same counts; only
     PSDs pushed in between may differ). `MAIN_COMMIT` and `POSTS` stay off until step 8 (`discord.json` goes to the artifact
     only), so tts-save.yml and card-edits.yml stay the only main writers and nothing posts twice. The jetson keeps building and pushing pickles; the
     site still reads AutoCroissant.
   - *Verify, daily, for 3 days or 10 card pushes, whichever is later:*
     - `diff_stats.py` between the `data` tip's stats.json and AutoCroissant main's stats.pkl, after the jetson's /push, shows
       only the predicted refinements (the command and the refinements measured on a Mac rebuild: the diagnostics skill's
       Workflow 4);
     - build_site run twice (with and without `CARD_DATA_DIR`) gives byte-identical `cards.json`, apart from the predicted dates:
       the re-dated cards and their duplicates, and the history entries of the deduped and re-dated versions;
     - the changelog's events, grouped by push, match the card pushes of those days;
     - most scheduled runs end at the gate (about 15 s).
   - *Undo:* `gh workflow disable card-sync.yml`; delete the `data` branch.
8. **Cutover, in one sitting.**
   - *First:* reconcile the jetson. Its `git status` shows no modified data file, and its last /push is in origin. Otherwise
     /push and re-seed with `seed=<that PICKLE>` (a PICKLE after the recovery run, step 7's precondition). [owner, Discord] `/remove_reminder 054b5933` and `/remove_reminder
     d2664752`, then add the interval `/update_stats` reminder; [owner] add `ACTIONS_TOKEN` to the jetson's config.py
     (decision 4).
   - *Then:*
     - site.yml: the data checkout, `CARD_DATA_DIR`, the daily cron;
     - card-sync.yml: `MAIN_COMMIT` and `POSTS` on (the bake, the main commit, the posts and the failure alert); `dry_run` defaults to false; `posted.json` set to the last
       shadow event, so Discord starts with the pushes after the cutover (the shadow period stays on the website's
       Changelog);
     - delete tts-save.yml and card-edits.yml in the same TTSCardMaker commit;
     - the bot cutover commit: `CARD_SYNC = True` in global_config.py (the reader, `/update_stats`, `/push`, `/pull`,
       `/fonts` and admin-only `force_update` are on main already, switched off: work/cutover-bot);
     - jetson deploy.
   - *Verify:*
     - the site footer reads "Card data built … from TTSCardMaker@…", with the same card count and no new warnings;
     - `/update_stats` says "already current";
     - `/query` on the three goldens;
     - the next card push lands on the site, in the save and in Discord within one interval plus one site build (§15), with
       one post for that push and at most one Card sync commit on main.
   - *Undo:* rollback of the whole move (§13).
9. **The Changes page tabs.**
   - *What:* the Changelog and Database edits tabs (§12.1), from `data/changelog.json`.
   - *Verify:* the tabs list the shadow-period and post-cutover events, grouped by push as the Discord posts were.
   - *Undo:* revert the site commit.
10. **Retire, within a day of clean runs** [owner: decision 6, 2026-10-22; tell bobberlington].
    - *What:* delete AutoCroissant's root `stats.pkl`, `old_stats.pkl`, `aliases.pkl` and `psd_details.json`, the legacy mode,
      the commits-API path and `/push`. Point `conftest.real_stats` at `db/stats.json`. Update the docs and memory:
      - change-control: the data class is written by the Action; the gate is the data commit review;
      - run-and-operate: `/update_stats`, `/push`, reminders, `card_data/`, the TTS save now from Card sync;
      - architecture-contract: pickles-in-git becomes data-in-git, JSON;
      - config-and-flags: `ACTIONS_TOKEN`, `CARD_DATA_DIR`;
      - ttscardmaker-companion: the data flow, Card sync in place of tts-save.yml and card-edits.yml;
      - impossibility-cards-reference: the db/ schemas;
      - memories `jetson-server`, `ttscardmaker-integration` and `next-session-plan`.
    - *Verify:* the suite is green; `/help` lists no /push; a full site build is green.
    - *Undo:* revert. The files come back from history.

## 15. Runner and API cost

| Item | Cost |
|---|---|
| A run that ends at the gate | about 15-20 s; most of the 48 runs a day (24 at hourly), about 15 runner-minutes a day (8 at hourly) |
| A working run (k changed PSDs) | about 2-4 min: the checkouts and the sparse builder (~10 s), Python and pip from cache (one environment, ~15-20 s on a hit), the fonts cache restore (215 MB, ~15 s), ~2-4 s per PSD, the save (~1 min, as tts-save.yml), the posts and commits (~10 s); plus the bake's own time per card when edits are pending |
| Working runs per day | one per interval with a card-input change: recently 2 / 5 / 2 (§2), against 2 / 8 / 4 builds if every commit built |
| Commits on main from Card sync | at most one per working run, only when a bake or the save changed something: recently about 0 / 2 / 1 a day (the PNG windows, §2), plus one re-pin commit after each baking run. Before: tts-save daily plus one commit per baked card |
| Commits on `data` | at most one per working run |
| Discord posts | one per push with card events (recently a few a day); at most 20 per run |
| A full re-parse | 813 downloads in three budgeted runs, about 8-15 min each (card-check's sequential whole-tree pass takes 25-30 min) |
| site.yml | as often as now for pushes; plus one dispatch per committing run; the cron drops from 4 to 1 a day |
| Price | standard runners are free on a public repo |
| API (GITHUB_TOKEN, 1,000 requests an hour per repo, shared with card-check and font-check) | per working run: k blob downloads (none for local blobs), 1 activity read, 1 contents read; per gate-only run 1 contents read; a full re-parse is budgeted at 400 per run |
| API (the bot) | one conditional request per sync (a 304 is free); the raw files are not API calls |
| Storage | the `data` branch grows by packed deltas of a few KB to tens of KB per data commit; AutoCroissant stops growing by about 10-40 KB per PICKLE |

**A card push reaches the website's data, the TTS save and Discord** within one interval (30 minutes, or an hour) plus about
3 minutes of run and one site build. The push's images and website edits still reach the site at once through its own push
trigger.

## 16. Owner decisions

Answered on 2026-10-22 and folded in above: the changelog posts one post per push, with failure alerts in the same channel;
one batched Card sync run does the bake, the build, the save and the post; the archive is merged and deployed. The six below were answered on 2026-10-22 (03:50 and 04:30): all as recommended,
except 6, which is one day, not 14:

1. **Where the data lives.** Answered: yes, the orphan branch. Recommended: an orphan branch `data` in TTSCardMaker, an approved exception to "commit to main;
   delete spare branches". It keeps main's history for people: Card sync touches main only for a bake or the save. The
   fallback, `data/db/` on main, is now cheap too (it rides the run's one main commit), at the cost of a main commit on every
   run that changed data (§5).
2. **Discord edits become card-repo edits.** Answered: yes.
   - `/update_metadata` writes `Card Edits.json` and `Card Authors.txt`: ability, stars, types and stats become "psd"-kind edits
     that persist, show "PSD pending" and are baked by the next Card sync;
   - `del_entry` becomes `hidden: true`;
   - the Discord-only values found at step 5 are carried into `Card Edits.json`.
   - *Not as answered, to put to the owner:* stars and types were refused from Discord until the bake printed them (§11);
     answered 2026-10-23: the bake prints them, and `/update_metadata` takes them again. The carry is 596 series values
     (`Card Edits.json` 43 entries to 632); confirm before TTSCardMaker lands.
3. **Aliases move from aliases.pkl into `Card Edits.json` `aliases`** (60 today), and `/alias` commits there. Answered: yes (aliases too).
4. **`ACTIONS_TOKEN` on the jetson** (fine-grained, TTSCardMaker only, Actions read and write), so that `/update_stats
   force_update` and `cards:` can start a Card sync run. Answered: yes; the owner creates it and puts it in the jetson's config.py. Without it those options reply with a link and the
   gh command.
5. **30 or 60 minutes.** Answered: 30. A push waits at most half an hour; the cost is about 7 more runner-minutes a day of
   gate-only runs (free on a public repo) and no more commits, because commits follow changes, not runs. Hourly halves the gate
   runs and makes a full re-parse take two hours instead of one. The bot's sync reminder follows the same interval.
6. **Within a day of clean runs (owner; 14 days was the proposal), delete the pickles, `psd_details.json` and `/push` from AutoCroissant main** (bobberlington's repo:
   tell him first). Answered: yes.

7. **More on `data` (2026-10-23 ~18:30).** Answered: yes to both.
   - **Bake verdicts** move out of `site/Card Edits.json` (the `_psd` blocks: baked, refused, redundant, the reason, the
     PSD blob) into `db/verdicts.json` on `data` (`{"schema": 1, "cards": {key: verdict}}`, one card per line). Card
     Edits.json keeps only people's edits; a refusal makes no main commit; a bake commits to main only the PSD, the PNG and
     the edit fields it drops as printed. Readers: card_bake (`final()`, `--pending`), build_site (PSD pending, refused),
     the bot's `/query` facts, card-sync's gate. Migration: one data commit with the current `_psd` blocks, then one main
     commit removing them from Card Edits.json.
   - **Bake and guard events** join the changelog: `baked` (fields, PNG changed), `bake-refused` (reason), `guard-png`
     (a PNG the guards re-made or recompressed), `guard-fix` (a Txt2 or other fix the guards landed). The Changes page shows
     them; the Discord post stays one post per push (they ride the push's post).
   Built after BUG 2 (`work/fx-rebuild`) lands, as part of steps 7-9.

## 17. To confirm (settled by a test, not by the owner)

- Whether a ruleset on a personal public repo can let only GitHub Actions update `data` (§5).
- That setup-python 3.14 with the fork pin installs on ubuntu-latest. 2026-10-23: requirements-cards.txt resolves for Linux
  x86_64 3.14 with wheels for every package but the fork (`uv pip compile --only-binary :all: --no-binary psd-tools`); the
  fork builds from source there as on the jetson. Since 2026-10-24 pyoxipng is in too there, built from its sdist (Rust:
  19.5 s on the Mac with every crate downloaded; PUSH_GUARDS.md section 20). The first cold run on the runner confirms both.
- That every HEAD PSD blob is in the Mac's working clone, for the by-hand full re-parse (one PSD checked 2026-10-22).
- How build_site's contribution counts treat a `_by` or `Edit-by` naming a Discord user rather than a GitHub login. The proposal
  is `discord:<name>`; it should be skipped or mapped, never given a person page.
- That `raw.githubusercontent.com` downloads at a commit sha could replace the REST blob downloads for a one-run full re-parse
  (their limits are not documented); later, optional.
- That the activity API lists a website edit's commit as its own push, as it does the bot's API commits (§2).
- That `tts_save --names` from `db/stats.json` gives exactly the names `cards.json` gives (step 4 checks it on a copy).

**Files referenced.**
- AutoCroissant (`~/Desktop/AutoCroissant`):
  - `commands/repo_traverser.py`, `card_models.py`, `card_edits.py`, `card_bake.py`, `tts_save.py`, `git_commit.py`,
    `github_api.py`, `psd_parser.py`, `psd_details.py`, `psd_analyzer.py`, `query_card.py`, `update_bot.py`, `font_check.py`;
  - `global_config.py`, `main.py`, `setup.py`;
  - `tests/conftest.py`, `tests/test_update_stats.py`, `tests/test_card_bake.py`;
  - `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py` and `inspect_pickle.py`;
  - new: `commands/card_data.py`, `card_history.py`, `card_db.py`, `changelog.py`, `requirements-card-data.txt` (now
    `requirements-cards.txt`); at step 8
    `commands/card_sync.py` (the bot's side) and `tests/test_card_sync.py`.
- TTSCardMaker (`~/Desktop/TTSCardMaker`):
  - `.github/workflows/site.yml`, `card-edits.yml` and `tts-save.yml` (both folded into card-sync.yml), `card-check.yml`,
    `font-check.yml`;
  - `site/build/build_site.py`, `site/assets/app.js`, `stats.js`, `details.js`, `edit.js`;
  - new: `.github/workflows/card-sync.yml`, `site/build/analytics.py`, the orphan branch `data`.
