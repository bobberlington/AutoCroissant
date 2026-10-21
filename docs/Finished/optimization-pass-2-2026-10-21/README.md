# Optimization pass 2 (2026-10-21)

Session checklist item 3: a measured pass over the code written the night before (the edit pipeline `psd_edit`,
`psd_regions`, `card_bake`, the font check `font_check`, the website's badges and build), plus `tts_save` and
`github_api` for the duplication question, `/query_ability` and `/fonts` in `query_card.py` / `main.py`. AutoCroissant
main 74a5c2c (working tree, committed by the orchestrator); TTSCardMaker f5a0e50 (five files: two workflow lists,
`build_site.py`, `app.js`, `edit.js`; no PSD or PNG). Three read-only audits ran first (perf.md, code-health.md,
tests.md), then the fixes, then a review round. bench/ holds the profiling scripts (they stub `config`; paths are this Mac's).

Machine: Apple M2 Ultra, 24 cores, 192 GB, load average 12-25 throughout (a rendering workflow shared the box), so wall
times are pessimistic by 20-40 % and only before/after pairs on the same day compare.

## Numbers

| Measure | Before | After | Note |
|---|---|---|---|
| `font_check --all` over 813 card PSDs (16.4 GB) | 98.3 s wall, 75.1 s user, 1 core | 9.8 s wall, 91.6 s user | process pool from 8 PSDs; `--json` and markdown byte-identical |
| Stock suite (`~/.venvs/ac-py314-psd125`) | 337 passed, 18 skipped, 23.7 s | 350 passed, 18 skipped, 30.8 s | 13 new tests; most of the +7 s is the pool-vs-sequential test and two subprocess import probes |
| Fork run of `tests/test_psd_edit.py` (`/tmp/wt-pipeline-dev`) | 26 passed, 1 skipped | 27 passed, 0 skipped | the `font` refusal now runs (1.1 s); 229 s wall under load vs 118 s at audit, not comparable |
| Commit plumbing (API blob/tree/commit/ref + SSH fallback) | 2 copies, ~330 lines | 1 module, 313 lines | `commands/git_commit.py`; card_bake 966 -> 777 lines, tts_save 1022 -> 891 |
| Duplicated helpers across card_bake / tts_save / font_check | 10 names, 8 constants | 0 | `blob_sha`, `fetch_verified`, token patch in `github_api.py` |
| `card_bake --dry-run` per card (Mini Doomer hp 5 -> 8) | 23.0 s | 23.0 s | 95 % is fork compositing (perf 1, 2): fork-side work, deferred |
| Parser gate | 813 cards, 0 errors | 813 cards, 0 errors, 0 differences | `/tmp/acfix/parse_all.py` vs the audit's snapshot, 16 s |
| Cython build in a /tmp copy | 11 .so | 11 .so, 0 errors; suite 349 / 18 against the .so (before the review round added one test) | `query_card.py` changed (compiled) |
| `import main` with the stub config | 51 = 51 | 51 registered = 51 decorated | |

## What changed

- **One commit path.** `commands/git_commit.py` (uncompiled; closure `github_api`, `repo_history`, `global_config`):
  `api`, `ref_head`, `commit_info`, `upload_blob`, `api_commit`, `git`/`git_env`/`git_tree`/`git_commit`,
  `commit_files` (API, then SSH into the history clone as `card_bake.land` did), module state `cant_write` and `limited`,
  `plural`. `card_bake.land` and `tts_save.update_repo_save` call it. Accepted behaviour changes: card_bake treats
  401/403/404 on blob upload as "use SSH" (was 403 only); tts_save's API commits carry the AutoCroissant author;
  tts_save's API timeout 120 -> 300 s. A rate-limited 403 still raises `Transient` in card_bake and keeps the run on
  SSH in tts_save (`ssh_on_limit`).
- **github_api.py**: `token_from_env`, `blob_sha`, `fetch_verified`; `card_bake._tree` goes through `api` (API-version header restored).
- **font_check**: `check_checkout` parses in a `ProcessPoolExecutor` (half the cores, chunksize 4, from `POOL_MIN` = 8 PSDs),
  merged in path order; `pushed_files` notes per-commit and 250-commit caps; `/fonts` failures print the traceback;
  private helpers renamed (`_get_json`, `_check_one`, `_card_name`, `_font_files`), `Report.as_dict`, truthful `__all__`.
- **psd_regions / psd_edit**: `fx_pad` and `union_box` public; `region_masks` and numpy gone (0 callers); psd_edit's own
  `_union`, the `KIND` getattr fallback, `Edit.report` and save_psd's dead branch deleted (the `_updated` fallback the
  fork needs stays); `_enabled_effects(strict=True)` can refuse `parity` with the cause; Args/Returns on 13 gates and
  helpers; American spelling (`normalize_ability`, `StatusColor`); `__all__` lists what card_bake uses.
- **card_bake**: `classified`, `colors`, Args blocks on `bake_card`, `select`, `rewrite_entry`, `land_verdicts`.
- **query_card.py** (compiled): `/query_ability` with one result and no PNG posts the card's name instead of
  `content=None`, which Discord dropped. **main.py**: `/fonts` lazy import binds the module, no shadowed name.
- **Tests**: `tests/test_git_commit.py` (new); card_bake: forbidden token -> SSH, verdict skipped when the edit changed
  before the verdicts commit, head-moved over 409 and 422, card-edits.yml shape (dry-run default, actor guard,
  `tts-save` group, the `committed \`` grep literal), third-party import allowlist; psd_edit: Royal Eradicator Main
  Cannon's two-row atk pinned, the `font` refusal runs on every fork run; font_check: cap notes, pool == sequential on
  8 real PSDs; query_output: single result without an image; conftest shares `blob_sha_bytes`, `text_engine`, `Reply`.
  `requirements-dev.txt` gains `pyyaml` (not installed here).
- **Docs and skills**: `EDIT_PIPELINE_DESIGN.md` (commit path, closure, `fx_pad`, Action list, decision 2 note);
  dated provenance lines in config-and-flags, run-and-operate, architecture-contract, build-and-env,
  failure-archaeology, analysis-toolkit, docs-and-style, research-methodology, ttscardmaker-companion.
- **TTSCardMaker**: `card-edits.yml` and `tts-save.yml` check out `commands/git_commit.py`; `valid_verdict` accepts only
  `baked` | `refused` (a typo'd verdict hits the existing warning); one `PSD_TIP` string in app.js used by the badges,
  the tile flag and edit.js; the `PSD_FIELDS` comment names its five homes.

## Judged not worth it (or not now)

| Finding | Why |
|---|---|
| perf 1, 2: write the merged PNG as the PSD preview; crop effects to the viewport (`-9 s`, `-10 s` per card) | Fork-side changes; `/tmp/wt-*` worktrees are in use by the rendering workflow. The largest wins of the pass, queued |
| perf 4: `/fonts` fills its cache from the local card clone | Needs the network to verify and changes `/fonts` behaviour; not an identical-results speedup |
| perf 5: cache the website accent colour by blob hash (`-2.9 s` of a 5.3 s warm build) | Worthwhile, outside this pass's TTSCardMaker scope (kept to finding 12) |
| perf 7: open each PSD once per bake (`-0.3 s`, 1.5 %) | Needs a `layer_records` signature change for little |
| health 10: `run(card=)` rename; `font_check._api` -> `api` | CLI contract name with 15 call sites; the general `api` lives in git_commit and would enter the font-check Action's closure |
| health 11: split `run()`, move `EXCLUDE_FOLDERS` to global_config | Refactor without a behaviour gain; the move edits a compiled module for two lines |
| health 9 (`font_of`) | A test pins `font_set[0]` |
| tests 1: `requests` in the fork venv | The owner's environment (no pip in that venv); the docstring names the gap |
| tests 9's module-scoped Doomer bake fixture | Would couple two tests' assertions; the duplicate re-bake and Mini Doomer in NAMED were dropped instead |

## Open items

- perf 1 and 2 on the fork once the worktrees are free: `PSDImage.save(preview=image)` and viewport-cropped effects; a stat bake would land near 3-4 s.
- `~/Documents/Tools/psd-tools/.venv` lacks `requests`: `tests/test_card_bake.py` cannot collect under the fork, so the
  end-to-end bake test still runs nowhere on this Mac.
- `pyyaml` is listed in `requirements-dev.txt` but not installed in `~/.venvs/ac-py314-psd125`; the two workflow-shape tests skip until it is.
- perf 5 (accent colour cache) and hardlinked dist copies on the website build: next TTSCardMaker pass.
- `_enabled_effects(strict=True)` cannot fire today (effects always readable): a harmless guard, kept.
