# Code-health audit (2026-10-21)

Read-only, before any change; line numbers are the audit's. Files: `commands/psd_edit.py` 967 lines, `psd_regions.py`
350, `card_bake.py` 966, `font_check.py` 601, plus `tts_save.py` 1022 and `github_api.py` 96 for duplication,
`main.py:278-285`, `query_card.py:577-619`, and TTSCardMaker `build_site.py` / `app.js` / `edit.js` / `style.css` at
f5a0e50. Suite at the time: 337 passed, 18 skipped.

## Census

| Measure | Count | After |
|---|---|---|
| Helper names defined in 2-3 of {card_bake, tts_save, font_check, repo_history} | 10 (`_git` x3, `_api` x3, `_git_env`, `_status`, `_check`, `_git_commit`, `blob_sha`, `fetch_verified`, `_plural`, `card_name`) | 1 (`card_name`: kept on purpose, see 5) |
| Constants defined 2-3 times | 8 (GIT_TIMEOUT x3, API_TIMEOUT x3, EXCLUDE_FOLDERS x3, GIT_IDENTITY, CONFLICT, COMMIT_TRIES, FONTS_DIR, `_cant_write`) | EXCLUDE_FOLDERS x3 (see 11) |
| Commit-path lines (tts_save / card_bake) | 153 / 175 | `git_commit.py` 313 incl. docstrings and tests' hooks |
| Dead code | `region_masks` 0 callers, `regions` 1 test caller, KIND fallback, 2 no-preview branches, `Edit.report` | deleted except `regions` and the `_updated` fallback the fork needs |
| Defs without a docstring | 44 (psd_edit 10, psd_regions 3, card_bake 20, font_check 11) | the 27 public ones with 2+ params got Args/Returns |
| Broad excepts | 10, 4 hiding a cause | `/fonts` prints the traceback; `_enabled_effects(strict=True)`; the rest kept (an unreadable effects block counts as none, as before) |
| `__all__` gaps | psd_edit 2 defs + 16 constants; font_check 3 defs + 10 constants | 0 |
| Spelling in commands/ | colour 31 vs color 8; normalise 6 vs normalize 10 | American in the new modules |
| 3.10 / negative-index hits in compiled files | 0 | 0 (`music_player.py:297-298` pre-existing, untouched) |
| Site: pending tip string / PSD field set homes / dead CSS | x3 / x5 / 0 | x1 / x5, cross-referenced / 0 |

## Findings

| # | Finding | Decision |
|---|---|---|
| 1 | Three copies of the GitHub commit path (API blob/tree/commit/ref + SSH fallback) in card_bake, tts_save, repo_history; they had already drifted (tts_save NO_WRITE 401/403/404 vs card_bake 403; tts_save's API commit without an author) | **Done**: `commands/git_commit.py`, `commit_files()` tries the API and falls to SSH as `card_bake.land` did; `cant_write` and `limited` module state; both callers switched. Accepted: card_bake adopts NO_WRITE; tts_save's commits carry the AutoCroissant author; tts_save API timeout 120 -> 300 s. `repo_history._git` stays (different signature) |
| 2 | `blob_sha`, `fetch_verified` and the GITHUB_TOKEN patch duplicated; card_bake ran two auth paths (`_api` built its own header; `_tree` used patched `HEADERS` and dropped the API-version header) | **Done**: `github_api.token_from_env`, `blob_sha`, `fetch_verified` (ValueError); card_bake's wrapper -> `Refused("blob")`; `_tree` through `api` |
| 3 | psd_edit duplicates `psd_regions._union` and imports the private `_fx_pad`; neither exported | **Done**: `fx_pad`, `union_box` public and in `__all__`; design doc section 6 updated |
| 4 | Dead: KIND getattr fallback (CardEdits.KIND exists), `region_masks` (0 callers), save_psd no-preview branches, `Edit.report` never read | **Done** except the `_updated` fallback (the fork's `save()` has no `update_preview` parameter; re-measured 0.07 s, mark set) and `regions()` (one test) |
| 5 | `__all__` incomplete: `TEXT_FIELDS` used by card_bake not exported; font_check's `card_name`/`font_files` public by accident | **Done**: exported; `_card_name` stays a local copy with a comment (importing card_models would drag pickle/StatsDatabase into the font-check Action) |
| 6 | 27 public functions with 2+ params lack Args; `bake_card` takes 8 positionals incl. a `chart` callable | **Done**: Args/Returns on the 27, one-liners on `gate_shape`, `gate_size`, `Snapshot.sha`, `Report`, both `main()`s |
| 7 | main.py `/fonts`: the lazy `from ... import check_repo_fonts` shadows the name bound two lines later | **Done**: `import commands.font_check as font_check` |
| 8 | `/query_ability` with one result and no PNG queued `content=None`: Discord rejects it, the dispatch loop drops it, the user sees "Found 1 results" and nothing | **Done** (compiled module, 3.10 syntax): posts the card's name, as the list branch does. Test added |
| 9 | Broad excepts hiding a cause: `/fonts` ("Font check failed: {e}", no traceback), `_enabled_effects` -> [] (parity pre-check passes an unreadable effect stack), `fx_pad` -> 0, `font_of` out-of-range index | **Partly**: traceback printed; `_enabled_effects(strict=True)` from `set_stat` refuses `parity` with the cause; `fx_pad`/`_enabled_effects` excepts kept broad after review (unreadable = no effects, as before); `font_of` kept (test pins `font_set[0]`) |
| 10 | Naming: three `_api` with three contracts; `_check` means two things; `sorted_out`, `for one in cards`; `Report.to_json` returns a dict; British spelling only in the new modules | **Done**: `_get_json`, `_check_one`, `classified`, `as_dict`, `normalize_ability`, `StatusColor`, `color`. **Skipped**: `run(card=)` (CLI contract, 15 call sites); `api` for font_check (would pull git_commit into that Action's closure) |
| 11 | `run()` nests six levels with a one-element-list once-cache; `is_card_psd`/`EXCLUDE_FOLDERS` live three times | **Skipped**: refactor without a behaviour gain; the constant move edits a compiled module for two lines |
| 12 | Site: `valid_verdict` accepts any string state (a typo'd verdict silently ignored); the "PSD pending" tip x3; the PSD field set x5 across the repos | **Done**: state in `("baked", "refused")` so the existing warning fires; one exported `PSD_TIP`; the `PSD_FIELDS` comment names the five homes |
| 13 | Verified, no change: 3.10 syntax and negative indexes in compiled files; the two Actions' sparse lists equal the import closure (no discord, config, pandas); the tests run `--help` on exactly those lists | `card-edits.yml` and `tts-save.yml` now list `commands/git_commit.py`; the import allowlist test fences the closure |
