# Tests audit (2026-10-21)

Stock suite (`~/.venvs/ac-py314-psd125`, psd-tools 1.25.0, Python 3.14.8): 337 passed, 18 skipped in 23.2 s; the six
audited files hold 152 tests (card_bake 21, card_edits 62, font_check 28, pickle_saves 5, psd_edit 27, query_output 9).
Fork (`/tmp/wt-pipeline-dev/src` on the psd-tools tool venv): `test_psd_edit.py` 26 passed, 1 skipped in 118 s;
`test_card_bake.py` does not collect there (no `requests`); with `requests` shimmed onto PYTHONPATH 21 passed in 52 s,
45.8 s of it one end-to-end bake test. Mutation audit and gap probes ran in a /tmp copy of the repo (config.py excluded,
pickles copied); every file restored byte-identical. One slip disclosed by the auditor: the rsync carried config.py for a few
seconds; it was deleted unread and the copy's tests ran on the conftest stub.

## Mutation check: 11 of 11 one-line breaks caught

| Mutation | Failing test |
|---|---|
| psd_edit.classify `value > STAT_MAX` -> `+ 1` | test_classify_table |
| psd_edit.gate_region `if outside` -> `if False` | test_merge_png_is_a_pixel_delta |
| card_bake.land head-moved -> `if False` | test_psd_changed_meanwhile_is_head_moved |
| card_bake._upload `if limited` -> `if False` | test_transient_errors_write_no_verdict |
| card_models.save_pickle `temp_path = path` | test_pickle_saves (3 of 5) |
| font_check.layer_fonts `units = len(text)` | test_layer_fonts_runs_and_normal_sheet |
| query_card RESULTS_PER_MESSAGE 10 -> 11 | test_query_output (1 of 9) |
| font_check.pushed_files `>=` -> `>` FILES_PAGE | test_pushed_files_notes_the_file_cap |
| card_bake.bake_card `if recent` -> `if False` | test_recent_refuses_before_fetching |
| card_edits PSD_STATES + "pending" | test_psd_meta_rides_beside_by_and_at |
| card_bake.land_verdicts `_same_verdict` never skips | test_run_drops_redundant_and_writes_verdicts |

## Royal Eradicator Main Cannon (the one 10+ card), measured

atk 12 = rows Top [10] + Bottom [2]; hp/def/spd template rows 9/8/3. `classify({'atk': 11})` -> `stat>10`;
`classify({'atk': 7})` -> a job (no refusal) unless `excluded=True` (the card is on `EXCESSIVE_STAT_EXCLUSIONS`);
`set_stat` 7 or 11 -> `unsupported-card` "atk: 2 bar rows" (template_row runs before the STAT_MAX check). Nothing asserted
this before; now pinned.

## Findings

| # | Finding | Decision |
|---|---|---|
| 1 | The documented fork command cannot collect `test_card_bake.py`: the fork venv has no `requests` (and no pip) | **Owner's environment**; docstrings name both the command and the gap |
| 2 | Mutation check: all 11 caught | Nothing to change |
| 3 | The 403-not-rate-limited -> SSH fallback and `_git_commit` had zero coverage | **Done**: `FakeGitHub.forbidden`; `tests/test_git_commit.py` covers `git_commit` push outcomes (rejected -> None, SSH failure -> Transient with the SSH_AUTH_SOCK note, success -> sha) with `git` faked |
| 4 | 409 never served; `land_verdicts`' "edit changed meanwhile: verdict not written" branch untested | **Done**: head-moved parametrized over 409/422; `test_verdict_skipped_when_edit_changed_before_the_verdicts_commit` |
| 5 | Royal Eradicator checked only `template_row`; the stat>10 vs unsupported-card order unpinned | **Done**: `test_royal_eradicator_two_row_atk` pins the rows, classify's three outcomes, set_stat's order and the exclusion-list membership. The reorder in set_stat is a separate decision, not taken |
| 6 | `pushed_files` noted the compare file cap only; per-commit 300-file and 250-commit caps silent | **Done**: both notes added (notify-only tool, no gate), tests for each |
| 7 | Action-shaped `--help` tests ran under a venv that has everything; a stray pandas/discord import in a shared compiled module would pass here and break the Action | **Done**: third-party import allowlist probe in both tests (PIL, attr(s), certifi, charset_normalizer, fontTools, idna, numpy, psd_tools, requests, urllib3, scipy, aggdraw) |
| 8 | `test_workflow_shape`'s YAML half never ran locally (no PyYAML); `card-edits.yml` had no shape test | **Done**: `pyyaml` in requirements-dev.txt (not installed here: the two yaml tests skip until it is); `test_card_bake` workflow test asserts `dry_run` default true, the `AutoCroissant` actor guard, `tts-save` group and ties the `committed \`` grep literal to `render()` |
| 9 | Fork runtime hogs: a duplicate 20 s re-bake in the end-to-end test; Mini Doomer baked three times across tests; 4 name edits | **Done**: the `cb.run(head=before)` re-bake dropped (covered by test_psd_changed_meanwhile in 0.3 s); Mini Doomer out of NAMED. **Skipped**: a module-scoped bake fixture (couples assertions) |
| 10 | The `font` refusal never ran on this machine (OCR A Extended installed -> skip) | **Done**: patches the fork's `FontMap._lookup` for a missing and a loose face; both refusal lines run on every fork run (27 passed, 0 skipped) |
| 11 | `_sha` x3, `_text_engine` x3, `Reply` x2; dead `_search(repo, 0)` call | **Done**: `conftest.blob_sha_bytes`, `conftest.text_engine` (from `psd_edit.text_engine`, not card_bake's: the fork venv lacks requests), shared `Reply`; dead call removed. The two `FakeGitHub` classes stay (different surfaces) |

## After

| Run | Result |
|---|---|
| Stock suite | 350 passed, 18 skipped, 30.8 s (15 fork-only in test_psd_edit, 1 in test_card_bake, 2 yaml) |
| Fork `test_psd_edit.py` | 27 passed, 0 skipped, 229 s wall under the concurrent rendering gates |
| Suite against the 11 compiled .so (/tmp copy) | 349 passed, 18 skipped (before the review round's extra test) |
| Both workflow yaml halves (fork venv's PyYAML) | asserted: card-edits dry_run default True, actor guard, tts-save group, permissions; font-check paths, `all` default, order |
| pyflakes on every changed module and test | clean (pre-existing nits only) |
