# Performance audit (2026-10-21)

Read-only, before any change. Line numbers are the audit's (main 74a5c2c working tree); the fixes moved them.
Scripts: bench/ (`bake_profile.py`, `composite_cost.py`, `font_sample.py`, `font_pool.py`, `site_profile.py`,
`accent_bench.py`). Fork interpreter: `PYTHONPATH=/tmp/wt-pipeline-dev/src ~/Documents/Tools/psd-tools/.venv/bin/python`.
Load average 12-16 during every run: wall times pessimistic by 20-40 %.

## Where a bake's 23 s go

`card_bake --dry-run --card 'Mini Doomer'` (hp 5 -> 8, the tests' fake GitHub, 30.3 MB PSD):

| Step | s | Share |
|---|---|---|
| render_region x3 (baseline 4.06, new 4.04, idempotence 4.22) | 12.3 | 54 % |
| save with preview (`_update_preview`: one full-canvas composite) | 9.45 | 41 % |
| PSDImage.open x8 | 0.82 | 3.6 % |
| gate_bytes (layer_records x2) | 0.23 | |
| parse_card x2 | 0.21 | |
| save_png | 0.21 | |
| artifacts, gate_region, fetch (fake), merge, set_stat | 0.12 | |
| **cb.run** | **23.0** | |

Same bake without the preview: 13.3 s (save 0.07 s). Ability edit: 24.5 s, `set_text_layer` 0.12 s. Composite cost
under the fork: hp row viewport 411x86 (1.3 % of the canvas) 4.39 s, ability viewport 1130x456 4.37 s, full canvas 9.57 s:
the viewport size barely matters because stroke distance transforms and drop shadows run over whole layers
(cProfile tottime: `_apply_source` 4.04, `utils.divide` 3.50, scipy `euclidean_feature_transform` 3.01 over 64 calls,
`_prune_offsets` 1.55).

## Font check

| Measure | Value |
|---|---|
| `--all` over 813 PSDs, 16.4 GB | 87-98 s wall, 71-75 s user, 91 % of one core |
| Per PSD: read / parse / font walk | 0.005 s / 0.08-0.10 s / ~0 |
| Tagged blocks of the parse | 87 % (EngineData 42 %, descriptors 31 %); image data copied, never decoded |
| FontIndex over MDW/Fonts (276 files, 226 MB, 284 faces) | 0.08 s |
| 120 PSDs sequential vs ProcessPoolExecutor(8) | 12.2 s vs 2.0 s (6.2x), identical results |
| `/fonts` first run at the 11 MB/s noted in font_check.py | fonts 21 s; 200-PSD batch ~4 GB ~6 min; the repo ~25 min over 5 calls |

## Website build (`build_site.py`, warm cache, offline copy, 849 cards, 801 share pages)

| Phase | s |
|---|---|
| accent_color x849 (sequential; WebP decode 1.01, Python pixel loop 0.76) | 2.90 |
| make_images (16 threads, summed) | 2.57 |
| copyfile x3396 | 0.73 |
| build_icons | 0.61 |
| repo_tree | 0.21 |
| site_banner | 0.12 |
| share_text x801, apply_edits x849, json.dumps | 0.02 |
| **total** | **5.29** |

Cold cache 19.5 s wall (225 s CPU of resizing over 16 threads). The numpy rewrite of accent_color: 2.11 -> 1.74 s only,
and 16 of 821 colours change tie-breaks; not worth it.

## Test suite (stock venv): 23.7 s, 337 passed, 18 skipped

| File | s | Tests |
|---|---|---|
| test_card_bake | 5.93 | 28 |
| test_parser_goldens | 4.64 | 39 |
| test_update_stats | 3.22 | 17 |
| test_psd_edit | 1.65 | 4 run, 16 fork-only |
| test_font_check | 1.20 | 3 (+1 yaml skip) |

## Findings

| # | Finding | Gain | Decision |
|---|---|---|---|
| 1 | The preview regeneration in `save_psd` is 41 % of a bake. The merged PNG the pipeline already holds IS the flattened card: a fork `PSDImage.save(fp, preview=<image>)` would write it as ImageData instead of compositing. Keeps design decision 2 | -9.3 s per card | **Deferred**: fork change; worktrees in use. `update_preview=False` is the measured 13.3 s path if decision 2 is revisited |
| 2 | Three viewport composites cost 12.3 s because the fork renders effects at full layer size. Crop each source to viewport + `fx_pad` before stroke/shadow rasterization | up to -10 s per card; with 1, a stat bake near 3-4 s | **Deferred**: fork change; the parity harness and goldens are the check |
| 3 | `font_check --all` is single-core PSD parsing. Map `psd_fonts` over a process pool in `check_checkout`; the module is spawn-safe (no config/discord) | 98 s -> 9.8 s measured | **Done**: `POOL_MIN` 8, half the cores, chunksize 4, merged in path order; outputs byte-identical. Action push mode (1-3 PSDs) unchanged |
| 4 | `/fonts` first run is download-bound; the Mac could fill `cache['psds']` from its card clone by blob sha in ~1.5 min | 25 min -> 1.5 min on the Mac; jetson unknown | **Skipped**: unverifiable offline, a behaviour change of `/fonts` |
| 5 | Website: cache the accent colour by blob hash beside the WebPs; `os.link` for the 3396 dist copies | -2.9 s and -0.7 s of a 5.3 s build | **Skipped this pass**: outside the TTSCardMaker scope (finding 12 only); next site pass |
| 6 | Suite: read the two fixture cards once per session; `pyyaml` for the yaml skip; the fork venv lacks `requests` for card_bake | ~1 s of 24 | **Partly**: `pyyaml` listed; fixture left (0.005 s per read: not the cost) |
| 7 | card_bake opens the same PSD 8 times per card | -0.3 s (1.5 %) | **Skipped**: needs a `layer_records` signature change |
