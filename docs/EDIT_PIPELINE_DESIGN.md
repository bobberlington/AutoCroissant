# Edit pipeline design (2026-10-20)

**Status:** designed 2026-10-20, questions answered the same day; Task 3 steps 4-7 (`docs/NEXT_SESSION_PLAN.md`); implementation running. **PSD writes on hold (user, 2026-10-20): other developers are editing the PSDs, so S5 live runs, S8, the title reprints, Sonic's conversion and star folder moves wait for the user's go; code and dry runs on /tmp copies continue; the intended PSD changes are tracked in `docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md`.** Synthesis of three
independent designs and two judge rounds: the "runner" contract (one CLI for both runners, `_psd` status in the edits file),
the "robust" gates (parity bar, record diff, outside-region equality, job key), the "mvp" scope (template cards only, fill-only
status words first, refuse the rest). Numbers measured 2026-10-20 unless marked "to confirm". `/tmp/` research files vanish on
reboot; harness and notes: `~/Documents/Tools/psd-research/`.

## 1. Goal and done

For each PSD-backed edit in TTSCardMaker `site/Card Edits.json`: apply it to the card PSD (digit toggles per stat row; name
and ability text through the fork's text engine, runs kept; status words coloured from the status chart), render only the
changed region, delta-merge it onto the official PNG, gate it, and land one commit: PSD + PNG + the baked fields removed from
the file. The website shows "PSD pending" until that commit. Pure Python, no Photoshop, no browser.

**Done when** (Task 3 step 6): the harness meets its targets (`~/Documents/Tools/psd-research/`, scored against
`score/results/cards_v1.json`), and 10 real edits (stats, a name, ability text with coloured status words) made on the website
are baked by the pipeline and look identical to Photoshop when the committed PSDs are opened on the Mac (section 12). Then
Photopea is dropped.

## 2. Facts the design rests on

Layer censuses below (Name/Ability layer counts, 107 effect cards, 100/100 byte-identical saves, the font triage) are the
2026-10-20 psd-research scans, not re-measured for this document: to confirm. File paths, line refs, the 7 names, sizes and
the workflow facts were re-checked.

- **Stat value = the visible digit.** `Stats Bars/Darks/<Stat> Dark/0-9` are exposure adjustment layers (bbox 0x0) whose mask
  covers cells N+1..10; layer N visible = value N, none visible = 10. The parser agrees (`StatTracker._row_value`,
  `commands/psd_parser.py:67-84`; rows add up in `total()` :86-103). Census of 651 cards: 371 have pure template bars, 234
  have no bars (items, N.M.E.), 46 deviate (4 multi-row, 4 odd paths, 24 value-named drawn layers, 7 "0" decorations, 6
  per-cell drawn, 1 none). A saved toggle differs in exactly 2 flag bytes (Mini_Doomer hp 5->8).
- **Effects on digits (idea 12).** 107 cards carry an enabled effect (GradientOverlay 379 layers, InnerShadow 45, ...) on one
  digit per row, the one visible when styled (Mettaton_EX `Atk Dark/6` styled, `7` visible). The `cards` tree skips effects on
  adjustment layers (`composite.py` `Compositor.apply` L1474; psd-research `RENDER_STATUS.md` class A, 87 cards 33-84% bad).
- **Above 10 (idea 13).** Sonic spd 20 = half-height `spd top`/`spd bottom` bars with all digits hidden plus a renamed `20`
  that changes 0 pixels; Royal Eradicator Main Cannon atk 12 = `atk dark/Top/10` at opacity 0 + `Bottom/2`. Canonical
  encoding since 2026-10-21 (question 3): top row 10 + bottom row N-10. `EXCESSIVE_STAT_EXCLUSIONS` is gone; `allow_high_stats`
  in the edits file silences STATS TOO HIGH for Sonic and the Cannon (`commands/psd_parser.py:1031-1033`).
- **Name.** The bot names a card from its file stem (`commands/card_models.py:209-211`) and, since 2026-10-21 (question 1,
  `PARSER_VERSION` 2), reads the printed title into `display_name` when it differs (`psd_parser._printed_title`: the visible
  `Text/Name` layer, else `Name` layers in the title region, else a layer named after the card; 157 of 813 cards, shown as printed, letter case included: the user's call of 2026-10-21); a Card Edits.json `name` still wins, with the title kept in `parsed_values`. 552 of 651 cards have a visible `Name` text layer, 153 name the layer
  after the card, 4 hide it (Sonic's title is a pixel layer), 5 are rasterized.
- **Ability.** 605 cards have exactly one visible `Ability` directly under top-level `Text` (box text, Single-line composer);
  34 elsewhere, 7 hidden and re-laid out, 5 several visible, 19 empty (the categories overlap, 670 > 651: to confirm). The
  parser reads the layer text
  (`commands/psd_parser.py:559-584`). 0 `Ability FX:` layers exist; 52 old-style overlay word layers do (Gasoline, Royal
  Eradicator `Blazing`); 0 abilities use the stoogedings font yet.
- **Status chart.** `~/Desktop/TTSCardMaker/MDW/Croissant's Special Folder/Satus Effect Colors.psd` (1174x4000, 18.7 MB):
  `Buffs` (12 text layers, none styled) and `Debuffs` (54, 15 styled: OuterGlow on Blazing, Frozen, Hexed, Glued, Infected,
  Hemorrhaging, Contracted, Cursed Inferno, Spirit Flame; GradientOverlay on Become Back Money, Corrupted Blood, Condemned;
  both on Withering, Frozen In Time; DropShadow+Stroke on Curse). Fill = the status colour. Cards already use the swatch
  colour for status words about 95% of the time (ttscardmaker-companion skill §6).
- **Text engine** (fork branch `feat/text-engine`, 647e19c6): `TypeLayer.set_text(text, keep="runs")` keeps the runs on the
  unchanged ends (`src/psd_tools/api/layers.py:3099`); `set_span_style(start, end, fill=(r,g,b))` (:3113);
  `rasterize(fonts=FontMap(dirs=[...]))` replaces the stored raster (:3142), raises `MissingFontError` on a missing face,
  reports `overflow` when a line no longer fits (fork `docs/text.rst`). Reopened `numpy('color')` equals the render; unedited
  saves are byte-identical (100/100 cards).
- **Compositor** (branch `cards`, 7ea581b7): `psd.composite(viewport=(x0,y0,x1,y1))` equals the same crop of a whole-card
  render (max diff 0); `force=True` redraws vectors and breaks `LINE` rows (diff up to 106 on Mini_Doomer), so baselines use
  `ignore_preview=True`; `layer.composite()` drops the backdrop and the parent group's effects. `save()` regenerates the
  merged preview when the document changed (`psd_image.py:295-327`; 20-45 s under load).
- **The edits file today.** `{"about": ..., "cards": {...}}`, 56 cards: `name` 29, `series` 28, `subseries` 28, `_by`/`_at` 56.
  22 of the 29 names equal the printed title already (Sans "snas.", Kyle, Clot_Black "Black Clot"); 7 differ, read back from
  (2026-10-21, parser read: 20 are redundant, 21 with Thunderfury aligned; Sans and Papyrus stay: their title layers are named
  after the print, not `Name`, so the parser reads no title for them)
  `Text/Name` with the fork: Clot_White "White CLot" (typo), PoD08-12 "M.O.A.B" / "D.D.T" / "B.F.B" / "Z.O.M.G" / "B.A.D" (the
  edit adds the trailing dot), Thunderfury "[Thunderfury: Blessed Blade of the Windseeker]" (the edit drops the brackets).
  **By section 3's rule those 7 are name jobs today**; no stat or ability job exists. Whether the 7 titles are to be reprinted
  is question 7 (section 14); nothing bakes before it is answered.
- **Consumers already converge.** `CardEdits.apply` restores `parsed_values` then applies the file (`commands/card_edits.py`
  :130-146), `_`-keys dropped at parse (:104); `build_site.py` undoes the bot's edits the same way (:589-592), drops `_`-keys
  (:387), passes only `_by`/`_at` into meta (:393), records `edited`/`orig` (:414-421), finds PNGs case-insensitively (`png_for`
  :563-567). `edit.js`: `FROM` marks name, stars, types, hp, def, atk, spd, ability, creatures "PSD" (:22); `editsFile` keeps
  keys outside `FIELDS` (:40); `commitFiles` retries 422 (:81-95); the unset rule (:308) drops an edit set back to the PSD.
- **Bot flow.** `traverse_remote` fetches the tree with an ETag, re-parses PSDs whose blob sha changed (`_psd_changed`,
  `commands/repo_traverser.py:77`), skips `Markers`/`MDW` (:36), reads Card Edits.json by blob sha (:120-128, :257), archives
  the old card when the blob changed (:372), then authors, edits, duplicates, edits again (:177-180). `update_stats` is
  `to_thread` with no lock (`main.py:359`); pickles are committed only by `/push` (`commands/update_bot.py` `git_push`, "PICKLE").
- **Commits today.** `commands/git_commit.py` (2026-10-21, shared by `tts_save` and `card_bake`): Git Data API blobs/tree/commit/
  non-force ref update, `COMMIT_TRIES` 3 on 409/422 (`CONFLICT`), SSH fallback `git_commit` into the blobless history clone when
  `POST git/blobs` answers `NO_WRITE` (401/403/404; `cant_write` remembers it), a rate-limited 403 raises `Transient` unless the
  caller asks for SSH for the run (`ssh_on_limit`: tts_save does, the Action does not; `limited` keeps the retries off the API); `tts_save._lock`, Discord `_post`. TTSCardMaker `tts-save.yml`: push on `**/*.png`, `TTS/decks.json`, `site/Card Edits.json`; `contents: write`;
  group `tts-save`; plain `git push`. `site.yml`: `paths-ignore: TTS/**`, cron every 6 h. Main's protection rule requires
  nothing (`branches/main/protection`: force pushes allowed, no required checks or reviews); no Actions secrets, no deploy
  keys. GitHub: pushes made with `GITHUB_TOKEN` trigger no workflows.
- **Sizes and limits.** 814 PSDs, 16.4 GB, median 19 MB, max 84.5 MB (11 over the 50 MiB warning, 0 over the 100 MiB block);
  `MDW/Fonts` 259 top-level font files, 220 MB (plus `stoogedings-src/`, 3,600 files, 423 MB in all: the runner caches the
  top level only). `git/blobs/{sha}` with the raw media type serves any of them (verified on the 84.5 MB file;
  `github_api.fetch_blob`, `commands/github_api.py:77`). Public repo: `ubuntu-latest` 4 vCPU / 16 GB, 6 h per job, 10 GB cache,
  1,000 API requests/h for `GITHUB_TOKEN`. PNGs: 5 case-only stem mismatches; 6 numbered duplicates over 5 PSDs (A_Snack2/3,
  All_Purpose_Medical_Kit2, Regeneration_Potion2, Watermelon2, NME_Stock2).

## 3. Two kinds of edits

One table, three consumers: `CardEdits.KIND` (new, beside `TYPES`/`ATTRS`/`FIXED_BY` in `commands/card_edits.py`),
`build_site.py PSD_FIELDS` (new, beside `EDIT_FIELDS` :71), `edit.js FROM` (exists). All 19 fields:

| Field | Kind | Pipeline (round 1) | After the bake |
|---|---|---|---|
| `hp` `def` `atk` `spd` | PSD + database | toggle `str(N)` visible in `<Stat> Dark`, clear the other nine (10 = none); template rows only; 0-10 | field removed |
| `ability` | PSD + database | `set_text(keep="runs")`, status words filled from the chart, `rasterize`; one visible `Text/Ability` only | field removed |
| `name` | PSD + database **when the value differs from the printed title** (the fresh parse's `display_name`, else the file name), else database-only | `set_text` + `rasterize` on `Text/Name`; refuse the card-named, hidden and rasterized layers (unless the value is already what the bot shows: redundant) | **field removed** (since 2026-10-21 the bot reads the title into `display_name`; the 21 redundant names drop at the first live bake) |
| `stars` | PSD + folder (creatures, items, fields: the `N Stars` folder wins; minions, aux items, N.M.E.: star layers) | refused ("not yet") | - |
| `types` | PSD (icon layers or font tokens) | refused ("not yet") | - |
| `creatures` | database-only (`CardEdits.KIND`; the per-creature rows of Pix, Tainted Lazarus, Twin Emperors stay hand-edited) | never touched; a stat edit beside it is refused (creature 1 sets the stats) | stays |
| `type` `subtype` `series` | folder | never touched | stay |
| `subseries` `note` `aliases` `dup_of` `hidden` `problems_ok` `allow_high_stats` | database-only | never touched; `_by`/`_at` kept | stay |

Classification is value-aware: PSD-kind only when the value differs from a fresh offline parse of the PSD (the goldens' route,
`tests/golden_cards.py:62-76`: `PSDParser` on an empty `StatsDatabase` + `IconVocabulary.build` from the committing tree's
`Types/`, stars and manifest), so it runs after the blob is fetched (section 5 step 2). Import from `commands.psd_parser`
(`IconVocabulary` :174, `PSDParser` :318, `CardValidator` :948) and `commands.card_models` (`StatsDatabase` :154), not from
`commands.psd_analyzer` as the goldens do: that module imports `discord` (:9), `pandas` (:13), `GIT_TOKEN` (:18),
`repo_traverser` (:40), `query_card` (:49) and `utils` (:50), none of which the runner has. An equal value is **redundant**:
dropped from the file with no pixels.

## 4. Architecture

One pure-Python entry point run identically by both runners:
`python -m commands.card_bake --repo MichaelJSr/TTSCardMaker --head <sha> [--dry-run] [--card <Name>]`. Two flags only
(simple-tooling rule): `--dry-run` writes the artifacts and commits nothing, `--card` restricts the run to one card. Whether
the saved PSD carries a regenerated preview is question 2's answer baked into the code, not a flag. Input: one tree snapshot
+ blobs by sha. Output: zero or more commits (one per card) and a markdown report on stdout in `tts_save._render` style. The
runner supplies credentials, the fonts dir, a cache dir, how to push, where the report goes. The core never reads
`config.py`, never touches the pickles, never writes outside its work dir.

New modules in AutoCroissant `commands/` (not cythonized; the `tts_save.py` pattern: no discord or config imports). Their
import closure is `psd_parser`, `card_models`, `card_edits`, `github_api` (requests; `config` in try/except; since 2026-10-21
also `token_from_env`, `blob_sha`, `fetch_verified`), `git_commit` (2026-10-21), `repo_history`, `global_config` (stdlib only)
and psd-tools; `psd_analyzer` is never imported (section 3).
- `commands/psd_edit.py` (new): `classify`, `baseline`, `set_stat`, `set_text_layer`, `status_fills` (chart reader),
  `changed_region`, `render_region`, `merge_png`, the gates. Pure functions over an open `PSDImage`.
- `commands/psd_regions.py` (new): port of `~/Documents/Tools/psd-research/score/regions.py` (`_fx_pad` :26, `stat_rows` :68,
  `regions` :94; its `region_masks` :210 stayed there, dropped here 2026-10-21 with no caller); the Action cannot check out
  psd-research. Exports `fx_pad` and `union_box` (renamed from the private names 2026-10-21).
- `commands/card_bake.py` (new): tree fetch, blobs (`github_api.fetch_verified`, sha1 `blob N\0` check), per-card loop, Card
  Edits.json rewrite (Python port of `editsFile`: keys sorted, `ORDER`, one line per card), commit through
  `git_commit.commit_files` (2026-10-21: the API path generalized from `tts_save.update_repo_save` and the SSH path made to take
  several files live there, used by both), report.
- `tests/test_card_bake.py` (new) on a small fixture built from `MDW/Templates/Creature_Template.psd` (size to confirm).

Fork `MichaelJSr/psd-tools` (both branches leave main b1b454ba; only the fork's `docs/changelog.rst` overlaps; only upstream's
125 tags; `origin/cards` 7ea581b7 and `origin/feat/text-engine` 647e19c6 are pushed, the work below is not): merge
`feat/text-engine` into `cards` (the `text` extra, uharfbuzz >= 0.56 and freetype-py >= 2.5, exists only on
`feat/text-engine`; `cards` has just `composite`), add `PSDImage.save(fp, update_preview=True)` (one line; today only the
private `_updated=False` hack skips the 20-45 s preview) and a tagged-block copy for `lfx2` (effects are a read-only proxy,
`api/effects.py:5`), tag `cards-20261020`, push. Pin in both runners:
`pip install "psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@cards-20261020"` (a commit sha if pip's
wheel cache needs it: to confirm). Plan step 7 drops the pin once upstream has the changes. Card knowledge (template rows,
chart, FX naming) stays in the bot. TTSCardMaker: `.github/workflows/card-edits.yml` (new); `site/build/build_site.py`,
`site/assets/app.js`, `site/assets/edit.js` for the badges (section 9). No psd-tools on the website side.

## 5. Per-card algorithm and gates

Job key = (Card Edits.json blob sha E, PSD blob sha P, the PSD-kind fields and values). A rerun on the pipeline's own commit
finds no PSD-kind field and does nothing: idempotence is the resume mechanism, state lives in git. Refusals are data with a
code, never exceptions; one bad card never blocks another.

0. **Snapshot.** Tree at head H; `CardEdits.parse` of the file blob; card key -> PSD path (same stem, case-insensitive like
   `png_for`), PNG path(s) and shas. Skip own commits (message `Card edit:` / actor AutoCroissant).
1. **Fetch** every card named in the file that has a field outside the folder and database-only kinds (section 3): PSD,
   PNG(s), chart (once per run), fonts (cached). Gate `blob`: sha1(`blob N\0` + data) == tree sha. Refuse `recent` first
   (PSD pushed < 30 min ago: don't race its author; by push time, not commit date: `activity` + one `compare` per run).
2. **Classify** (section 3) against the open PSD and a fresh parse. Refuse: `duplicate` (5 PSDs with numbered PNGs),
   `unsupported-card` (non-template rows, not one visible `Text/Ability`, name layer not `Text/Name`, overlay word or
   ` FX: ` layers, `[type]` tokens on a card with image icons), `stat>10` (on the value alone; `allow_high_stats` is the website's business).
3. **Baseline parity.** Open with the fork; region per edited field (section 6); `psd.composite(viewport=region,
   ignore_preview=True)` vs the official PNG crop. Gate `parity`: bad pixels (|diff| > 24, `score_cards.py THRESH`) <= 0.5%
   and mean abs <= 1.0 (Mini_Doomer row: mean 0.62, max 2; the bar stays, the user 2026-10-21: section 7 idea 12). A row the
   fork already renders wrong (class A) is refused here, never pasted.
4. **Apply.** Stat: visibility toggle; the row's effect follows the visible digit (idea 12, 2026-10-21): the hidden digit's
   `lfx2`/`lrFX` blocks move onto the shown digit's record, replacing a block whose effects are all off (the block flag is not
   enough: Mini_Doomer `Hp Dark/2` has `effects.enabled` True over an InnerShadow and a GradientOverlay both `enabled=False`);
   a shown digit with its own enabled effect keeps it and the hidden one's is cleared; to 10 nothing moves; from 10 the one
   hidden styled digit's block comes back, several such digits refuse `unsupported-card`; a hidden styled digit beside a visible
   unstyled one (Mettaton_EX Atk 6) is left alone, so the path matters (7 -> 10 -> 4 styles 4 with the 6's effect, 7 -> 4 moves
   nothing). Two rows (idea 13): Top shows the value and Bottom 0 up to 10, then Top none
   and Bottom value - 10 (`psd_edit.set_stat`). Ability/name: `set_text(keep="runs")`; for each chart status name in the new text,
   `set_span_style(start, end, fill=(r/255, g/255, b/255))` per occurrence (`fill` takes 0..1, `layers.py:3113-3128`; the chart
   swatch is 8-bit; since 2026-10-21 every occurrence in the text, edited or not: the chart is the authority, question 6);
   `rasterize(fonts=FontMap(dirs=[MDW/Fonts]))`. Then the FX layers (idea 14, 2026-10-21): every `<Ability> FX: ...` layer is
   deleted and one is cut per styled status word in the new text (`Ability FX: Blazing`, `Ability FX: Frozen#2` for a repeat,
   named as `Effect Layers.jsx` names them): the JSX's key copy (the word's runs white, the rest black) laid out and rasterized
   again with the engine selects the word's pixels (`_word_mask`, red over black at threshold level 1), the ability's own raster
   gives the pixels (`numpy("color"/"shape")` after `rasterize`), `PixelLayer.fromarray` makes the layer with a unicode name
   and a fresh layer id (the seed resource moves past it), the chart layer's `lfx2`/`lrFX` blocks are deep-copied onto it, and
   it goes directly above the top-level `Text` group. A hand-made overlay word layer over the ability is hidden when it names a
   styled chart status, else the card refuses `unsupported-card`. `styled-word` remains only for a psd-tools without the engine
   pieces (`psd_edit._fx_engine`).
5. **Render + merge** (section 6). Gate `region`: pixels differing between old and new PNG lie inside the padded region and
   outside it the new PNG is byte-equal to the old. Gate `shape`: PNG size and mode unchanged.
6. **Save.** Preview regenerated (an honest flattened image in the committed PSD, 20-45 s) or `save(update_preview=False)`
   (0.2 s, stale preview): question 2 decides once, for every card; no flag. Gate `bytes`: compare parsed layer
   records old vs new (names, flags, channel lengths, EngineData), not raw offsets (a text edit shifts every later record): a
   toggle changes exactly the two flag bytes of `N_old`/`N_new` plus the effect blocks its Edit declares moved (the same bytes leave
   one record and land on the other, or leave only when cleared), a text edit one record + its channels; an ability edit's FX
   layers: exactly the declared `<Ability> FX: ...` records gone and the regenerated ones come (pixel records, transparency
   channel first, carrying the chart's `lfx2`/`lrFX` bytes), one flag byte per hidden overlay; records are paired in file order
   once those are taken out (`gate_bytes`). Gate `size`: PSD within
   +-20% of the old blob. Reopen: layers unchanged but the toggled digits; one digit per edited row; `layer.text` == the edit.
7. **Fonts.** Gate `font`: no `MissingFontError` under `policy="raise"` and no `font_loose` on the edited layer; gate
   `overflow`: `report.warnings` has no `overflow` ("ability does not fit").
8. **Re-parse.** Gate `reparse`: offline parse of the new PSD; `to_dict()` == the old parse with exactly the edited fields
   replaced (ability compared after the parser's own whitespace rule on both sides); `alt_names` unchanged;
   `CardValidator.validate` (`commands/psd_parser.py:948`, imported from there) is a subset of the old problems minus
   `FIXED_BY[field]`; no `EFFECT LAYER *` (the validator's coverage >= 0.95 is the FX layers' gate); `STATS TOO HIGH` only when
   every stat above 10 sits on Top + Bottom rows (idea 13; a raise from <= 10 adds it there and is not a new problem, fixed
   2026-10-21 review). FX layers back as declared after the reopen: the names in order,
   visible, styled, directly above `Text`, no overlay word layer showing.
   Name: `Text/Name` read-back == the edit and the stem unchanged (the parser reads the new title into `display_name`; `gate_reparse` leaves that field out of the comparison).
9. **Idempotence.** Gate `idempotence`: classify on the new PSD + new PNG + rewritten file finds nothing to bake and the
   region render matches the merged PNG within the parity bar (the delta merge keeps Photoshop's pixels).
10. **Commit.** Re-fetch head H' and the file at H'. Gate `edit-changed`: the card's PSD-kind values at H' equal what was
    baked, else bake nothing for it (never remove a value you did not render). Tree on `base_tree` H': PSD, PNG (existing
    path and case), `site/Card Edits.json` with the baked fields removed and `_psd` written (section 9), re-serialised by the
    `editsFile` port. Entry rule: an entry with no `FIELDS` key left is deleted whole, `_by`/`_at`/`_psd` with it (the port
    does this explicitly; `editsFile` itself keeps any non-`FIELDS` key, `edit.js:39-40`, and would leave a `_`-only entry);
    otherwise `_by`/`_at` stay. Author/committer `AutoCroissant <autocroissant@gmail.com>`, non-force ref update, 409/422 ->
    back to step 10 up to 3 times (`head-moved` after that). Message: subject only, `Card edit: Goombella spd 4 -> 5` (the
    editor, shas, gate codes and fork tag go to the report and `_psd`). Never in a commit: pickles, `MDW/`, `TTS/`.
11. **After.** `gh workflow run site.yml` and `tts-save.yml` (a `GITHUB_TOKEN` push triggers no workflow).

## 6. Delta-merge rule

Region = union(old layer bbox, new layer bbox, `layer.mask.bbox` of every toggled digit; the digits' own bbox is 0x0) +
`psd_regions.fx_pad` (outside strokes on the layer and its ancestors + max(shadow distance + size, glow size)), clipped to the canvas.
Measured: a toggle's diff stays inside the digits' mask union ((360,1200,1173,1284) for Mini_Doomer's hp row); a name edit's
diff is the bbox union padded by exactly the `Text` group's 7 px Stroke. Render that viewport with `psd.composite(viewport=...)`
(baseline `ignore_preview=True`; after an edit `is_updated()` bypasses the preview), paste it over the official PNG, leave every
other pixel. Never `force=True`, never `layer.composite()`. Duplicate PNGs would each need a re-export: refused in round 1.

## 7. Ideas 12, 13, 14

- **12. Effects on toggled digits.** Code done 2026-10-21 (`psd_edit.set_stat`, section 5 step 4 for the rule): the hidden digit's
  `lfx2`/`lrFX` blocks move onto the shown digit (`tagged_blocks.pop` / `[key] =`; the fork's effects proxy re-reads them, no
  cache); gate `bytes` checks the same bytes left one record and landed on the other. Measured on Mettaton_EX hp 5 -> 8 through
  every gate: 2 flag bytes + 5884 block bytes (lfx2 5476 + lrFX 408 with headers), baseline parity mean 0.90 / 0.0% bad / max 7
  (bar 1.0 / 0.5%), 18,311 pixels replaced, parser reads 8, rerun nothing, 10.2 s without the preview; spd 6 -> 9 likewise (mean
  0.898). The moved effect paints the new cells: inside digit 2's mask after hp 5 -> 2 the render averages the official PNG's
  colour inside digit 5's mask within 0.3 per channel (the lit cells were 20+ brighter before), so the fork's mask-as-shape agrees
  with Photoshop's export on these cards. Still refused at the baseline, by the row's own bar and not the digits: Mettaton's
  toggles whose region takes in lit cells (hp 5 -> 2 mean 1.03, atk 7 -> 6 mean 1.04; 0% bad, max 7: the pixel bars' GradientOverlay
  renders at 1.17-1.30 per lit cell, the dark cells at 0.87-0.96), and every Royal Eradicator Main Cannon row (ColorOverlay +
  BevelEmboss digits: 5-7% bad, max 110, with the block popped 58%: the fork paints most of the bevel, not all). Census
  (`/tmp/scout/scan.json`, 2026-10-21): 396 styled rows; 338 with the visible digit styled, 41 with only a hidden digit styled,
  1 at value 10 with one hidden styled digit (Green_Mage atk), 2 at 10 with several (Azamoth_the_Forgotten_King hp: refused),
  14 with several visible styled digits (False_Knight, Tainted_Lazarus: non-template anyway).
  **Decision (the user, 2026-10-21):** the bar stays (`PARITY_MEAN_MAX` 1.0 / `PARITY_BAD_MAX` 0.5% / `PARITY_THRESH` 24); Mettaton EX's
  toggles over lit cells (mean 1.03-1.04) wait for a fork fix of the pixel bars' GradientOverlay shade (a rendering round candidate),
  not a looser bar.
- **13. Stats above 10.** Code done 2026-10-21 (`psd_edit.template_rows`): a `<Stat> Dark` group holding two template rows named
  Top and Bottom (the Cannon's `atk dark`), Top also holding a `10` marker: an eleventh exposure digit visible at opacity 0. The
  marker is load-bearing, not a stray: the parser's `StatTracker.total` skips a row with no visible digit, so Top none + Bottom 2
  reads 2 (measured on a /tmp copy with the marker hidden); with it, Top none + Bottom 0 reads 10, Top 7 + Bottom 0 reads 7, Top
  none + Bottom 5 reads 15. Writing: value <= 10 shows Top = value (10: none) and Bottom = 0; 11-19 Top none and Bottom = value - 10;
  20 needs a Bottom marker too (none on the Cannon: refused `stat>10`); `classify` refuses > 20, and 11-20 on a one-row card
  (`stat>10`, "a second row is standardization work (ledger L3)"); gate `reparse` accepts `STATS TOO HIGH` only for a stat above
  10 on two rows (also when the raise from <= 10 adds it: 7 -> 15 passes, review fix 2026-10-21). Measured on the Cannon (12 -> 7, 15, 10; hp 9 -> 2 as the replace case): 3/2/2/2 flag bytes, 6360 block bytes
  (Bottom 2's ColorOverlay + BevelEmboss follows the shown digit), the parser reads each value back; the render gates refuse it
  (idea 12 above). Earlier 2026-10-21: `EXCESSIVE_STAT_EXCLUSIONS` deleted (STATS TOO HIGH unless the edits file says
  `allow_high_stats`: Sonic, Royal Eradicator Main Cannon; goldens refreshed). Still open: Sonic's conversion (L3: two rows under
  a `Spd Dark`, a marker in each).
- **14. FX layers from the text** (shared with Task 3c). Code done 2026-10-21 (`psd_edit.set_text_layer`, section 5 step 4 for
  the recipe; no fork change: the glyph pixels come from the JSX's own key-copy technique over the engine's public `model`,
  `layout` and `raster` modules, `PlacedGlyph` still has no cluster index). Derived state: every `<Ability> FX: ...` layer goes
  and one comes back per styled status word (`Ability FX: Blazing`, `#n` for a repeat), above `Text`, the chart's `lfx2`/`lrFX`
  copied; the hand-made overlay word layers naming a styled chart status are hidden (question 6), the others refuse. Proof
  against `Effect Layers.jsx` (Convert overlays, the 2026-10-20 trial copies under `/private/tmp/fontexplore/fx/mig/out/`) on the
  unchanged text: Gasoline `Ability FX: Blazing` IoU 0.966 with the JSX pixel cut, 99.9% of our ink on its ink, bbox equal but
  for one column (the fork's `g`), validator coverage 1.0 (the JSX's 3,422 ink pixels, ours 3,312); the chart's Blazing glow and
  Gasoline's overlay glow are the same parameters, and an FX cut from Photoshop's own raster renders like the JSX state over
  the FX box (mean 0.01, 0% bad) and within the bar over the ability region against the official PNG (0.112% bad, mean 0.70);
  with the text re-rasterized by the fork (what an edit does) the region reads 1.10% / 1.47: the Century raster's fidelity (8% of
  glyph pixels), not the FX. Royal Eradicator Main Cannon IoU 0.937, 100% of ours on the JSX's; the JSX kept the overlay's
  orange-red glow (255,42,26) and word, we normalize both to the chart's red, so the FX box differs from the PNG and the JSX
  state by design (13-14% bad) while the ability region stays within the bar (0.357% / 0.86). Both proof cards keep image-icon
  gaps, so the token rule refuses their edits in the pipeline. A real edit through every gate on Mini_Doomer (fork, no preview):
  adding Blazing and Frozen 24.8 s (baseline 7.0, render 7.7, idempotence 7.7), two FX layers (2 x 5,916 block bytes landed, 9,964 and
  7,777 ink pixels, coverage 1.0), parity 0.018% / 0.86, 201,292 pixels replaced, PSD +60,688 bytes (the fork's RLE of the
  re-rasterized Ability plus the two layers); then dropping Frozen and repeating Blazing 17.8 s, `Blazing#1`/`#2`, the two old
  records removed and two added, +4,056 bytes; validator clean, reruns redundant. Through `card_bake.run(dry_run=True)` against a
  fake GitHub (chart fetched by sha): Blazing added to Mini_Doomer bakes in 29.1 s with the preview regenerated, the committed-to-be
  PSD 30,329,782 -> 30,231,517 bytes (the fork's RLE of the re-rasterized Ability is tighter than Photoshop's, FX layer included), the
  report line names `FX layers: Ability FX: Blazing`. Gate `bytes` pairs records in file order around
  the declared removed/added FX records; gate `reparse` checks the layers back, visible, styled, directly above `Text`, no overlay
  showing, no `EFFECT LAYER` problem. Tests: `tests/test_psd_edit.py` (the JSX comparison skips when the trial copies are gone),
  `tests/test_card_bake.py` dry run.

## 8. Runner

**GitHub Actions (primary).** TTSCardMaker `.github/workflows/card-edits.yml` (new):
```yaml
name: Card edits
on:
  # push trigger deferred (section 14, question 7): dispatch only until Thunderfury's entry matches the print (S5)
  push: {branches: [main], paths: ["site/Card Edits.json"]}
  workflow_dispatch: {inputs: {card: {type: string}, dry_run: {type: boolean}}}
permissions: {contents: write, actions: write}
concurrency: {group: card-edits, cancel-in-progress: false}
jobs:
  bake:
    if: github.actor != 'AutoCroissant'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v7      # sparse, non-cone: /site/, /Types/, /MDW/Fonts/ (top level), "/MDW/Croissant's Special Folder/Satus Effect Colors.psd"
      - uses: actions/checkout@v7      # bobberlington/AutoCroissant -> bot/: commands/{card_bake,psd_edit,psd_regions,psd_parser,card_models,card_edits,github_api,git_commit,repo_history}.py global_config.py (never psd_analyzer: discord, pandas)
      - uses: actions/setup-python@v7  # 3.12, cache: pip
      - uses: actions/cache@v6         # MDW/Fonts + the chart, keyed on `git ls-tree HEAD` of those paths (site.yml does this for PNGs)
      - run: pip install requests "psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@cards-20261020"
      - run: python -m commands.card_bake --repo "$GITHUB_REPOSITORY" --head "$GITHUB_SHA" | tee "$RUNNER_TEMP/report.md" >> "$GITHUB_STEP_SUMMARY"
        working-directory: bot         # as tts-save.yml's `cd bot`
        env: {GITHUB_TOKEN: "${{ secrets.GITHUB_TOKEN }}", PSD_TOOLS_FONT_DIRS: "${{ github.workspace }}/MDW/Fonts"}
      - uses: actions/upload-artifact@v4   # report + before/after/diff crops per card
      - run: gh workflow run site.yml && gh workflow run tts-save.yml   # only when a commit was made
```
Action versions match `site.yml`/`tts-save.yml` today (checkout@v7, setup-python@v7, cache@v6). PSDs are fetched by blob sha
inside the script; card folders are never checked out (16.4 GB). The extras should have x86_64 and aarch64 manylinux wheels
(uharfbuzz, freetype-py, aggdraw, scipy; exact versions and wheel availability to confirm, PyPI not reachable while writing);
only `_rle` compiles. Budget:
every timing so far is from this Mac at load average 65-80 (open 0.5 s, stat-row viewport 8-22 s, save with preview 22-44 s,
without 0.2 s; harness full composite p50 11.3 s per card under 20-way contention): estimate under 2 min per card, to confirm
idle. `tts-save.yml` fires on the same event and pushes plainly: give it a fetch-and-retry on non-fast-forward (or one shared
concurrency group; cross-workflow sharing to confirm). Fonts: 10 faces (Century, TimesNewRomanPSMT, Impact, ArialMT,
ComicSansMS, ...; 112 layers) resolve only from the Mac's platform directories and 6 are missing everywhere (29 layers;
`~/Documents/Tools/psd-research/text-design/score-runs/TRIAGE.md`; card share to confirm): add them under `MDW/Fonts`
(licence: owner) or those cards refuse on `font`.

**Jetson (manual fallback, same module).** Code done 2026-10-21 (S7): `commands/card_runner.py` (uncompiled) runs `card_bake.run`
inside the bot, by hand only: `/bake_cards [card] [dry_run]` (admin, `dry_run` default True). **One live runner, the Action
(decided by the user 2026-10-21):** the bot never queues a bake on its own; the `notice_edits` hook (a dry-run `bake_cards` queued
at the end of `update_stats` when the edits file's blob sha changed) landed and was removed the same day. Lifting the PSD hold is
the push trigger of `card-edits.yml`, nothing in this repo. The bake takes `card_runner.bake_lock`, which `update_stats` holds for
its whole run (queued commands run concurrently, `process_command_queue` main.py:880-889), so an update and a bake never overlap;
whoever finds the lock held says so and waits. The report goes where the command ran (a reminder's FakeInteraction too) through
`queue_file` like tts_save's `_post`: the head lines and sections as messages, the full markdown attached as `card_bake_report.md`,
then the commit links; a failure is one line. card_bake and psd_edit are imported inside the bake, so the bot starts without the
fork; without its text engine the bake refuses with one line. Token: the bot's `GIT_TOKEN` (card_bake's `token_from_env()` only fills in when config.py has
none), so no environment variable is needed in the unit. Push through `git_commit.commit_files` (API, SSH through the history
clone with the unit's `SSH_AUTH_SOCK` when the token can't write); never `log --name-only` or `diff` in the history clone
(git 2.25 downloads every blob; it grew to 4 GB once). Still to do, only if the fallback is ever needed (orchestrator): install the fork tag into
`~/venvs/autocroissant` (aarch64 wheels resolve; `_rle` builds with gcc 10.5; stock psd-tools 1.25.0 there on 2026-10-21)
once `tests/test_parser_goldens.py` passes under it; a fonts dir on the jetson (`card_bake._fonts_dir()` reads
`PSD_TOOLS_FONT_DIRS`, else `LOCAL_DIR_LOC/MDW/Fonts`, and the jetson has no `~/Desktop/TTSCardMaker`: text edits refuse on
`font` until card_bake keeps fonts and the chart in `~/Documents/card-bake/` by sha or the unit sets the variable; the work dir
is a fresh `mkdtemp(prefix="card-bake_")` per run there, `RUNNER_TEMP` on the Action); one live bake with Actions disabled; the
CPU/RAM peak. Costs: 4 shared cores, 6.8 GB total (4.2 GB free on 2026-10-21; peak to confirm), 20-85 MB uploads, the
reminder's cadence (daily `/update_stats` at 5AM).

## 9. Website: pending, baked, refused

- `build_site.py`: `PSD_FIELDS = {name, stars, types, hp, def, atk, spd, ability}` (no `creatures`: database-only, section 3;
  `edit.js FROM` :22 marks it "PSD" today and S1 changes that to no entry, so the editor stops calling it a PSD field); per
  entry `pending` = the set PSD fields `psd_edit.classify` would try (a name against the bot's title, else the file name, via
  `title_text`, only for `parser_version` >= 2 pickles; an ability via `normalize_ability`); duplicates copy the original's but
  `name`. Since 2026-10-21 (TTSCardMaker 98337b7) a name to bake shows "PSD pending"; refusals are not predicted (site NOTES §5). The site shows the bot's title instead of the file name (the stem stays a searchable alias). Pass `_psd` into meta beside `_by`/`_at` (:393) -> `psd` on the entry.
  `reparsePending = tree PSD blob sha != card.blob_sha` covers the gap between the bake commit and the next PICKLE push.
- `_psd` lives in Card Edits.json, written only by the pipeline: `{"state": "baked" | "refused", "for": {field: value},
  "psd": "<blob8>", "reason": "<code>: <why>", "at": ...}`. Both readers drop `_`-keys (`card_edits.py:112-113`, `build_site.py:387`),
  `editsFile` keeps them (`edit.js:40`): no new file. Rewritten every run; ignored when `for` or `psd` no longer match.
- `app.js:347`: "PSD pending: SPD, Ability" beside "Edited" (tip: the card image is being regenerated); "Image update refused:
  ability does not fit"; "Re-parse pending"; `pending` in `FLAGS` (:25); the Edited panel (:588) lists them. `edit.js`:
  PSD-group rows show the note; the unset rule (:308) already handles "back to the PSD".

## 10. Reporting and failure handling

Four sinks: the commit body; the job summary + artifact (report, before/after/diff crops, parse diff, record diff); `_psd` for
the editor; Discord on the jetson (summary, `.md`, commit link). Refusal codes are a closed list: `duplicate`, `unsupported-card`,
`stat>10`, `recent`, `blob`, `parity`, `styled-word`, `font`, `overflow`, `bytes`, `size`, `region`, `shape`, `reparse`,
`idempotence`, `edit-changed`, `head-moved`, `error`. Since 2026-10-21 `unsupported-card` also names a row whose several digits
carry enabled effects (which one the row shows is ambiguous) and the overlay word layers the chart has no style for (Smoldering,
+2 DEF, Despair...), `stat>10` says whether the value is above 20, needs a second row, or needs a Bottom `10` marker, `parity` on a
styled digit means the render missed the bar, no longer that the digit is styled, and `styled-word` means only that this
psd-tools cannot cut glyph pixels or make a pixel layer (the fork's text engine is missing): a styled word gets its FX layer.
Per card all-or-nothing; nothing bakeable = exit 0 with a one-line
report; any exception inside a card's steps refuses that card as `error` with the traceback in the report; a crash outside
them leaves nothing behind (`$RUNNER_TEMP`), the next push recomputes.

## 11. Races

- **/update_stats and PICKLE pushes.** The pipeline never writes pickles. Whatever order the bake commit and an editor's
  publish land, the database converges on the next `/update_stats`: the PSD blob changed -> re-parse -> the baked value
  stands because the field is gone; a field still present is re-applied over `parsed_values`. The old card goes to `old_stats`
  (expected); `diff_stats.py` before `/push` must list exactly the baked cards. The shared Lock (section 8) is the only
  coupling: `update_stats` queues no bake (the notice was removed 2026-10-21) and the bake queues no update.
- **New edits mid-bake.** Step 10's re-read at H' plus the non-force update; the editor's own 422 retry absorbs our commit.
  An editor rewrite that re-inserts a baked field now equals the PSD -> redundant -> dropped next run, no pixels (whether
  edit.js rebuilds from stale in-memory data after a 422: to confirm; a flicker of "Edited" at worst).
- **Two bakes.** Actions `concurrency: card-edits`; jetson: the Lock. A PSD pushed in the last 30 min is skipped. Site lag
  after a bake (old value until `/update_stats` + `/push` + build) is named by "Re-parse pending" (section 9).

## 12. Acceptance: the 10 edits

Made through the website on named cards, `--dry-run` first (artifacts reviewed), then live. 1-4: stats on pure-template rows
with no digit effect: Mini_Doomer hp 5 -> 8 (`Auxiliary/Minions/Mini_Doomer.psd`), one to 10, one to 0, one on a card with
`Lines` at top level. 5: name, Clot White "White CLot" -> "White Clot" (`Auxiliary/Minions/Clot_White.psd`), the one real typo;
already in the file since 2026-10-20 (MichaelJSr), so it is the first name job the live run meets, not an edit to make.
6-8: abilities with status words: one adds an unstyled word (Poison, Enfeebled), one removes a status word, one adds Blazing
(styled: after section 7's FX lands, else a third plain word). 9: a plain ability edit that reflows, after a deliberately
too-long version was refused with `overflow`. 10: a stat on a styled-digit card (`Auxiliary/Creatures/Mettaton_EX.psd`) once
round 2 lands, else a fifth plain stat. Per card the user opens the committed PSD in Photoshop on the Mac: no missing-font or
"update text layers" prompt (engine rasters accepted as is: to confirm), the digit right, a click into the Ability keeps its
runs; export the PNG and score it against the committed PNG with `~/Documents/Tools/psd-research/score/score_cards.py` regions:
0 bad pixels (> 24) outside the edited region, within the per-region bar inside, identical at 100%. Then jetson `/update_stats`
-> `diff_stats.py` lists exactly those cards -> `/push` -> no badge on the site. 10/10 = Photopea dropped; scores to `docs/Finished/`.

## 13. Implementation steps

| Step | What | Gate | Size |
|---|---|---|---|
| S0 fork | merge `feat/text-engine` into `cards`; `save(update_preview=)`; `lfx2` block copy; tag `cards-20261020`; push | harness score unchanged vs `cards_v1.json`; M4 round trip on 3 cards; goldens green under the fork; pytest green | S |
| S1 bot + site | `CardEdits.KIND` + `classify` (value-aware, redundancy); `PSD_FIELDS`, `pending`, `_psd` meta, `reparsePending`; badges | table test over all 19 fields (extends `tests/test_card_edits.py::test_subseries_edit_is_database_only` :97); the live file classifies to exactly the 7 name jobs of section 2 (dry run, no commit) and nothing else; a staged hp edit shows "PSD pending" | S |
| S2 bot | `psd_regions.py`; `psd_edit.py` stats path: baseline, toggle, viewport render, delta merge, gates `bytes`/`region`/`reparse`/`idempotence`; `--dry-run` artifacts | Mini_Doomer hp 5 -> 8 dry run: 2 flag bytes, parser reads 8, 0 pixels differ outside the row mask, rerun does nothing | M |
| S3 bot | text path: name and ability via `set_text`/`set_span_style`/`rasterize`, chart fills, `font`/`overflow` refusals | name edit on Mini_Doomer and 3 single-Ability cards pass every gate; a too-long ability refuses | M |
| S4 bot | `card_bake.py`: tree, blobs, `editsFile` port, commit (API + SSH), report, `_psd` | today's Card Edits.json round-trips byte-identical; a dry commit on a throwaway branch (owner-run) | M |
| S5 TTSCardMaker | `card-edits.yml` dry-run, then live with `gh workflow run --card`, actor guard, tts-save retry; the push trigger goes live only after question 7 (else the first push run reprints the 7 titles) | one stat bake on Mini_Doomer lands PSD + PNG + one-line file diff; site.yml and tts-save.yml run; `diff_stats` lists that card | S |
| S6 bot + fork | idea 14: FX regeneration (glyph boxes, pixel cut, chart effect block) **code done 2026-10-21**, no fork change (section 7: Gasoline IoU 0.966 / Cannon 0.937 with the JSX cuts, coverage 1.0, Mini_Doomer ability edits with FX pass every gate); idea 12 and idea 13 **code done 2026-10-21** (section 7: Mettaton_EX hp 5 -> 8 passes every gate; the Cannon writes and re-parses but its bevelled digits refuse `parity`) | Gasoline and Royal Eradicator FX layers within the validator's 0.95 coverage and the region bar; Mettaton_EX rows pass parity; goldens refreshed | L |
| S7 jetson | a manual fallback: fork in the bot venv, Lock, SSH commit, Discord report. **Code done 2026-10-21** (`commands/card_runner.py`, `/bake_cards`, `bake_lock`; the edits-sha notice landed and was removed the same day: the user chose ONE live runner, the Action, and no notice on the jetson); the venv install only if the fallback is ever needed | only if ever needed: one bake with Actions disabled; CPU/RAM peak recorded | M |
| S8 user | the 10 edits of section 12 | 10/10; results in `docs/Finished/`; Photopea retired to an archive branch | S |
| S9 bot + TTSCardMaker | star edits on folder-starred cards (creatures, items, field cards): move PSD + PNG to the new `N Stars` folder; aliases, `old_stats`, authors and the TTS save follow (question 4; ledger L14). **Planned**, after round 1 (row added 2026-10-21); a move is a PSD/PNG write, so it waits for the hold | proposed: a `/tmp` dry run moves one card: PSD + PNG at the new path, the old path gone, the parser reads the new stars, `diff_stats` lists only that card, the TTS save pins the new path | M |

## 14. Open questions for the user — answered 2026-10-20

1. **Yes**: the parser reads `Text/Name` into `display_name` (edits still override); the 22 name edits that equal the print
   become redundant and are dropped by the bake's redundancy rule; a baked rename leaves the file. **Done 2026-10-21**
   (`PARSER_VERSION` 2: 157 cards gain a `display_name`; 20 edits redundant now, 21 with Thunderfury; Sans and Papyrus have no
   readable `Name` layer; the next `/update_stats` re-parses every card once).
2. **Yes**: saved PSDs regenerate the flattened preview (`update_preview=True`). 2026-10-21 audit: the regeneration is 9.4 s of a
   23 s bake (a full-canvas composite); the merged PNG the driver already holds is that flattened image, so a fork
   `save(preview=<image>)` would keep the decision at no render cost (fork change, not done: the worktrees were in use).
   2026-10-21: round 3's fork branch `perf/save-with-preview` (98804ede, head 6f91981e) implements `save(preview=)`; `card_bake` still
   calls `apply_jobs(..., update_preview=True)` and switches to passing the merged PNG once the new pin tag is installed.
3. **Top row 10 + bottom row N-10** (Royal Eradicator's layout; the parser already sums it). Sonic is converted once to that
   layout (standardization work; refused until then). `EXCESSIVE_STAT_EXCLUSIONS` goes. **Done 2026-10-21**: the list is
   deleted; Sonic and Royal Eradicator Main Cannon carry `allow_high_stats` in the edits file (the only two cards above 10).
4. **Pipeline moves the files** for folder-starred cards (PSD + PNG to the new `N Stars` folder with aliases, old_stats,
   authors and the TTS save following) — after round 1 (step S9).
5. **Commit the Mac-only font files to `MDW/Fonts`** (the user's call; licence risk noted), so every runner renders every card.
6. **Restyle to the chart and hide the overlays**: an ability edit normalizes off-chart status words to the chart's colour
   and effect and hides the hand-made overlay word layers it replaces (part of S6).
7. **Reprint six titles** (Clot White's typo, the five PoD trailing dots); **Thunderfury keeps its brackets**: its entry is
   aligned to the print (redundant after 1). The push trigger can go live once that entry is fixed (S5). **Aligned 2026-10-21**
   ("[Thunderfury: Blessed Blade of the Windseeker]", written through `card_bake.dump_edits`).

## 15. Known wrong paths (fenced off)

- `force=True` for a baseline render: redraws vectors, `LINE` rows differ by up to 106. Use `ignore_preview=True`.
- `layer.composite()` for the delta: no backdrop, no parent-group effects (the `Text` group's Stroke and DropShadow).
- (Until 2026-10-21) removing a `name` field after a bake: the site showed the file stem again because the bot never parsed
  the title; now it does, and a baked or printed name is dropped like any field. Renaming the
  PSD/PNG instead: breaks `aliases.pkl` targets, Card Authors/Edits keys, `old_stats`, the TTS save.
- A separate status file or a `_`-key filter in `build_site.apply_edits`: both readers already drop `_`-keys and `editsFile`
  keeps unknown keys; `_psd` in the edits file is enough.
- Checking out card folders in the Action (16.4 GB) or reading PSDs from `raw.githubusercontent.com` (lags the tree).
- Fill-only recolouring of a styled status word (15 of 66): not Photoshop parity. Fenced since 2026-10-21 by construction: a
  styled word is coloured and gets its FX layer in the same edit (section 7 idea 14), or the card refuses `styled-word`.
