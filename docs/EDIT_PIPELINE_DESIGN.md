# Edit pipeline design (2026-10-20)

**Status:** designed 2026-10-20, questions answered the same day; Task 3 steps 4-7 (`docs/NEXT_SESSION_PLAN.md`); implementation running. **PSD writes on hold (user, 2026-10-20): other developers are editing the PSDs, so S5 live runs, S8, the title reprints, Sonic's conversion and star folder moves wait for the user's go; code and dry runs on /tmp copies continue; the intended PSD changes are tracked in `docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md`.** Synthesis of three
independent designs and two judge rounds: the "runner" contract (one CLI for both runners, `_psd` status in the edits file),
the "robust" gates (parity bar, record diff, outside-region equality, job key), the "mvp" scope (template cards only, fill-only
status words first, refuse the rest). Numbers measured 2026-10-20 unless marked "to confirm". `/tmp/` research files vanish on
reboot; harness and notes: `~/Documents/Tools/psd-research/`.

**Live since 2026-10-22.** The user lifted the hold for the bake ("you can bake the five title reprints and switch on
automatic baking. let's use 30 minute cadence"): the first live run (39761732714) baked Clot White and the four PoD
trailing dots in 26 commits, one per card. Since then a run makes one commit, bakes at most 40 cards (the rest wait for
the next run), and `card-edits.yml` runs every 30 minutes, live, behind the `pending` gate, telling Discord when a scheduled
run fails (section 8). Since 2026-10-23 the website can also preview one card's edit: the same bake on the runner, nothing committed to main, the image on the branch `previews` (section 16).

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
  `Text/Name` layer, else `Name` layers in the title region, else a layer named after the card, else the one layer in
  the title region named after its own text, as Photoshop names a new layer (2026-10-23: Papyrus, Sans; not on rule
  pages; any visible text layer there, in Text or not, so a card with no other title and one header label named after
  its text would read that label: the old-vs-new census is the guard); 157 of 813 cards, shown as printed, letter case
  included: the user's call of 2026-10-21; on 2026-10-23's 812
  PSDs 117 before the own-text rule, 119 with it); a Card Edits.json `name` still wins, with the title kept in
  `parsed_values`. 552 of 651 cards have a visible `Name` text layer, 153 name the layer
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
  (2026-10-21, parser read: 20 are redundant, 21 with Thunderfury aligned; Sans and Papyrus stayed: their title layers are
  named after the print, not `Name`; since 2026-10-23 the parser reads a layer named after its own text, so they are
  redundant too: a dry bake drops both)
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
| `ability` | PSD + database | `set_text(keep="runs")`, status words filled from the chart, `rasterize`; the one visible `Ability` text layer, in any group or at the root (2026-10-23), no other layer printing its words where they show (a copy under the art does not count) | field removed |
| `name` | PSD + database **when the value differs from the printed title** (the fresh parse's `display_name`, else the file name), else database-only | `set_text` + `rasterize` on the one layer the parser reads the title from (`PSDParser.title_layers`: `Text/Name`, else a `Name` layer in the title region, else a layer under `Text` named after the card, else one named after it elsewhere in the title region, letters and digits alike: the status effects' top-level titles; 2026-10-23; else the one visible text layer in the title region named after its own text, as Photoshop names a new layer: Papyrus's `Text/The Great Papyrus`, Sans's `Text/snas.`; several such, Knight Lost A Lot's stacked words, read as none, and a rule page's heading never counts; 2026-10-23, owner: which layer holds the title is the developers' freedom). A bake into such a layer renames it after the new title, as Photoshop renames a layer it named (`Edit.renamed`; `gate_bytes` allows exactly that name), so the parser still reads it; refuse none, several (an epithet), a title split over layers or printed again by another layer that shows (Doctor Zomboss's copy under the art does not) (unless the value is already what the bot shows: redundant) | **field removed** (since 2026-10-21 the bot reads the title into `display_name`; the 21 redundant names drop at the first live bake) |
| `stars` | PSD + folder (creatures, items, fields: the `N Stars` folder wins; minions, aux items, N.M.E.: star layers) | star layers shown, hidden, moved or copied in the one `Stars` group (below); folder-starred cards refused for that field (a move: S9) | field removed |
| `types` | PSD (icon layers or font tokens) | header type icons hidden, moved, shown from the card's own hidden layer or made from `Types/<name>.png` (below); a token row refused for that field | field removed |
| `creatures` | database-only (`CardEdits.KIND`; the per-creature rows of Pix, Tainted Lazarus, Twin Emperors stay hand-edited) | never touched; a stat edit beside it is refused (creature 1 sets the stats) | stays |
| `type` `subtype` `series` | folder | never touched | stay |
| `subseries` `note` `aliases` `dup_of` `hidden` `problems_ok` `allow_high_stats` | database-only | never touched; `_by`/`_at` kept | stay |

**Stars and types (2026-10-23, owner: "teach the bake instead of blocking the card").** `psd_edit.plan_stars` /
`plan_types` work an edit out on the open card; a layout they don't know refuses that field only (`Refusal.field`), and
`card_bake` bakes the card's other fields, leaving the refused one in the file with a refused verdict on the PSD the
commit leaves (final, so not pending). Stars, in the one `Stars` group that draws them: a count's own folder (`3 stars`
beside `5 stars`, the templates' 1-5 layouts) shows and the shown one hides; else, where the stars sit one by one, fewer
hide the last in reading order (rows top to bottom, each right to left), more show the card's own hidden stars first and
then copy the last star layer (`Layer.duplicate`, smart-object transform, `fxrp` and the layers clipped to it moved with
it) at the layout's spacing, the first that stays clear: rows of two with an odd one centred under them
(Status_Effect_Template's `3 star`) or over them (The Claw; the odd one moves aside when a pair fills its row, a lone
star is a pair's right one), a row or column carried on, a zigzag carried on (the bloons, Dark Esau, Mettaton), a lone
star's column. Fewer hide the first in reading order instead when only that leaves a layout (The Claw's top one). A
layer in the Stars group named after a type icon (Pike Beam's) is no star to hide or copy. Types, in the header row
the parser reads (image layers named after a `Types/` icon, above the midline): kept icons stay, a removed one hides and
the row closes up (a kept icon past the new end moves to the first free spot, centre on centre), a new one takes the
next spot: the card's own hidden layer of that type when it is the spot's size (the templates keep one per type), else
a pixel layer made from `Types/<name>.png` fitted into the spot and styled like the icon it replaces (its effects,
blending and the layers clipped to it, a Curves say). Spots past the last icon carry the row on at its spacing, or go
under a lone icon (Item_Template's column); when those would touch the text or stars, a row starts a second row under
its first icon (Abigail) and a lone icon takes its right. A star or icon must stay on the card, off the text and type icons (stars) or the text and
stars (icons), and the parser must count exactly the edit. Gates on top of the usual ones: `gate_outside` renders the
whole card before and after each stars or types edit (no ICC, no preview) and refuses a pixel changed outside the
edit's region (a star's clipped Curves left on another layer is how it showed up on Mini Doomer); `gate_bytes` takes a
copy only as its source's record moved (new id and name), a moved layer only with its rect and mask moved and its
transform and anchor blocks changed, a made icon only as a pixel record with the reference's effect blocks; the
re-parse reads the stars back and the types in any order.

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
+ blobs by sha. Output: at most one commit per run (section 5 step 10) and a markdown report on stdout in `tts_save._render` style. The
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
finds no PSD-kind field but the refused ones, whose verdicts are unchanged, and commits nothing: idempotence is the resume
mechanism, state lives in git. Refusals are data with a code, never exceptions; one bad card never blocks another.

0. **Snapshot.** Tree at head H; `CardEdits.parse` of the file blob; card key -> PSD path (same stem, case-insensitive like
   `png_for`), PNG path(s) and shas. The pipeline's own commits are not skipped (2026-10-22): a card refused `recent` or
   `head-moved` is retried by the next run even when nothing else landed on top.
1. **Fetch** every card named in the file that has a field outside the folder and database-only kinds (section 3): PSD,
   PNG(s), chart (once per run), fonts (cached). Gate `blob`: sha1(`blob N\0` + data) == tree sha. Refuse `recent` first
   (PSD or official PNG pushed < 30 min ago: don't race its author, and don't merge into a teammate's re-export unseen; by
   push time, not commit date: `activity` + one `compare` per run).
2. **Classify** (section 3) against the open PSD and a fresh parse. Refuse: `duplicate` (5 PSDs with numbered PNGs),
   `unsupported-card` (non-template rows, not one visible `Ability`, not one title layer the parser reads, overlay word
   layers the chart has no style for, `[type]` tokens on a card with image icons in the ability's gaps), `stat>10` (on
   the value alone; `allow_high_stats` is the website's business). (` FX: ` layers the chart has no style for were
   refused until 2026-10-23; they are rebuilt in their own style since.)
3. **Baseline parity.** Open with the fork; region per edited field (section 6); `psd.composite(viewport=region,
   ignore_preview=True)` vs the official PNG crop. The bar: bad pixels (|diff| > 24, `score_cards.py THRESH`) <= 0.5%
   and mean abs <= 2.0 (Mini_Doomer row: mean 0.62, max 2; 1.0 until the user raised it on 2026-10-22, section 7 idea 12).
   **Since 2026-10-23 a miss of the bar bakes and is told** (the owner: "let's not refuse bakes for failing the pixel check
   anymore. rather, we should improve psd-tools and fix these as we come across them"): the report's card line and the
   commit note say, per edited box, `name baked with a pixel difference: bad 1.97%, mean 1.60, max 217`
   (`psd_edit.miss_notes`). Gate `parity` is now only the safety floor: more than 25% of the region's pixels past 24
   (`PARITY_FLOOR`) is a broken render, refused before anything is written. So is a box that misses the bar and whose
   merge replaces no pixel (`<field>: the render shows no change`, review 2026-10-23): the fork drew nothing new there,
   so the PSD would carry the edit and the PNG the old card. Only the replaced pixels no other edit's region reaches count
   (`own_pixels`, review 2026-10-23: a retitle in a region stretched over a blank ability made its merge non-empty and it
   baked); when the others cover the whole region, all count. A blank ability box stays under the floor on 9 of 24
   sampled cards (Shield_of_the_Guilty 10.7%, Clot_Blood 12.5%, Lost_Kin 16.6%, Rusty_Knife 19.1% ... Grey_Prince_Zote
   23.6%); the 8 re-run are all refused now. A box within the bar that draws no change (a case fix under all caps) still
   bakes. The other floor is the data one, step 8: what the bot reads back from the baked PSD must be the edit. Before
   the ruling a row the fork rendered wrong (class A) was refused here. Measured 2026-10-23 with a real edit through
   every gate on the bot-gate scan's 118 parity misses (35 titles, 30 abilities, 53 rows of 15 stat cards; frozen corpus
   6e50c3ee, fork r6-color-glyphs b8739c37; `~/Documents/Tools/wt/scratch/soft-parity/softscan.py`): the old bar refused
   66 of them in the bake's own regions (the scan's boxes are wider, so 51 already baked); now 117 bake, those 66 with a
   note (max bad 6.35%, Corruptor's spd; re-render max 5.40%), none hits the floor, and 1 is refused either way for its
   badge layer (PoD06Rainbow_Bloon's ability). The fewest pixels one of those 117 replaced is 2,287, so the no-change
   rule refuses none of them. **What the floor lets through, told** (review, 2026-10-23): a stat row drawn 8 px off is
   8.5-9.7% bad (Mini_Doomer, Lich_King; 3 px: 12.8-17.5%), a title 3 px off 19.8-25.9% (The_Freezer, Mini_Doomer,
   Lich_King; 8 px off and blank titles are refused, 28-54%; blank rows 72-84%). A wrong font passes it, as the baseline
   is Photoshop's saved pixels and no gate redraws them: set_text_layer refuses `font` first when a layer's whole redraw
   overlaps its saved ink under FONT_IOU_MIN 0.3 (drawn_in_font; review 2026-10-24: Dull/Shiny Knife, Dark Tentacle and
   Snow Angel's titles drawn in the other Omori font, 0.12-0.15; the next of 1,453 layers 0.40). A missing inline badge
   was not measured; it should come out at or under a blank box. Not caught: a blank ability whose edit also changes
   overlay words or FX layers (the merge replaces their pixels), or a new render blank while the old one draws. Tighter
   options are the owner's: a mean ceiling (blank boxes: mean 17.7-32.5; the unbakeable list's misses: at most 7.39,
   Dark_Bramble's title) or the floor counted over inked pixels only.
4. **Apply.** Stat: visibility toggle; the row's effect follows the visible digit (idea 12, 2026-10-21): the hidden digit's
   `lfx2`/`lrFX` blocks move onto the shown digit's record, replacing a block whose effects are all off (the block flag is not
   enough: Mini_Doomer `Hp Dark/2` has `effects.enabled` True over an InnerShadow and a GradientOverlay both `enabled=False`);
   a shown digit with its own enabled effect keeps it and the hidden one's is cleared; to 10 nothing moves; from 10 the one
   hidden styled digit's block comes back, several such digits refuse `unsupported-card`; a hidden styled digit beside a visible
   unstyled one (Mettaton_EX Atk 6) is left alone, so the path matters (7 -> 10 -> 4 styles 4 with the 6's effect, 7 -> 4 moves
   nothing). Two rows (idea 13): Top shows the value and Bottom 0 up to 10, then Top none
   and Bottom value - 10 (`psd_edit.set_stat`). A bar with a digit mask wholly off the canvas (or empty) refuses
   `unsupported-card` before anything toggles, naming the digits or the whole bar (2026-10-23: Green_Mage's four bars sit
   below its 2400 px canvas and its printed cells are pixel layers; the empty render region was a `ValueError`, an
   `error` that failed the run). A mask that starts on the canvas and reaches past it (Pure_Vessel) bakes.
   Ability/name: `set_text(keep="runs")`; for each chart status name in the new text,
   `set_span_style(start, end, fill=(r/255, g/255, b/255))` per occurrence (`fill` takes 0..1, `layers.py:3113-3128`; the chart
   swatch is 8-bit; since 2026-10-21 every occurrence in the text, edited or not: the chart is the authority, question 6;
   the fill only: an FX layer keeps its own style, q6 2026-10-23 below);
   `rasterize(fonts=FontMap(dirs=[MDW/Fonts]))`; on a fork with the line splice (`psd_edit.splices()`), `rasterize(keep=capture(...))`
   redraws only the changed glyphs (`SPLICE_UNIT`) and keeps Photoshop's raster elsewhere (`Edit.band`). Then the FX layers
   (idea 14, 2026-10-21; BUG 2, 2026-10-23): the bake does what `Effect Layers.jsx`'s Update all does. The model, from the
   JSX: an FX layer is a pixel layer `<base> FX: <phrase>[ | <phrase>...]` ("#n" after a phrase: only its nth occurrence)
   holding the base text layer's own raster under a key mask: a copy of the text with the phrases' runs white and every
   other run black, red over black at level 1 (`phraseRanges`: any case, any blanks or forced breaks between words, whole
   words when there are any, else every match: Monkey D Goofy's `1#1` is the "1" of "3T1"). The icon font's colour glyphs
   keep their SVG colours in any key, so they are taken out (an all-black key's ink, TTSCardMaker 11faa634) unless the
   phrases meet an icon-font run: a badge layer (`Ability FX: (1CD) | (2U)`, queue A's gradient cards, no style of its
   own; it restores the badges' colours over the Ability's Gradient or Color Overlay) takes every colour glyph on purpose.
   The red key leaves a colour glyph's red-less pixels out (a typed `[undead]`'s black eye sockets and jaw, a
   `[feather]`'s green; the JSX's too), which then print in the overlay's colour, so since review 2026-10-24 a badge
   layer's key also takes its colour glyphs' silhouettes where no other glyph inks (`_phrase_mask(badge=True)`: a key
   drawn without colour glyphs, `Options(color_glyphs=False)`, each one its glyf outline, the icon's alpha past 110).
   The style is the chart's (Add), the overlay's (Convert overlays, whose words take the overlay's colour: its first drawn
   run, 8308103f) or the old layer's (Update all), opacity, fill, blending, visibility and place kept. Per old FX layer
   (`psd_edit.plan_fx`, `fx_kind`): a badge layer (`"badge"`), a styled chart status's (`"status"`) or a layer carrying a
   styled chart status's style on another phrase (`"styled"`: Goofy's `1#1`, Clot Gold's `1 Become Back Money`, the
   enabled effects compared by `_style_key`, a disabled pattern fill apart) is made again, and since 2026-10-23 any
   other too, in its own style (`"own"`: Smoldering's stroke, +2 DEF, Despair, Mr Krabs' style-less `2T`...; the
   owner's ruling of ~17:40, the 2026-10-21 refusal "FX layers not in a chart status's style" gone). Only an FX layer
   that is no pixel layer (`""`; none on any card; Photoshop would draw a type layer from its text) refuses
   `unsupported-card` before anything changes. Where each layer's phrases land is worked out before the text is set
   (`follow_fx`): `phrase_ranges` over the new text (a phrase that occurs more often is cut on every occurrence), but a
   `"styled"` or `"own"` layer (`ANCHORED`), whose style and colour belong to its own words, stays on the characters it
   covered (`kept_chars`;
   review 2026-10-23): Goofy's `1#1` is the first "1" of a text with no whole-word "1", so a "1" typed before it renumbers
   the layer (`1#2`, told), a whole-word "1" typed anywhere ("Draw 1 card.") would take the JSX's `1#1` off "3T1" and
   refuses `unsupported-card` ("would leave its 1 for another"), a styled phrase matched inside other words that would
   spread to a new one refuses too, and a numbered phrase whose occurrence the edit removed leaves (the layer deleted,
   told). A `"status"` layer, its own style kept (q6), lands only on its status's own words (`status_spans` of the new
   text; review 2026-10-23, round 3): never the "Frozen" of "Frozen In Time" or of "Unfrozen" (the JSX's literal rule
   re-cut Snow Bunny's Frozen layer onto the first word of a typed "Frozen In Time", " In Time" left flat in the swatch
   colour and no Frozen In Time layer, and onto part of "Unfrozen"): those it leaves ("'Frozen In Time' is no Frozen",
   the layer deleted when nothing is left), an unnumbered one is renamed by number when a word of its own stays
   (`Frozen` -> `Frozen#2` in "Frozen In Time or Frozen", the JSX's own form for some occurrences), and a numbered one
   follows its own word, renumbered, as a styled one does: Standstill's `Frozen In Time#2` (line 2) stays there as
   `#3` when "Frozen In Time, Frozen In Time." replaces line 1, `#1` follows the word it kept (`#2`), and the typed
   one gets a new `#1` of the chart's style (the literal names moved `#2` to line 1 and gave line 2's word a new layer
   of the chart's style, untold). Two layers renumbered onto one name refuse. `_rebuild_fx` cuts each in place from the new raster and keeps its record (id, name, flags,
   opacity, blending, clipping, blocks, place): only rect and channels change, written by the fork's raster writer
   (`TypeLayer._store_raster` on the pixel layer). The key is the engine's layout; on a splice the raster under it is
   Photoshop's for kept glyphs, whose anti-aliasing reaches a pixel past the engine's on some faces (The Radiance's
   Perpetua: 246 of 5,210 px), so the mask takes the raster's own ink one pixel around the key's that is no other glyph's
   (the complement key's, grown a pixel; `_phrase_mask`): the Photoshop cut footprint exactly on The Radiance, Mini
   Doomer, Goofy, Star Cannon and Gasoline. A layer none of whose glyphs was redrawn (the band misses its box) whose phrases
   cover the same kept characters (`kept_chars`, `_kept`) keeps its record whole, unless it holds pixels its words' cut does
   not (`_holds`: the B1 batch's leaked badge pills, std-batch2 FX.md), then it is cut again clean. A phrase gone from the
   text leaves the name (the parser flags a name the text lacks; the report says why: "(1GCD) left the ability", "the
   ability has 1 Frozen In Time now"); a layer with none left is deleted; a non-badge layer whose words now fall in a
   badge refuses `unsupported-card` (the JSX would cut every badge). Since 2026-10-23 (owner, ~17:40) a typed badge is
   always set in the icon font (below), so on a badge card it joins the badge layer as a real badge (Mini Doomer's
   "Then (BOT)." bakes; the review's refusal of a badge typed as body text, whose letters the JSX's cut would have lifted
   out of the Ability's overlay, white, only stays as a safety net for text the font draws no badge of). A badge the
   edit typed in the icon font joins the badge layer's name (each badge layer names every badge of its card; the layer
   draws every colour glyph): PoD11ZOMG's "(2U)" made "(3U)" renames `(1GCD) | (2U) | (2CD)` to `(1GCD) | (2CD) | (3U)`.
   So does one edited by deleting letters (review 2026-10-23: Turtle's "(1GCD)" made "(1CD)" kept every character, did
   not count as typed, and the layer, its phrase gone, was deleted: a blank pill under the Color Overlay, every gate
   passing): any icon-font badge or token no badge layer names joins it, and a badge layer with no badge left while the
   icon font still draws something refuses `unsupported-card`. Gate `gate_fx` checks it independently (`bare_icons`):
   under a tinted Ability (`_tinted`: an enabled colour, gradient or pattern overlay at any opacity, or a blend other
   than Normal, its own or a group's) every icon-font glyph is covered by a badge FX layer, every ink pixel of it
   (99% let a skull's eyes glow; review 2026-10-24), unless it was bare before the edit too (Blood Bomb's Pin Light
   `[machine]`, the one such card on main). An `"own"` or `"styled"` layer whose
   occurrence the edit changed rather than removed (characters of it kept, its phrase gone: Flameblade's "Attacks
   inflict 2T" made "3T", Palace of Shadow's "+2 DEF" made "+3 DEF", Mr Krabs' "2T1" made "3T1", Morpho Knight's
   "+2/4/6 ATK") refuses `unsupported-card` (ruling B: they stay on the characters they covered, refused where that is
   ambiguous; deleted, their style was lost and the report said "its words left the ability"), and so does Clot
   Gold's "2T1" made "2T2" (its styled `1 Become Back Money` was deleted, told, and a chart-style one made); an FX
   layer phrase the ability lacks before the edit ("Frozen In Time #1", a space before "#": no number to the JSX)
   refuses too (none on main; it crashed `plan_fx`). So does an `"own"` or `"styled"` phrase the edit deletes and
   types again elsewhere (Palace of Shadow's "+2 DEF" moved, or its whole line: the layer would leave every character
   it covered for typed ones; review 2026-10-23, round 2) and one the edit glues letters onto where a word of it ended
   (The Watcher's "Despair" made "Despairing": "ing" printed plain beside the styled half; Goofy's "3T1" made "3T10").
   A typed word that joins an unnumbered layer whose style is no chart status's for it (an `"own"` or `"styled"` layer,
   or an L7 `"status"` layer in its old overlay's style: `FxPlan.own_style`) is told: "Ability FX: Smoldering also
   covers the typed 'Smoldering', in its own style", or "moved to the typed 'Blazing'" when none of the layer's old
   words stayed (Doctor Zomboss's Blazing swapped with Frostbite). A
   rebuilt layer keeps its own style, the chart's or not (q6: own style kept, unattended call 2026-10-23, owner to
   confirm; the owner's standing ruling is to give the developers freedom and handle their choices generally): the 8
   L7 cards whose layers kept their old overlay's style are no longer restyled to the chart (section 7, BUG 2), and only
   a new layer takes the chart's (below). Text typed into a `"styled"` or `"own"` layer's own
   words (a match holding characters it covered), or a typed copy that joins it, takes the phrase's colour, its first
   drawn run (8308103f), but a status word among them keeps the chart's swatch (the owner's 2026-10-23 ruling). A
   stroke or shadow alone draws over the words' own fill: a copy of Palace of Shadow's blue "+2 DEF" printed in the
   body's grey and one of Morpho Knight's red "+2/4/6 ATK" in yellow, every gate passing (review 2026-10-23, round 2).
   A copy joining a layer whose letters are in more than one colour refuses (none on main). Gate `gate_fx` checks it
   independently (`gate_fx_colours`): each `"own"` or `"styled"` layer, matched by layer id, covers letters (status
   words apart) of only the colours its words had before the edit. A styled status word no rebuilt layer holds whole (its status's layer on
   exactly it, or a `"styled"` phrase around it: Clot Gold's `1 Become Back Money`; one a layer covers in part refuses,
   where counting any overlap left " In Time" without a layer) gets a new one (`new_fx`: `Ability FX: Blazing` for every occurrence, `Blazing#2` when only some are uncovered; one a digit touches,
   "2Blazing", beside a whole-word one has no JSX name and refuses `unsupported-card`), cut the same way, the
   chart layer's `lfx2`/`lrFX` deep-copied, a fresh layer id (the seed resource moves past it), directly above the ability's
   top-level group (`Text`; the layer itself at the root), where the JSX's Add puts it. A hand-made overlay word layer over
   the ability is hidden when it names a styled chart status, else the card refuses `unsupported-card`. The report tells
   every FX layer deleted, renamed or cut again for a leak (`Edit.notes`, `Baked.fx_notes`). `styled-word`
   remains for a psd-tools without the engine pieces (`psd_edit._fx_engine`) and for FX words with no pixels.
5. **Render + merge** (section 6). Gate `region`: pixels differing between old and new PNG lie inside the padded region and
   outside it the new PNG is byte-equal to the old. Gate `shape`: PNG size and mode unchanged.
6. **Save.** Preview current (an honest flattened image in the committed PSD: Photoshop's preview with the merge's pixels via
   the fork's `save(preview=)`, 0.13 s; regenerated where it can't, 9-11 s) or `save(update_preview=False)` (stale preview):
   question 2 decides once, for every card; no flag. Gate `bytes`: compare parsed layer
   records old vs new (names, flags, channel lengths, EngineData), not raw offsets (a text edit shifts every later record): a
   toggle changes exactly the two flag bytes of `N_old`/`N_new` plus the effect blocks its Edit declares moved (the same bytes leave
   one record and land on the other, or leave only when cleared), a text edit one record + its channels; an ability edit's FX
   layers: exactly the declared `<Ability> FX: ...` records gone and the new ones come (pixel records, transparency
   channel first, carrying the chart's `lfx2`/`lrFX` bytes), a rebuilt one its rect and channels only, plus its unicode
   name when a phrase left; never an effect block, its blending or opacity (`_fx_rebuilt`, q6), one flag byte per
   hidden overlay; records are paired in file order once those are taken out (`gate_bytes`). Gate `size`: PSD within
   +-20% of the old blob. Reopen: layers unchanged but the toggled digits; one digit per edited row; `layer.text` == the edit.
   Since 2026-10-23 a face an edit added (`Edit.fonts`: the icon font on a card without it) may join the end of Txt2's
   FontSet, nothing else there.
   Txt2 (2026-10-22): Photoshop reads a type layer's text from its story in the document's `Txt2` block (the layer's
   `TextIndex` is the story's index) and rebuilds the layer's TySh from it on save, so the five titles baked that morning
   (TySh only) reverted when the card was opened (`B.F.B.` -> `B.F.B`). The fork's `set_text`/`set_span_style` now write the
   story too (`fix/txt2-sync`; Photoshop redoes the story's glyph layout itself); gate `bytes` also holds `gate_stories`:
   every story holds its layer's text, only the edited layers' stories changed, the rest of Txt2 and every other
   document-wide block are the old bytes. A card that already has a stale story is refused `unsupported-card` until
   `python -m commands.txt2_repair` rewrites it; a fork without `TypeLayer.sync_story` edits no text (transient error).
   The gate also holds every type layer's `TextIndex` (a layer without one orphans its story) and the styles: per
   character, the story's font, size, colours and the other mirrored keys equal the layer's (`story_style_drift`; a card
   that already differs is refused `unsupported-card`). A forced line break (`\x03`, a newline on the website) stays
   one where the edit kept it (`keep_breaks`; the fork ends a paragraph run there, as Photoshop does). **Typed badges and
   icons (owner 2026-10-23 ~17:40):** every timing tag and every token the stoogedings font draws that the edit types
   (`icon_text`: a tag of `badge_tags()` in any letter case, "(1cd)"; a `[token]` the parser's vocabulary knows with a
   variant the font has, "[Fire:Sticker]", "[warp star]", "[fire:no]"; an icon character the font maps, "★", which the
   website writes as "5★": review 2026-10-23, it stayed in the body font, a missing-glyph box, every gate passing),
   anywhere in the ability, is set in the icon font
   at the card's own badge or token style (`icon_style`: its icon-font badges' or tokens' STYLE_KEYS, size, baseline
   shift, tracking..., at their commonest size, ties to the body's size, then the first; of those the commonest style:
   Pizza's 32 pt, not its odd 18.66 pt first badge, owner 2026-10-24); a card with none sets it in `ICON_FONT` (added to the layer's and Txt2's FontSets,
   `Edit.fonts`) at the text's own style, the most common card style (measured on main: 358 of 659 icon-font runs share
   every key with the text beside them, 442 its size; the templates have none). An old badge or token edited in place
   (a digit changed inside it: the new one holding most of its kept characters) keeps its own style; one typed right
   after a badge, which the engine gives that badge's style, takes the card's (owner 2026-10-24: Pizza's (2U) typed
   after its 18.66 pt (1GCD) stayed 18.66 pt; 57 cards on main have badges in two sizes). A typed `:sticker` or
   `:tint` (drawn in the text's colour) takes the colour of its kind on the card (`variant_fill`, owner 2026-10-24):
   the same icon's first (Palace of Shadow's [dark:sticker] white on its white first line, its [light:sticker] the
   body's grey), the edited text's before the other text's (the other read off the card before the bake: a title
   edit after the Ability's sees none the Ability typed); none on the card: the surrounding text's, as before; its
   kind in as many of one colour as of another (a [fire:sticker] on Palace) refuses `unsupported-card`. Text typed after a badge that the font
   draws no badge of ("(2 points)") takes the body style; letters left in the icon font that it draws no badge of
   (a badge replaced by "(abc)", or one cut down by deleting letters, "(1GCD)" made "(GCD)") refuse
   `unsupported-card`, unless the font drew them plainly before the edit too; a typed character its font has no glyph
   for (gid 0, `no_glyph`: a box in the engine and in Photoshop) refuses `font`. A `[token]` the website draws as an
   icon that the font prints as letters (`unlettered_icons`: text.js takes the base before the first colon, `norm()`ed,
   whatever the variant, the font ligates exact spellings only: "[fire:]", "[fire: no]", "[Fire.]", "[warp  star]")
   refuses `font` before anything changes, typed or made by deleting letters (review 2026-10-24: "[fire:]" counted as
   drawn, `ICON_VARIANTS`' "" being the plain "[fire]"'s, and baked as the letters, cut into Mini Doomer's badge layer
   white over its gradient, while the editor showed the fire icon; `token_variant` now takes only the font's
   spellings). The re-parse reads a typed "★" as
   "[star]" on the edit's side only (`canonical_tokens(chars=False)` on the parse's): one left in the body font reads
   back as "★" and refuses `reparse`. A typed badge or token under a tinted Ability (`_tinted`: an enabled Color,
   Gradient or Pattern Overlay at any opacity, or a blend mode other than Normal, its own or a group's) with no badge
   FX layer to restore its colours refuses `unsupported-card` (review 2026-10-23: only an opaque Normal overlay
   counted, so the 18 Pin Light Isaac items without a badge layer, Soul of Eden's typed (2CD) white on cyan, and Ice
   Staff's two 69% gradients printed washed badges; std-batch2's fix for Pin Light badges is a badge FX layer, which
   all 4 Pin Light cards with badges carry). On main 71 Abilities are tinted: 29 carry a badge layer (every badge
   layer sits on one), 41 have no icon, and Blood Bomb's `[machine]` is bare. `[type]` tokens refuse only on a card whose type icons are image
   layers the parser reads into the ability's gaps (`image_icons`: the website's text holds their tokens; Nebulous
   Precipice is the only one on main), no longer on every card without the icon font. The website's text spells each
   icon as the parser reads it ("[dark]" for "[dark:sticker]", "[fire]" for "[Fire]" or "★"), so `keep_tokens` spells
   an icon the edit kept as the layer does (review 2026-10-23: an ability bake on Palace of Shadow, Irithyll, Afraid,
   Corruption, Error Room, Inkhole, Blindness, Elemental World or Dark Bramble redrew their `:sticker`/`:tint` icons
   plain, every gate passing). **A token typed in the title (owner 2026-10-24)** is set in the icon font too, at the
   title's own size and colour (`icon_style` of the title, else the run the engine gives it), under the same gates:
   glows bake since the fork's `cards-20261024b` (they refused `font` before), `unlettered_icons` does not run (the
   website shows a title's "[fire:]" as its letters, as the card does), a title under an overlay, blend, opacity or clipped layer (`_tinted`: 367 of 728 titles on main, mostly
   gradient overlays; also a Satin, Inner Glow or Inner Shadow, and a group's overlays: The Radiance, Captain Spaceboy)
   refuses `unsupported-card`, words typed after a title's token take the title's style (`title_styles`), and `gate_fx` runs `bare_icons`,
   `see_through_icons`, `lettered_icons` and `unshaped_icons` on the title (`titles`, `_gate_icons`). A stroke, shadow,
   glow or bevel draws round the icon as Photoshop does. A timing badge typed there refuses `unsupported-card` (the
   ruling names tokens; the website shows a title as plain words): owner to rule. The parser reads a title's icons
   as its name (`title_layers`), not as a header's types or stars: the re-parse read Bench's "[fire]" as a fire type
   (no title on main uses the icon font, so nothing reads differently; psd_parser is a Cython module). An edit on a layer
   set in the stoogedings icon font sets typed `[tokens]` in that font; its glyphs are SVG colour glyphs. A fork that
   draws them in colour (round 6) bakes them; where the engine draws one in one colour (an older fork, or SVG it does not
   cover: warning `monochrome_emoji`) the bake refuses `font` unless an opaque overlay effect covers the layer (Mini
   Doomer's gradient).
7. **Fonts.** Gate `font`: no `MissingFontError` under `policy="raise"` and no `font_loose` on the edited layer; gate
   `overflow`: `report.warnings` has no `overflow` ("ability does not fit").
8. **Re-parse.** Gate `reparse`: offline parse of the new PSD; `to_dict()` == the old parse with exactly the edited fields
   replaced (ability compared after the parser's own whitespace rule on both sides, and since 2026-10-23 its tokens as
   the parser reads them in the icon font on both sides, `canonical_tokens`: a typed "[Fire]" reads back "[fire]";
   `classify` calls such an edit redundant; a colon with no variant, "[fire:]", which the font prints as letters and
   the parser reads as [fire], stays as typed since review 2026-10-24, so the gate no longer shares that rule); `alt_names` unchanged;
   `CardValidator.validate` (`commands/psd_parser.py:948`, imported from there) is a subset of the old problems minus
   `FIXED_BY[field]`; no `EFFECT LAYER *` (the validator's coverage >= 0.95 is the FX layers' gate); `STATS TOO HIGH` only when
   every stat above 10 sits on Top + Bottom rows (idea 13; a raise from <= 10 adds it there and is not a new problem, fixed
   2026-10-21 review). FX layers back as declared after the reopen: the names in order, the new ones
   visible, styled, directly above the ability's top-level group, no overlay word layer showing. Gate `gate_fx` (BUG 2,
   2026-10-23) measures them on the saved card with the fork's text geometry (`text_geometry`: each glyph's ink box in
   the new layout): every FX layer inks only inside its phrases' glyph boxes (a badge layer also the icon font's) padded
   by `FX_BOX_PAD` (3 px), covers `FX_COVER` (99%) of the Ability's ink in its words' boxes that no other glyph's box
   reaches (a badge layer every ink pixel of its badges and icons there: review 2026-10-24), one the edit cut holds the
   Ability's own pixels byte for byte, and a `"styled"` or `"own"` layer's name
   still reaches the characters of it the edit kept (`Edit.held`, independent of `follow_fx`; a range counts as kept
   only when every character of it is, `kept_spans`; `held_words` picks the layers by layer, not name, since a renamed
   one may take a deleted one's name, review 2026-10-23); stray pixels count at any alpha (refused `reparse` naming the layer;
   `test_fx_gates_refuse_a_wrong_cut` shifts, crops, repaints and fades a rebuilt layer, and `gate_bytes` refuses its
   opacity or name changed). With the chart, `gate_status_words` (review round 3, independent of `follow_fx` and
   `new_fx`): every styled status word of the saved Ability is held whole by an FX layer's `phrase_ranges` (its
   status's layer on exactly it, or another phrase around it), and a status's layer lies on its status's own words only.
   And `lettered_icons` (review 2026-10-24, independent of `icon_text`): the icon font draws no more letters than before
   (an icon-font glyph the shaper made of one printable ASCII character, no ligature: "[fire:]" set in the font).
   Name: the title layer's read-back == the edit and the stem unchanged (the parser reads the new title into `display_name`; `gate_reparse` leaves that field out of the comparison).
9. **Idempotence.** Gate `idempotence`: classify on the new PSD + new PNG + rewritten file finds nothing to bake and the
   region render matches the merged PNG within the parity bar (the delta merge keeps Photoshop's pixels). Since 2026-10-23
   the render is held to the same floor (refused past 25% bad); a miss is told, as `(re-render)` when only it missed.
10. **Commit, one per run** (since 2026-10-22; `card_bake.land`). Re-fetch head H' and the file at H'. Per card: gate
    `head-moved`, its PSD and PNG at H' still hold the blobs it read, and gate `edit-changed`, its PSD-kind values at H' equal
    what was baked (never remove a value you did not render); a card that fails either stays out of the commit as a try-again
    refusal and the others still land. Tree on `base_tree` H': every remaining card's PSD and PNG (existing path and case),
    `site/Card Edits.json` with the baked and redundant fields removed and the `_psd` verdicts written (section 9), re-serialised
    by the `editsFile` port. Entry rule: an entry with no `FIELDS` key left is deleted whole, `_by`/`_at`/`_psd` with it (the port
    does this explicitly; `editsFile` itself keeps any non-`FIELDS` key, `edit.js:39-40`, and would leave a `_`-only entry);
    otherwise `_by`/`_at` stay. Author/committer `AutoCroissant <autocroissant@gmail.com>`, non-force ref update, 409/422 ->
    back to step 10 up to 3 times (`head-moved` for every card after that). Message: `Card edits: 5 baked, 20 redundant, 4
    refused`, one `- <card>: <what>` line per baked card (`- Goombella: spd 4 -> 5`; a box that missed the bar adds its
    note, `- The Freezer: name 'The Freezer' -> 'The Freezee'; name baked with a pixel difference: bad 1.97%, mean 1.60,
    max 217`, 2026-10-23), then one `Edit-by: <login>`
    trailer per baked card and editor, sorted (build_site.EDIT_BY counts an editor's trailers as their cards in the run's
    commit); shas, gate codes and the fork tag go to the report and `_psd`.
    Until 2026-10-22 it was one commit per card plus one for the refused verdicts (26 commits for the first live run).
    Never in a commit: pickles, `MDW/`, `TTS/`.
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
  not a looser bar. **Revised (the user, 2026-10-22):** mean 2.0 (`PARITY_MEAN_MAX`; bad 0.5% and max 24 unchanged), so those
  toggles, the Cannon's one-row hp (mean 1.06) and Z.O.M.G.'s title (1.28) pass the baseline; the 0.5% bad-pixel bar still refuses
  a row the fork renders wrong (the Cannon before round 4: 5-7% bad). **Revised again (the owner, 2026-10-23):** the bar
  no longer refuses; a miss bakes with a note, only past 25% bad (`PARITY_FLOOR`) is a render broken (section 5 step 3).
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
  copied; the hand-made overlay word layers naming a styled chart status are hidden (question 6), the others refuse. Since
  2026-10-22 an FX layer that is no styled chart status (a badge or token run, an unstyled word) refuses the edit too: it
  would be dropped (19 queue A cards carry badge layers; the FX tests run on Mini Doomer from before its `Ability FX: (BOT)`). Proof
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
  **BUG 2 (2026-10-23, `work/fx-rebuild`, not landed): Update all, not derived state.** The FX layers are no longer deleted
  and regenerated: each is cut again in place (section 5 step 4), so badge layers, chart-style word layers and the L7
  cards' converted overlays keep their records, styles and places, and a repeat is cut into the same layer (`Blazing`
  over both, as the JSX's phraseRanges does), no longer `Blazing#1`/`#2`. Dry census with every gate (UNBAKEABLE_CARDS
  BUG 2 block): 32 of the corpus's 34 FX cards and 58 of main's 68 bake an ability edit; an Update-all no-op leaves 47 of
  55 cards byte for byte in their FX records and PNG, the other 8 the L7 cards restyled to the chart (q6, then). **q6: own
  style kept, unattended call 2026-10-23, owner to confirm:** a rebuilt FX layer keeps its own style (`_rebuild_fx`
  restyles nothing; `gate_bytes` lets no effect block change on a rebuilt record); only a new layer takes the chart's.
  Re-run on the 8 L7 cards (copies of TTSCardMaker main, `~/Documents/Tools/wt/bf-fx-q6-scratch/`, `runs/{old,new}.json`):
  the Update-all no-op now leaves all 8 PNGs pixel for pixel and every FX record whole (before: 8,871-65,014 px
  changed, 0-31,725 past 24); the website no-op leaves 4 identical (Doctor Zomboss, Snow Bunny,
  Freeze Spell, Pandora's Box) and the other 4 change only where an off-swatch status word takes the chart's fill (the
  2026-10-21 rule; its FX layer cut again under its own style): Royal Eradicator Main Cannon 2,775 px past 24, Standstill
  2,438, Zero 575, Oryx 44, all within the parity bar; a moved word bakes on all 8, their FX records changing only rect
  and channels. Tests: `test_noop_ability_bake_keeps_an_l7_fx_layers_own_style` (Snow Bunny, PNG pixel-identical),
  `test_rebuilt_fx_layer_keeps_its_own_style` (both fail on pre-q6 43a54f9 on its "restyled to the chart's" note);
  `test_gate_bytes_keeps_a_rebuilt_fx_records_style` pins the rebuilt record's tuple and what may change (the pre-q6
  gate refused an undeclared effect block or blend change too). Review
  (2026-10-23): Update all's literal rule would move a chart-style layer named by count (`1#1`) or matched inside a word
  to a new whole word, so those follow their kept characters (`follow_fx`) or refuse; a badge typed as body text onto a
  badge layer refuses; census re-run with a phrase typed again and "Draw 1 card." in front (same results otherwise). Effect
  Layers.jsx's Update all run in Photoshop on 8 baked copies rebuilt the same layers (same boxes, style and place, IoU
  0.955-1.0, no differing pixel on the shared ink). Tests: `test_noop_ability_bake_keeps_the_badge_layer`,
  `test_ability_bake_moves_the_badge_layer_with_its_badge`, `test_ability_bake_drops_the_fx_layer_of_a_word_it_removes`,
  `test_ability_bake_cuts_a_leaky_fx_layer_again`, `test_goofys_styled_one_is_made_again`, `test_phrase_ranges_follow_the_jsx`;
  after the review (2026-10-23): `test_goofys_one_stays_on_its_one`, `test_a_badge_typed_as_text_refuses_on_a_badge_card`,
  `test_fx_gates_refuse_a_wrong_cut`, `test_new_fx_refuses_a_status_word_the_jsx_cannot_name`. Review round 3
  (2026-10-23): a `"status"` layer follows its status's own words (above); `test_numbered_status_fx_follows_its_own_word`,
  `test_status_fx_lands_only_on_its_status_words`, `test_new_fx_counts_a_word_covered_only_when_held_whole`,
  `test_kept_spans_need_every_character_kept`, `test_status_fx_layer_leaves_a_longer_status_word` (Snow Bunny, with
  `gate_status_words`), `test_numbered_status_fx_layers_stay_on_their_words` (Standstill; line 2's PNG is the no-op
  bake's). Merged with origin/main (1bf6892, then cc3cd8c; later merges in the phase 2 paragraphs): its title-rename check in `gate_bytes` now knows each rebuilt FX layer's
  declared name.
  **Phase 2 (2026-10-23 evening, owner's answers of ~17:40, same branch):** (1) FX layers in no chart status's style are
  rebuilt in their own style (`fx_kind` `"own"`, anchored as `"styled"` ones: they stay on the characters they covered
  and refuse where Update all would move them); the 2026-10-21 refusal is gone. Census on copies of TTSCardMaker main's
  8 such cards (Morpho Knight, The Watcher, Computer Virus, Mr Krabs, Palace of Shadow, Fire Staff, Ice Staff,
  Flameblade), six edits each with every gate (scripts and JSON `~/Documents/Tools/wt/bf-fx-p2-scratch/`,
  `census.py`, `runs/b.json`, crops `crops-b/`, fork `cards-20261023c`): 42 of 48 bake, 2 n/a; the website and
  Update-all no-ops leave 7 of 8 PNGs pixel for pixel (Mr Krabs: 285 px past 24 at the edges of "Become Back Money",
  its yellow recoloured to the chart's white under its gradient FX layer, the owner's chart rule); a word typed again
  joins its layer on 5 cards, told; refused: Mr Krabs's whole-word "2T" typed (its `2T` sits in "2T1"), 3 overflows.
  (2) Every typed timing tag and stoogedings token is set in the icon font at the card's own badge or token style
  (section 5 step 6); census on 17 cards (badge-layer, icon-font, plain, overlaid and image-icon cards), five edits each
  (a tag, a lowercase tag, a [type] token, a ":no" and a ":Sticker" variant, a badge and a token in front): 62 of 85
  bake, each typed icon at its card's icon style (Scapular's 19.96 pt shifted -12.52, Cabbage's tracking 100), the face
  added on the 3 plain cards that fit, badge layers renamed to name the typed badges (told) or already naming them;
  refused: Cat (overlay, no badge layer, 5), Nebulous Precipice (image icons, 5), 13 overflows (`runs/c.json`,
  `crops-c/`); regression on main's 68 FX cards x round 3's 4 edits: 234 of 258 bake, no former bake refused, 51
  former refusals bake (`runs/regress.json`). (3) `keep_tokens` (found by the census): the website's canonical tokens no longer overwrite the card's
  `:sticker`/`:tint` spellings (9 cards on main). (4) Review minor items: FX notes in the commit line and the website's
  verdict reason, `held_words` by layer, the refusal paths tested, a typed word joining a layer of no chart style told,
  `_numbered`'s `#0` read as every occurrence (as the JSX's `splitOccurrence`; the parser counts it as before). Tests:
  `test_fx_kind_names_how_each_fx_layer_is_made_again`, `test_own_style_fx_layers_are_made_again_in_their_style`,
  `test_own_style_fx_layer_takes_a_typed_word_in_its_style`, `test_an_own_style_status_word_keeps_the_charts_fill`,
  `test_held_words_follow_the_layer_not_its_name`, `test_follow_fx_refuses_what_would_move_an_anchored_layer`,
  `test_ambiguous_fx_layers_refuse`, `test_a_typed_status_word_joining_its_layer_is_told`,
  `test_a_typed_badge_joins_the_badge_layer`, `test_typed_icons_on_a_card_without_the_icon_font`,
  `test_typed_icons_refuse_where_they_would_print_wrong`, `test_icon_text_follows_the_fonts_spellings`,
  `test_icon_style_is_the_cards_commonest`, `test_an_edit_keeps_the_cards_own_token_spellings`,
  `test_parentheses_typed_after_a_badge_are_body_text`, `test_effect_layer_numbers_read_as_the_jsx`,
  `test_fx_notes_reach_the_commit_and_the_verdict`. The parser's `#0` change was reverted in review round 2 (it counted
  the same as `max(n, 1)` and made `psd_parser.py` a changed Cython module): no Cython rebuild.
  **Phase 2 review (2026-10-23 night, same branch; origin/main 8983719, then 8cbb211, merged in):** four silent wrong bakes, each
  refused or baked right now (section 5 step 4 and the typed-icons paragraph; UNBAKEABLE BUG 2 block for the census,
  `~/Documents/Tools/wt/bf-fx-r5-scratch/`): a badge edited by deleting letters joins its badge layer (it was deleted:
  a blank pill), with gate `gate_fx`'s `bare_icons` behind it; a typed badge under any tint (Pin Light, a 69% gradient)
  without a badge layer refuses; a typed "★" is set in the icon font and recuts the badge layer, a missing glyph refuses
  `font`; an edit inside an own-style or styled FX phrase refuses (ruling B). Also letters a deletion leaves in the icon
  font, an FX phrase the ability lacks ("Frozen #2" crashed `plan_fx`). Tests:
  `test_typed_icons_refuse_under_any_tint_without_a_badge_layer`, `test_a_badge_edited_by_deleting_letters_keeps_its_badge_layer`,
  `test_gate_fx_refuses_a_bare_badge_under_a_tinted_ability`, `test_letters_a_deletion_leaves_in_the_icon_font_refuse`,
  `test_a_typed_star_is_set_in_the_icon_font`, `test_a_typed_star_is_cut_into_the_badge_layer`,
  `test_reparse_reads_a_star_outside_the_icon_font_as_typed`, `test_follow_fx_refuses_an_edit_inside_an_anchored_phrase`,
  `test_an_edit_inside_an_own_style_phrase_refuses`, `test_an_fx_phrase_the_ability_lacks_refuses`.
  **Phase 2 review, round 2 (2026-10-23 night, same branch; origin/main be280e6, then 73dfc3c, merged in):** a typed copy joining an `"own"` or `"styled"` layer kept
  the body's fill, which its stroke or shadow alone draws over (Palace of Shadow's blue "+2 DEF" copied grey, Morpho
  Knight's red "+2/4/6 ATK" yellow, every gate passing): it takes the phrase's colour, and gate `gate_fx_colours`
  checks each such layer covers letters only of its words' old colours; a moved own-style phrase (deleted and typed
  elsewhere: the layer jumped to the typed copy, against "stay on the characters it covered") and letters glued on
  where a word of one ended ("Despair" made "Despairing", Clot Gold's "Money" made "Moneying", Goofy's "3T1" made
  "3T10": half a word styled, untold) refuse; a status layer whose old words all
  left is told "moved to the typed ..." (it said "also covers"); the FX notes reach a partial bake's refused verdict;
  the bake's icon vocabulary keeps only the manifest's icons (a Types/ icon the font lacks would print its letters in
  the icon font and read back as the token); the parser's `#0` change reverted (no Cython rebuild). Census on main's 18 cards with an own-style layer, 7 edits
  each (UNBAKEABLE BUG 2 block, `~/Documents/Tools/wt/fxr3-scratch/`): the former edits' results and notes unchanged
  (Palace's typed copy now blue); a moved phrase refuses on 4, glued letters on 9. Tests:
  `test_a_typed_copy_of_an_own_style_phrase_takes_its_colour`, `test_a_typed_copy_of_a_phrase_in_several_colours_refuses`,
  `test_a_moved_own_style_phrase_refuses`, `test_letters_glued_onto_an_own_style_phrase_refuse`,
  `test_a_status_layer_moved_to_a_typed_word_is_told_so`, `test_gate_fx_refuses_an_own_style_layer_over_a_new_colour`,
  `test_the_bakes_vocabulary_knows_only_the_fonts_icons`, `test_rebuild_fx_refuses_a_badge_in_plain_letters`,
  `test_fx_notes_reach_a_partial_verdict`.
  **Phase 2 review, round 3 (2026-10-24, same branch; origin/main 7ca804e merged in):** two silent wrong bakes fixed, and
  the cheap minor findings (dry census with every gate on copies of TTSCardMaker 7dcc36fd, fork `cards-20261023c`,
  `~/Documents/Tools/wt/bf-fx-r6-scratch/`: `census_bake.py`, `runs/{new,base,turtle,ice}.json`, the branch tip before
  the fixes in `base/`). (1) A typed token's red-less pixels stayed out of the badge layer, cut with the JSX's red key
  grown a pixel, and printed in the overlay's colour, every gate passing (`gate_fx` and `bare_icons` let 1% of a glyph
  out): Heavy Mole's typed `[undead]` left 39 opaque px of the skull (glowing yellow eyes), Turtle's 10, `[feather]`
  and `[nebula_star]` specks. A badge layer's key now also takes its colour glyphs' silhouettes where no other glyph
  inks, and both checks let no ink pixel of a badge or icon out. Census on the 31 badge-layer cards: a typed `[undead]`
  and `[feather]` bake on 26 of 31 each (21 and 18 before: the 13 `reparse` refusals were the 1% line), no opaque icon
  pixel outside the badge layer (each of the 39 bakes before left 2-39); the no-op and a typed `[fire]` give the same results and
  PNGs as before. Every manifest icon typed on Turtle: 85 of 91 bake (66 before), 6 refuse as before (`light`,
  `lightshadow`, `6star`, `7star`, `8star`, `silksong_star`: faint flare pixels past their glyph boxes); `[ice]` glued
  after the final "." bakes on Mini Doomer, Palace of Shadow, Scapular and PoD11ZOMG (refused before), and Mini
  Doomer's "[ice:no] and [Fire:Sticker]" too. (2) "[fire:]" counted as drawn (`ICON_VARIANTS`' "" is the plain
  "[fire]"'s): set in the icon font it printed the letters, was cut into a badge layer (white over Mini Doomer's
  gradient) and read back as [fire] while the editor showed the icon. `token_variant` takes only the font's spellings
  (`icon_text`, `canonical_tokens`, so `keep_tokens`); a token the website draws as an icon and the font as letters
  refuses `font` before anything changes (`unlettered_icons`: "[fire:]", "[fire: no]", "[Fire.]", "[warp  star]", or
  "[dark:sticker]" made "[dark:]"); census: " Deal 1 [fire:] damage." refuses on the 29 badge cards that get that far.
  Gate `lettered_icons` (in `gate_fx`, measured on the shaped text) refuses icon-font letters an edit adds; none of
  the 381 icon-font Abilities on main draws any. The parser now shares the rule (`psd_parser.token_variant`, round 3's
  rest below). (3) Minor: an edit inside
  a phrase whose letters are in several colours refuses (the fill would make them one, `gate_fx_colours` passing; none
  on main); `status_spans` finds no status word inside a `[token]` ("Stuffed" in "[stuffed star]"; none on main);
  `apply_jobs` edits the ability before the title (Fire Staff's name and ability together refused `bytes`: the title's
  record index was read before an FX layer under it was deleted; the review found Morpho Knight, Computer Virus,
  Undyne and Flameblade alike). Tests: `test_a_badge_layer_takes_its_icons_red_less_pixels`, `test_a_colon_with_no_variant_is_no_token`,
  `test_a_token_the_font_draws_as_letters_refuses`, `test_reparse_reads_a_token_with_no_variant_as_typed`,
  `test_gate_fx_refuses_letters_in_the_icon_font`, `test_an_edit_in_a_phrase_in_several_colours_refuses`,
  `test_status_spans_skip_tokens`, `test_an_ability_edit_goes_before_a_title_edit`.
  **Round 3's rest (2026-10-24 ~01:00, same branch; origin/main 2427a23 merged in):** the parser's `_read_icon_tokens`
  and `_flag_misplaced_tokens` take a token's variant as the font ligates it (`psd_parser.token_variant`, which
  `icon_text` and `canonical_tokens` import): "[fire:]" in the icon font reads as typed, an UNKNOWN ICON TOKEN, and
  is no misplaced token in the body font (no card on main has one: psd_parser is a Cython module, so a deploy
  rebuilds; no PARSER_VERSION bump, nothing reads differently). Gate `unshaped_icons` (in `gate_fx`, given the
  vocabulary): each `icon_text` span of the new Ability shapes to one icon-font cluster spanning it, new against the
  old card; independent of `icon_text`'s rule, which `gate_reparse` and the parser share (0 of the 381 icon-font
  Abilities on main fail it). A whole ability deleted on an FX card refuses as on any card (`reparse`: ABILITY TEXT
  NOT FOUND), its FX layers deleted with their words; `_rebuild_fx` refused `styled-word` ("no pixels to cut") before
  looking at them. One open per PSD per bake: `parse_card(psd=)`, card_bake shares its open of the old PSD with the
  parse, `classify` and `apply_jobs(psd=)`, which edits a `deepcopy` (as fast as a second open) and reads the old
  records first; `gate_bytes`, `gate_reparse` and the FX gates read the one reopened saved PSD. Dry census, every
  gate, copies of TTSCardMaker main, fork `cards-20261023c` (`~/Documents/Tools/wt/scratch/bug2-p3/`: `p2/`,
  `h/runs/`, `rev3/`): phase 2's runs b 42 of 48, c 62 of 85, regress 234 of 258 (14 n/a), each result as before; the
  68 cards' no-op and " Die." bakes pixel-identical to the tip before this (136 of 136). The review's cases (322 bakes):
  Turtle's 91 icons 85 bake (66), `[undead]`/`[feather]` on 8 badge cards 16 of 16 (10), `[ice]` glued at the end 4
  more, " Deal 1 [fire:] damage." 0 of 11 (11 baked as letters), Snow Bunny's blank ability `reparse` (was
  `styled-word`); on the 173 tinted badge-layer bakes no opaque icon pixel is outside the badge layer (uncovered.py
  counts 3 on Turtle's "[Fire:NO]." : the "."'s own edge inside the slash glyph's box), the 29 official cards 0. Tests:
  `test_parser_reads_a_colon_with_no_variant_as_letters`, `test_reparse_reads_a_token_with_no_variant_as_typed`,
  `test_gate_fx_refuses_a_token_not_shaped_as_one_icon`, `test_a_blank_ability_on_an_fx_card_refuses_as_blank`,
  `test_a_bake_opens_each_psd_once`.
  **Round 4 (2026-10-24, same branch; origin/main a8ece7f merged in):** a badge layer takes every icon-font character
  and every pixel of its glyphs at any alpha (`_glyph_ink`: the layout's glyphs of those characters rasterized alone),
  so no pixel of a colour icon's see-through, red-less glow ([support]) is left out of it (yet the glow still printed
  in the overlay's colour: corrected in round 6); `gate_fx` and
  `bare_icons` count every colour-glyph pixel past `FAINT_INK` (`_colour_ink`: two renders, every run red then blue,
  agree), to the nearest icon glyph's, and a badge layer may ink over them past its glyph boxes. `_tinted` counts an
  opacity or fill below full (The Patients). `status_spans` skips only what the icon font draws (a third argument:
  `icon_text` of a text to set, `icon_spans` of a layer), so "[Blazing]" is styled as the JSX styles it. Tests:
  `test_a_badge_layer_takes_a_colour_icons_see_through_glow`, `test_a_status_word_in_lettered_brackets_takes_its_fx_layer`,
  `test_apply_jobs_checks_each_typed_icon_is_one_cluster`, The Patients in `test_typed_icons_refuse_under_any_tint_without_a_badge_layer`.
  **Round 5 (2026-10-24, same branch):** `keep_tokens` diffs whole tokens, not characters (a token typed right after a
  kept `[unknown:sticker]` took its `]` and the sticker went plain: Error Room, Corruption, Elemental World, Palace of
  Shadow). A new FX layer goes above the Text group's clip chain (Claw's and Miasma's MagentaSmorg and Exposure 1 left
  the title and text in the PSD, not the PNG), and `gate_untouched` renders the whole card around the text edits: any
  pixel outside their regions refuses `region` (about 5 s a render). `_tinted` counts visible layers clipped to the
  Ability or a group above it (an adjustment or fill; a drawn layer over its box: Pizza's Multiply `Layer 12`), so a
  typed badge or token there refuses. `unshaped_icons` counts a layer without the icon font too, so the 10 plain cards
  that print tags as body text (Papyrus' `(1CD)`) take typed badges and tokens again. Tests:
  `test_keep_tokens_pairs_whole_tokens`, `test_new_fx_layer_goes_above_the_groups_clip_chain`, Claw and Miasma in
  `test_typed_icons_refuse_under_any_tint_without_a_badge_layer`, `test_a_plain_cards_body_font_tags_do_not_refuse_a_typed_token`. Full suite after it
  (79efb50): 1556 passed, 12 skipped, 0 failed; 60 sampled cards x 2 edits refuse no more than before (one fewer).
  **Round 6 (2026-10-24, same branch; review round 3 of phase 2):** round 4's fix was incomplete: a badge layer is a
  copy of the Ability's pixels over it, so a pixel at alpha a still shows a*(1-a) of the tinted Ability below (Heavy
  Mole's [support] olive-grey, [light] yellow-haloed, every gate passing). `see_through_icons`: an icon glyph a badge
  layer holds whose soft pixels (partial, not an edge) weigh past `SEE_THROUGH` (0.03) of its ink is see-through
  ([support], [light], [guild_seal], [lightshadow], [elden_ring], [silksong_star], [6-8star] 0.11-0.38, [warpstar]
  0.049; opaque ones 0.02 at most); `gate_fx` refuses a new one. `apply_jobs` edits `edit_copy(old)`: the fork pairs
  smart filters by id() of the record, which a plain deepcopy kept from the source, so the copy rendered Lorian,
  Dragon_Mount, Defcon_Panda and 71 more without them and the PNG took that inside the regions (stat bakes silently);
  `gate_copy` renders the copy against the card (since the review fixes below: `gate_pairs` first, the render only on
  a card with smart filters). `gate_untouched` leaves out every edit's region, the
  stats' too (a stat with the name or ability always refused `region`), and runs on stat-only bakes. `covered_ink`:
  new text under a visible Normal layer above it (Papyrus' spaghetti) or over a stat bar row the old text did not
  reach (Paint Warrior's blank box) refuses `overflow`. Tests: `test_a_see_through_icon_under_a_badge_layer_refuses`,
  `test_the_edit_copy_keeps_smart_filters`, `test_a_stat_and_the_ability_bake_together`, `test_a_bake_runs_gate_untouched`,
  `test_new_text_under_a_layer_above_it_or_over_a_bar_refuses`; `test_a_bake_opens_each_psd_once` asserts the bake
  landed; the plain-card test moved to Koro-Sensei (on Papyrus the token now runs under the spaghetti).
  **Owner rulings 2026-10-24 (same branch; origin/main c249eba merged in: psd-tools `cards-20261024`, round 12):**
  phase 3's sample and cases re-measured on the new fork first (`~/Documents/Tools/wt/scratch/bug2-p4/`, venv
  `~/.venvs/ac-py314-r12`): 50 cards x 4 edits the same as on `cards-20261023d` result for result (105 baked, 50 n/a,
  31 overflow, 14 unsupported-card), the 43 named cases as phase 3 reported, each bake's PNG 0 px past 24 against its
  PSD's render. (1) A token typed in the title is set in the icon font at the title's size and colour under the
  ability's gates (section 5 step 6); the parser reads a title's icons as its name, not a header's types or stars (813
  of 813 cards parse the same). Census, 66 titles of every effect mix x 3 edits: 26 cards bake (51 bakes: strokes,
  shadows, glows, bevels, off gradients), each typed token in the icon font at its title's size; 57 refuse under a
  tinted title, 38 overflow, 3 `region` (Skeleton Mate, any title edit), 5 split titles. (2) A typed `:sticker` or
  `:tint` takes its kind's colour on the card (`variant_fill`). Census, 12 cards x 5 edits: 54 bake, Palace of
  Shadow's [dark:sticker] white (it was grey), its [light:sticker] and :tints grey, its [fire:sticker] refused (a
  tie); 5 overflow. (3) A new badge or token takes the commonest size (`icon_style`; one edited in place keeps its
  own). Census, 18 two-size cards x 3 edits (a (2U) after and before the odd badge, a [fire] after it): 49 bake, every
  typed badge at the card's commonest badge size and token at its token size (16 of 16 typed right after the odd
  badge took the odd size before); Pizza refuses (its clipped Multiply `Layer 12`), Clock overflows. Tests:
  `test_icon_style_takes_the_commonest_size_first`, `test_a_typed_badge_takes_the_cards_usual_badge_size`,
  `test_variant_fill_takes_its_kinds_colour`, `test_a_typed_sticker_takes_the_colour_of_its_kind`,
  `test_a_token_typed_in_the_title_is_set_in_the_icon_font`, `test_gate_fx_checks_a_title_edits_icons`; each fails with
  its fix taken out (`mut-*/`).
  **Review fixes 2026-10-24 (same branch, r12 venv; `scratch/bug2-p4/fix5/`):** `_tinted` counts a group's Color,
  Gradient or Pattern Overlay, Satin, Inner Glow or Inner Shadow (a [fire] in The Radiance's or Captain Spaceboy's title
  baked as a one-colour silhouette; Butterfly's own Inner Glow now refuses too). A typed `:sticker`/`:tint` reads the
  other text layer's colours off the card before the bake (`set_text_layer(original=)`: Salt's, Glitz Pit's and Basic
  Shield's title stickers took the Ability's typed one's colour; now the title's). `keep_breaks` pairs only breaks
  between the old text's characters ("Old Foxy\r" + " [fire]" set the token on a new line; 23 titles end in a break).
  Words typed after a title's token take the title's run style (`title_styles`; "Bench [fire] II" refused). A title's
  "[fire:]" stays letters in its own font. `covered_ink`: layers already over the old text are the design, and
  together may cover as much of the new as of the old (Leshy "Leshys", DooPliss "Dooplisss" bake; "Lesh" refuses, 512
  px more under the art; Papyrus still refuses). `apply_jobs` renders the card whole only once an edit is made, in one
  fork `RenderCache`; the copy's smart-object pairings are checked without a render (`gate_pairs`), and `gate_copy`
  renders a fresh copy, uncached, only on a card with smart filters (80 of 813), after the edits. Measured under load average
  ~30: Pure Vessel hp 9.3 s -> 5.9 s (3 full renders -> 2), its refused long ability 5.7 s -> 0.6 s (no render), Mini
  Doomer and Lorian (smart filters) about the same. `PSD_RCACHE_VERIFY=1` over the bake tests: no mismatch but the
  deliberately unpaired copy (the cache keys leave pairings out, which is why `gate_pairs` runs first). Reviewer's
  cases re-run (`fix5/rsb/logs`): every title sticker or tint typed beside an Ability one takes the title's colour;
  nothing else changed. Tests: `test_a_token_under_a_groups_overlay_refuses`,
  `test_a_title_sticker_takes_the_titles_colour_beside_a_typed_ability_one`,
  `test_a_title_edit_keeps_its_blank_last_paragraph_off_the_new_words`,
  `test_words_typed_after_a_title_token_take_the_titles_style`, `test_apply_jobs_runs_the_title_icon_checks`,
  `test_apply_jobs_checks_its_copy`, `test_a_bake_renders_the_card_only_once_an_edit_is_made`, Leshy and DooPliss in
  `test_new_text_under_a_layer_above_it_or_over_a_bar_refuses`; each fails with its fix taken out (`fix5/mut.py`).
  **Review fixes 2 (2026-10-24, origin/main 3ca616e merged; `scratch/bug2-p4/fix6/`):** a typed `:sticker` or `:no`
  (`LAYERED`: several overlapping glyphs in one cluster) is set at tracking 0: it took the card's tracking (Barry's
  tokens 280, The Hive's title 240), which the fork adds between the cluster's glyphs, sliding the outline or no-circle
  off the icon (32 baked silently in the reviewer's set C; no :sticker or :no on main is tracked); a plain token keeps
  the card's. Every typed badge or token has Ligatures on (5 titles have them off; the engine applies the font's rlig
  anyway, Photoshop may not). `keep_breaks` pairs a title's runs of blanks as runs (the website shows each as one
  space: "Pepperoni\r\rPizza" lost a break and its second line moved up; 0 of 728 titles now change an unedited
  character, with or without a typed " [fire]"). `type_row` leaves out the title layers (`title_layers`): a baked
  "Po [fire]" refused every later types edit. `_tinted` returns its cause and both refusals name it ("the title's
  look (its Inner Glow)" on Butterfly, not "overlay, blending, opacity or clipped layers"). Tests:
  `test_typed_layered_tokens_are_untracked_and_ligated`, `test_a_title_edit_keeps_its_runs_of_blanks`,
  `test_a_title_token_is_no_header_type_row`, `test_a_title_token_under_its_own_inner_glow_refuses`; each fails with
  its fix taken out (`fix6/mut.sh`).
  **Review fixes 3 (2026-10-24, r12 venv; `scratch/bug2-p5/`):** the title check's `badge_layers` takes the
  vocabulary (Fool Eater's and The Krusty Krab's root type icons; also `plan_stars` and `plan_types`, which change no
  plan on main's 172 star cards), and `inks_over` refuses `overflow` where the title's new ink covers a visible layer
  below it that the old title's ink did not touch (Spice's star in 'images', Nosk's 'Hk star', Flame Trap's, Bag of
  Evil's, Energy Ritual's, Bubble's banner). A title that would take more lines (Basic Shield's box, the Cursed Idol), or
  a line below the old title, refuses `overflow`. `typed_icon_fit` checks each typed badge or token on the engine's
  render: past `ICON_OVERLAP` (16) px over the layer's other glyphs it refuses `overflow` (Machine Gun, Bag of Good,
  Niiue, Kel, Mettaton EX); one set at the text's own size (no badge or token on the card, or a title) whose smallest
  glyph is outside `ICON_HEIGHT` (0.6 to 2.2) of the tallest letter on its line refuses `unsupported-card` (Holy Blade
  0.20, the pixel font of Machine Gun, Quote, Springtrap... 2.28; the rest 0.86 to 2.15). Census of every card
  (`census.py`, in memory): the typo title fix bakes on the same 715 as before; new refusals: 10 abilities (" Gain
  [fire]."), 14 titles with " [fire]", 11 with "[fire] ", 17 with " [dark:sticker]", each looked at. Refusal wording: the tint refusal says "typed icon" or
  "typed badge", for a title "the title's look (...) would recolour the typed icon"; the see-through refusal names an
  overlay or blend colour only on a tinted layer. Renders: the baseline's "preview" render is cut from the bake's
  whole render of the card (`crop_pair`; byte-identical PSD and PNG on Pure Vessel and Lorian), the merge's region render
  and `gate_idempotence`'s run in the bake's RenderCache, the second resuming the first (its key holds the pins, so the
  merge's stays a cold pass); Pure Vessel's hp bake 5.6 to 4.8 s. A pinned whole render (`gate_untouched`) misses the
  cache (the pins are in its key) and cost more than it saved. Tests: `test_a_title_token_off_the_header_icons_and_stars`,
  `test_star_and_type_plans_see_the_root_type_icons`, `test_a_typed_icon_far_from_its_letters_refuses`,
  `test_a_title_token_that_wraps_refuses`, `test_icon_style_breaks_a_size_tie_by_the_body_size`,
  `test_gate_pairs_checks_each_pairings_filter_and_place`, `test_covered_ink_weighs_layers_over_the_old_ink_anywhere`,
  `test_a_bake_cuts_its_region_renders_from_the_whole_ones`, the untinted case in
  `test_a_see_through_icon_under_a_badge_layer_refuses`; each fails with its change taken out (`bug2-p5/mut.py`).
  **Glow tokens (2026-10-24, pin `cards-20261024b`, venv `~/.venvs/ac-py314-fork`; `scratch/bug2-p6/`):** the fork draws
  a `:glow`/`:darkglow` halo as Photoshop does, so typed ones bake under the same gates (a fork without the halos,
  `glow_halos`, a stale install: an ImportError and card_bake's `Transient`, no verdict; it was a lasting `font` one). A stroked Text group strokes the halo into a black disc,
  Photoshop's look: Photoshop's own Nebulous Precipice with `[darkmatter:glow]` (`~/std-work-1022/glow/run`) against
  the fork's re-render, 0.47% of the token window past 24 levels (max 75); moved by an edit, 0.52% of the disc. Two
  gates changed for them: off the canvas counts ink past `INK_ALPHA` only (`_ink_off_canvas`; Red Knife's halo tail
  at alpha 0.03, Salt's title at 0.16 refused), and for glow edits `inks_over` and the stat-bar check count the
  changed ink of the layer's outermost group with effects (`_effect_ink`): Salt's title disc covers 'Layer 19'
  (MISS), which the title's own ink missed. Under a badge FX layer a glow refuses `see_through_icons`, under a tint
  as any typed icon. Census of every card (in memory): " Gain [fire:glow]." sets on 516 of 791 Abilities (" Gain
  [fire]." 536: 20 more run off the canvas, the wider glyph wrapping onto another line), " [ice:darkglow]" on 120 of
  812 titles (" [fire]" 200: 31 more reach the stars or type icons, 49 more cover a layer below, the disc); the other
  edits unchanged on the new fork (two overlap counts moved by 1-8 px). Dry bakes of 58 typed glows (plain, own
  stroke, stroked group, badge, tinted, titles): 27 bake, crops looked at. Tests:
  `test_glow_tokens_bake_as_photoshop_draws_them`, `test_glow_tokens_refuse_on_an_older_fork`,
  `test_a_faint_halo_past_the_canvas_is_no_overflow`, `test_a_glow_disc_over_a_layer_below_the_title_refuses`; each
  fails on the code before.
  **Review fixes 4 (2026-10-24, `scratch/bug2-p6/fix-2/`):** deleting a line whose next starts alike kept the deleted
  line's runs on the kept characters (changed_range's prefix is greedy: Dragon Mount's "+25% SPD" printed "+2" red,
  Chef Saltbaker's kept "(2 points)" blue, a kept badge split across two sizes and refused as letters), on main too:
  a deletion now slides to whole lines, else words or badges (`deletion_start`, and each deletion part in
  `edit_parts`), `splice_text` holds the engine's prefix to the part, and a deletion widens into a word only when it
  cut one (Gasoline's kept "If"). In-memory census of every one-paragraph deletion (`delfill.py`): kept characters off
  their own fill or size 224 -> 24, the 24 all status words given the chart's swatch (`delpar.py`: the `lettered_icons`
  and "would be set in the icon font" refusals of such deletions 36 -> 0). An
  Ability's new line printed past the old text onto the design below (Deep Focus' flourish, Holy Card's torn border,
  Autopsy Report off its box) is told in the notes, not refused (`ink_beyond`; 18 of 503 cards given a line).
  A text layer with tab stops refuses `unsupported-card` (the engine lays none: Michael's Blessing's arrows collapsed).
  A typed glow on a fork without the halos is an ImportError and card_bake's `Transient`, so no lasting `font` verdict.
  Tests: `test_a_deletion_cuts_whole_lines_or_words`, `test_deleting_a_line_keeps_the_next_lines_own_style`,
  `test_a_new_line_on_the_design_below_is_told`, `test_tab_stops_refuse`, `test_glow_edit_on_an_older_fork_is_transient`.
  **Review fixes 5 (2026-10-24, `scratch/bug2-p7/`):** a stroked title's glow disc ran under a star or type icon
  above it, every gate passing (Confusion, Krabby Patty, Unbreakable Strength): the halo's changed ink now counts on
  the stars and type icons and under the layers above (`covered_ink`); in the effect scope's group, where the composite
  hides the halo under a layer, the halo's edge beside it and a badge's box count (`_scoped`, `_meets`); a clipped
  icon counts only where its base inks (`_halo_meets`: Elemental World's ice texture past its icon). Census of the
  review's 2,415 glow edits (`census_halo.py` on the fix): 3 bakes now refuse, the 3 found; no other verdict changed. Where the
  splice falls back (about 2% of ability typos: Wendy, WX-78...), the engine redraws every line: the old text so
  redrawn is measured against the PNG (`whole_redraw`; Wendy 15.5% bad), a miss told in the notes ("ability redrawn
  whole, its lines laid out again: ..."), past the floor refused. Tests: `test_a_glow_disc_under_a_star_refuses`,
  `test_a_glow_disc_beside_a_layer_composited_with_it_counts`, `test_a_text_redrawn_whole_is_measured`; Gasoline,
  Salt and Nightmare King Grimm in `test_deleting_a_line_keeps_the_next_lines_own_style` (it fails without
  `deletion_start` or the word-bound rule).
  **Phase 6, content flags (owner ruling 2026-10-24 ~18:30, `scratch/bug2-p8/`):** "In general I do not want to
  refuse bakes": the editor previewed the edit, so a content problem bakes best-effort and is told in `Edit.notes` as
  `<code>: <field>: <why>` (bake report, commit line, baked verdict's reason, preview notes): overflow (off the canvas,
  onto the stars or icons, more title lines or lower, under a layer above or over a bar, past the design below for a
  title too: `ink_beyond`, past a texture clipped to it: `ink_past_clip`), typed badges and tokens (tints, letters,
  image-icon cards, fit and height, see-through, unshaped, bare: `gate_fx` returns notes), font stand-ins
  (`font_map`: FontMap policy "substitute"; a whole redraw in a stand-in still hits the 25% floor), saved pixels in
  another font, new validator problems. An FX layer whose style can't follow (follow_fx's refusals, a phrase the ability
  lacks, words now in a badge) is deleted, told, its words plain (styled status words get the chart's layer from
  `new_fx`, which skips, told, a word it can't name); an overlay word layer the chart has no style for is hidden, told;
  a typed copy in a phrase of several colours takes the first; a badge typed in a title prints as plain words. A stale
  Txt2 story is synced (`txt2_repair.sync_stories`) and the stories allowed in `gate_stories`, told. Typed badges and
  tokens are unstretched (`unstretch`: HorizontalScale so the aspect factor is 1; Monster Meat's [fire] 2.13 -> 0.81,
  the designers' 0.81). Still refused: waits, impossible cases, the bake's integrity gates, and what it can't edit yet
  (split/several titles, non-template rows, tab stops, stars/types layouts, FX layers that are no pixel layers, two FX
  layers one name). Census (`census.py`, 7 edits x 812 cards, `set_text_layer` in memory): passes 2,401 -> 5,069 of
  5,642; 573 still refused there (all `unsupported-card` layouts), none newly refused, no errors. The 2,668 former
  refusals that pass it, run through `apply_jobs` (every gate, dry, on copies; review 2026-10-24, after the pixel digest
  and nested-shadow fixes): 2,660 bake flagged, 8 refused by integrity gates: `parity` whole title redraws past the 25%
  floor (Dull Knife x3, Dark Tentacle, Snow Angel, Shiny Knife) and `reparse` (Nebulous Precipice x2: the image-icon
  card reads back an extra [darkmatter]); 10 crops spot-checked. 46 refusal tests turned into flag tests, 2 deleted, 1 new
  (`test_typed_icons_are_unstretched`). Full suite: 1787 passed, 13 skipped, 1 known timing flake
  (`test_a_stat_edit_bakes_the_bakes_pixels`, passes alone). Review 2026-10-24: `gate_bytes` hashes every channel's pixels (`layer_records`
  "channels": id, length, digest; a same-length corruption of an untouched layer refused), `fx_pad` adds up nested
  owners' reach (Skeleton Mate's group shadow over the Ability's), a flagged bake keeps its verdict, refused verdicts
  carry `rules`: 1792 passed, 13 skipped.

## 8. Runner

**GitHub Actions (primary).** TTSCardMaker `.github/workflows/card-edits.yml`, as it runs since 2026-10-22 (the file is the
authority):
```yaml
name: Card edits
on:
  schedule: [{cron: "*/30 * * * *"}]   # live, behind the gate below
  workflow_dispatch: {inputs: {card: {type: string}, dry_run: {type: boolean, default: true}}}
permissions: {contents: write, actions: write}
concurrency: {group: "card-edits on a schedule, card-edits-<run id> on a dispatch", cancel-in-progress: false}
jobs:
  bake:
    if: github.actor != 'AutoCroissant'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v7      # sparse, non-cone: /site/Card Edits.json, /TTS/ (and every tree, for the fonts' sha)
      - uses: actions/checkout@v7      # bobberlington/AutoCroissant -> bot/: commands/{card_bake,psd_edit,psd_regions,psd_parser,card_models,card_edits,github_api,git_commit,repo_history}.py global_config.py (never psd_analyzer: discord, pandas)
      - uses: actions/setup-python@v7  # 3.12 (3.14, like every Action, since 2026-10-24)
      - run: pip install requests "psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@cards-20261021"   # timeout 10 min
      - id: gate                       # schedule only: python -c 'from commands.card_bake import pending ...' -> go=true|false (timeout 10 min)
      - uses: actions/cache@v6         # MDW/Fonts by its tree sha (if go)
      - run: python -m commands.card_bake --repo "$GITHUB_REPOSITORY" --head "$GITHUB_SHA" [--pending] [--dry-run] [--card "$CARD"] | tee "$RUNNER_TEMP/report.md" >> "$GITHUB_STEP_SUMMARY"
        working-directory: bot         # if go; --pending on a schedule; dry run only for a dispatch with dry_run ticked; timeout 50 min
        env: {GITHUB_TOKEN: "${{ secrets.GITHUB_TOKEN }}", PSD_TOOLS_FONT_DIRS: "${{ github.workspace }}/MDW/Fonts"}
      - uses: actions/upload-artifact@v4   # report + before/after/diff crops per card
      - run: gh workflow run site.yml && gh workflow run tts-save.yml   # !cancelled(), only when the run committed ("committed `" in the report)
      - name: Tell Discord             # (failure() || cancelled()) on a schedule, first of a streak: DISCORD_CHANGELOG_WEBHOOK with the run's link
```
**The gate** (`card_bake.pending`, 2026-10-22): a scheduled run bakes only when some PSD-kind edit has no final verdict, a
refused `_psd` for exactly its values on the PSD blob the tree holds now (build_site.apply_edits shows a refusal by the same
rule; a baked field leaves the file). So a new, changed or redundant edit, a try-again refusal (`recent`, `head-moved`,
`edit-changed`, a runner error) and a refused card whose PSD changed all open it, and a complete live run closes it. Chosen
over "Card Edits.json changed since the last successful run" (tts-save.yml's kind): it needs no run history or state, it
retries try-again refusals and refused cards whose art was fixed (a diff of the edits file sees neither), and a wrong "go"
costs a run that commits nothing. A scheduled run then bakes only the gate's cards (`--pending`, `run_report(pending_only=)`:
a card whose refused verdict holds is not fetched or classified again, so hundreds of refused cards cost no time from
`BUDGET`); a dispatch re-checks every card. Measured on the live repo: 29 cards pending before the first live run (5 baked, 20
redundant, 4 refused), none after it. A run that ends at the gate takes the checkouts and the 25 s pip install (about a
minute; `card_bake` imports psd-tools at the top, so the gate can't run before it). A `font` refusal also carries
`fonts`, a key of MDW/Fonts' top-level files (`Snapshot.fonts_key`), and is final only while they hold: adding the face
retries it. A `duplicate` or no-PNG refusal likewise carries `pngs`, a key of the PNG paths it rests on
(`Snapshot.pngs_key`: the official PNG and the numbered ones beside the PSD, or the numbered PNG a PSD-less card is), and is
final only while they hold: removing `NME_Stock2.png` retries it, a re-export of the same paths doesn't (a verdict written
before 2026-10-22 has no key and is retried once). A pixel refusal under the floor (`parity: baseline bad 0.11% ...` or
`idempotence: <field> re-render bad ...`, written under the old bar before 2026-10-23: Masked Dedede, PoD11ZOMG and Zero
Override on that day) is never final: the next scheduled run bakes it or refuses it again past the floor or as a render
that shows no change, and those stay final. Until that run the website still shows it refused: TTSCardMaker
`site/build/build_site.py` (about line 855) shows any `_psd` refusal while its `for` values and `psd` prefix hold and
knows nothing of the floor (review 2026-10-23). `pending()` takes these cards on the first 30-minute tick after this lands
(3 cards, under `MAX_BAKES`), and the bake commit rebuilds the site, so the window is one tick; a dispatch closes it at
once. Teaching build_site the floor is a TTSCardMaker change, not made. A `Card
Edits.json` that isn't JSON fails the gate (Discord is told) rather than ending it silently. Not covered: another refused
card whose PSD and edit are unchanged is not retried after a pipeline, fork or chart fix; a dispatch (`dry_run` unticked)
does that.

**Cap and alert** (2026-10-22). A run bakes at most `MAX_BAKES` = 40 cards and starts none after `BUDGET` = 40 min; the
rest are refused `capped` (try-again: no verdict, pending, the next tick takes them). A title bake took 29-31 s in run
39761732714, so without it about 110 edits ran past the job's 60 minutes, which cancelled the job before the one commit,
every 30 minutes; the cap also bounds what one unattended commit reprints. `main` exits 1 when the commit failed although
cards had shares, or a card has an `error` (transient ones too: no text engine, no chart, a missing module, a 5xx); the
upload and the site/TTS dispatch still run (`always()` / `!cancelled()`), and a scheduled run that failed posts its link to
`DISCORD_CHANGELOG_WEBHOOK`, once per streak: the step skips the message when the last finished scheduled run (cancelled
and skipped ones not counted) failed too, so a persistent failure (a transient error writes no verdict and keeps the gate
open: no chart, no text engine, a missing module) is one message, not 48 a day, and the next streak after a success is
told again. Steps time out before the job does (pip and the gate 10 min, the bake 50): a step timeout is a failure, and a
job cancelled at its 60 minutes counts too (`cancelled()`); concurrency only cancels pending runs, which start no step.
Nothing is committed on a timeout (the ref update comes last). Not guarded: a large batch of new website edits bakes
unattended at up to 40 cards a run (about 80 an hour); the cap bounds each commit and `git revert` undoes one, and a
"too many new pending: wait for a dispatch" threshold is a possible later guard.
**Own concurrency group**: GitHub keeps one pending run per group across workflows, so in tts-save's group the next tick
would replace a pending TTS save; both workflows redo their commit on a moved head, and a bake that commits runs
tts-save.yml afterwards. Dispatches get a group per run (2026-10-22), so a tick never replaces a pending `--card` or dry
run unseen; a live dispatch may overlap a scheduled bake, which `land` makes safe (a card both baked is left out of the
later commit as `head-moved` or `edit-changed`, the edits file is re-read at the head). **Unpinned bot**: the schedule checks out bobberlington/AutoCroissant's default branch, so an
AutoCroissant main commit is live within 30 minutes (the fork is pinned by tag, the bot code is not); land AutoCroissant
changes the workflow depends on before the TTSCardMaker change that calls them (in the other order every tick fails at the
gate with an ImportError, `card_bake.pending` and `--pending` being new on 2026-10-22).
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
once `tests/test_parser_goldens.py` passes under it; fonts on the jetson are done (B11, 2026-10-21: `card_bake._fonts_dir()`
reads `PSD_TOOLS_FONT_DIRS`, else `LOCAL_DIR_LOC/MDW/Fonts`, else `fonts_cache/` brought to the snapshot's MDW/Fonts by blob
sha, `font_check.refresh_cache`; a font that fails to download is a try-again, never a `font` verdict); the work dir
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
  "psd": "<blob8>", "reason": "<code>: <why>", "at": ...}`; a refused one carries `"rules": card_bake.RULES` too (one
  without, from before 2026-10-24's ruling, is not final: `--pending` tries it again), and a flagged bake's verdict
  stays alone (`{"_psd": ...}`) once no field is left (`rewrite_entry(keep=)`), for the card page. Both readers drop `_`-keys (`card_edits.py:112-113`, `build_site.py:387`),
  `editsFile` keeps them (`edit.js:40`): no new file. Rewritten every run; ignored when `for` or `psd` no longer match.
- `app.js:347`: "PSD pending: SPD, Ability" beside "Edited" (tip: the card image is being regenerated); "Image update refused:
  ability does not fit"; "Re-parse pending"; `pending` in `FLAGS` (:25); the Edited panel (:588) lists them. `edit.js`:
  PSD-group rows show the note; the unset rule (:308) already handles "back to the PSD".

## 10. Reporting and failure handling

Four sinks: the commit body; the job summary + artifact (report, before/after/diff crops, parse diff, record diff); `_psd` for
the editor; Discord on the jetson (summary, `.md`, commit link). Refusal codes are a closed list: `duplicate`, `unsupported-card`,
`stat>10`, `recent`, `blob`, `parity`, `styled-word`, `font`, `overflow`, `bytes`, `size`, `region`, `shape`, `reparse`,
`idempotence`, `edit-changed`, `head-moved`, `capped`, `error`. Since 2026-10-23 `parity` and `idempotence` on pixels mean
past the 25% floor (a broken render), or for `parity` a box that misses the bar and whose render shows no change; a miss of
the bar is no refusal but a note in the commit body and the report (section 5
step 3). Since 2026-10-21 `unsupported-card` also names a row whose several digits
carry enabled effects (which one the row shows is ambiguous) and the overlay word layers the chart has no style for (FX
layers in no chart style were refused too until 2026-10-23: rebuilt in their own style since; since BUG 2, 2026-10-23,
also an FX layer whose words would fall in a badge, a styled or own-style FX layer whose JSX name would move to other
words, two FX layers renumbered onto one name, a styled status word a digit touches beside a whole-word one, an FX
layer that is no pixel layer; since the evening, a badge or token typed under the Ability's overlay with no badge FX
layer, letters left in the icon font it draws no badge of; since the phase 2 review, a typed badge under any tint
(overlay at any opacity, non-Normal blend) with no badge FX layer, letters a deletion leaves in the icon font, a badge
layer whose badges all left while the icon font draws something, an edit inside an own-style or styled FX phrase, an FX
layer phrase the ability lacks), since 2026-10-23 a bar or a digit mask wholly off the canvas (`hp: the bar 'Hp Dark' lies outside the card ...`,
`hp: digit 5 of the bar 'Hp Dark' lies outside the card ...`), `font` also a typed character its font has no glyph
for (`ability: '☃' has no glyph in its font (drawn as a box)`), `reparse` also an icon-font glyph a tinted Ability
leaves without a badge FX layer (`(1cd): under the Ability's look (its Color Overlay) with no badge FX layer over it`; the look names the first overlay, inner glow/shadow, satin, blend, opacity or clipped layer it finds),
`stat>10` says whether the value is above 20, needs a second row, or needs a Bottom `10` marker, `parity` on a
styled digit means the render missed the bar (since 2026-10-23: past the floor), no longer that the digit is styled, and `styled-word` means only that this
psd-tools cannot cut glyph pixels or make or refill a pixel layer (the fork's text engine is missing), or that an FX layer's
words have no pixels: a styled word gets its FX layer.
Since 2026-10-24 (owner, phase 6 in section 7) the content codes no longer refuse what an edit's text, badges, styled
words or fonts look like: `overflow`, `font` (but no stand-in at all), `styled-word` (but no glyph cutting) and the
content `unsupported-card` reasons are notes on a baked card; `reparse` keeps only the read-back checks.
Per card all-or-nothing, but for stars and types (2026-10-23): their refusal is the field's, the card's other fields
bake and the commit line says `stars not baked: <why>`; nothing bakeable = exit 0 with a one-line
report; any exception inside a card's steps refuses that card as `error` with the traceback in the report; a crash outside
them leaves nothing behind (`$RUNNER_TEMP`), the next run recomputes. A commit that fails (`head-moved` after 3 tries, a
GitHub error) is a run note: no card gets a verdict (a refused card's line says its verdict was not written), the gate
stays open for the next run, and the run exits 1 (section 8, cap and alert). `capped` (2026-10-22) is the cap's
try-again code.

## 11. Races

- **/update_stats and PICKLE pushes.** The pipeline never writes pickles. Whatever order the bake commit and an editor's
  publish land, the database converges on the next `/update_stats`: the PSD blob changed -> re-parse -> the baked value
  stands because the field is gone; a field still present is re-applied over `parsed_values`. The old card goes to `old_stats`
  (expected); `diff_stats.py` before `/push` must list exactly the baked cards. The shared Lock (section 8) is the only
  coupling: `update_stats` queues no bake (the notice was removed 2026-10-21) and the bake queues no update.
- **New edits mid-bake.** Step 10's re-read at H' plus the non-force update; the editor's own 422 retry absorbs our commit.
  An editor rewrite that re-inserts a baked field now equals the PSD -> redundant -> dropped next run, no pixels (whether
  edit.js rebuilds from stale in-memory data after a 422: to confirm; a flicker of "Edited" at worst).
- **Two bakes.** Actions `concurrency: card-edits` for the schedule (one run going and one pending, a newer pending tick
  replaces the older, a running one is never cancelled; not tts-save's group since 2026-10-22, section 8); a dispatch has a
  group of its own and may overlap it (`land`'s expected blob shas and the re-read at the head keep that safe); jetson: the
  Lock. A PSD or PNG pushed in the last 30 min is skipped (`recent`). Site lag
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
| S5 TTSCardMaker | `card-edits.yml` dry-run, then live with `gh workflow run --card`, actor guard, tts-save retry; the push trigger goes live only after question 7 (else the first push run reprints the 7 titles) | one stat bake on Mini_Doomer lands PSD + PNG + one-line file diff; site.yml and tts-save.yml run; `diff_stats` lists that card. **Live 2026-10-22**: run 39761732714 reprinted the titles (one commit per card then); since then one commit per run, and a 30-minute schedule behind the `pending` gate replaced the push trigger (section 8) | S |
| S6 bot + fork | idea 14: FX regeneration (glyph boxes, pixel cut, chart effect block) **code done 2026-10-21**, no fork change (section 7: Gasoline IoU 0.966 / Cannon 0.937 with the JSX cuts, coverage 1.0, Mini_Doomer ability edits with FX pass every gate); idea 12 and idea 13 **code done 2026-10-21** (section 7: Mettaton_EX hp 5 -> 8 passes every gate; the Cannon writes and re-parses but its bevelled digits refuse `parity`) | Gasoline and Royal Eradicator FX layers within the validator's 0.95 coverage and the region bar; Mettaton_EX rows pass parity; goldens refreshed | L |
| S7 jetson | a manual fallback: fork in the bot venv, Lock, SSH commit, Discord report. **Code done 2026-10-21** (`commands/card_runner.py`, `/bake_cards`, `bake_lock`; the edits-sha notice landed and was removed the same day: the user chose ONE live runner, the Action, and no notice on the jetson); the venv install only if the fallback is ever needed | only if ever needed: one bake with Actions disabled; CPU/RAM peak recorded | M |
| S8 user | the 10 edits of section 12 | 10/10; results in `docs/Finished/`; Photopea retired to an archive branch | S |
| S9 bot + TTSCardMaker | star edits on folder-starred cards (creatures, items, field cards): move PSD + PNG to the new `N Stars` folder; aliases, `old_stats`, authors and the TTS save follow (question 4; ledger L14). **Planned**, after round 1 (row added 2026-10-21); a move is a PSD/PNG write, so it waits for the hold | proposed: a `/tmp` dry run moves one card: PSD + PNG at the new path, the old path gone, the parser reads the new stars, `diff_stats` lists only that card, the TTS save pins the new path | M |

## 14. Open questions for the user — answered 2026-10-20

1. **Yes**: the parser reads `Text/Name` into `display_name` (edits still override); the 22 name edits that equal the print
   become redundant and are dropped by the bake's redundancy rule; a baked rename leaves the file. **Done 2026-10-21**
   (`PARSER_VERSION` 2: 157 cards gain a `display_name`; 20 edits redundant now, 21 with Thunderfury; Sans and Papyrus have no
   readable `Name` layer; the next `/update_stats` re-parses every card once). 2026-10-23: their layers are named after their
   own text, which the parser now reads (Papyrus "The Great Papyrus", Sans "snas."; no version bump: re-parse the two with
   `/update_stats cards:Papyrus, Sans` and, during the shadow, a card-sync dispatch with `cards`: BUGS_AND_IMPROVEMENTS
   V6 has the order), so both edits are redundant.
2. **Yes**: saved PSDs regenerate the flattened preview (`update_preview=True`). 2026-10-21 audit: the regeneration is 9.4 s of a
   23 s bake (a full-canvas composite); the merged PNG the driver already holds is that flattened image, so a fork
   `save(preview=<image>)` would keep the decision at no render cost (fork change, not done: the worktrees were in use).
   2026-10-21: round 3's fork branch `perf/save-with-preview` (98804ede, head 6f91981e) implements `save(preview=)`; `card_bake` still
   calls `apply_jobs(..., update_preview=True)` and switches to passing the merged PNG once the new pin tag is installed.
   2026-10-22: done against tag `cards-20261021`. `apply_jobs` keeps Photoshop's preview and replaces the merge's pixels with the
   unconverted render (`stored_preview`, `render_pair`, `merge_preview`; 8-bit RGB with a preview); `save_psd` passes it as
   `save(preview=)`. Mini_Doomer hp 5 -> 8: save 11.2 s -> 0.13 s, bake 13.0 s -> 1.8 s; the preview changes on exactly the
   PNG's 18201 delta pixels, which read back (ICC applied) equal to the merged PNG. Stock psd-tools (no `preview`
   keyword) and other cards regenerate as before.
3. **Top row 10 + bottom row N-10** (Royal Eradicator's layout; the parser already sums it). Sonic is converted once to that
   layout (standardization work; refused until then). `EXCESSIVE_STAT_EXCLUSIONS` goes. **Done 2026-10-21**: the list is
   deleted; Sonic and Royal Eradicator Main Cannon carry `allow_high_stats` in the edits file (the only two cards above 10).
4. **Pipeline moves the files** for folder-starred cards (PSD + PNG to the new `N Stars` folder with aliases, old_stats,
   authors and the TTS save following) — after round 1 (step S9).
5. **Commit the Mac-only font files to `MDW/Fonts`** (the user's call; licence risk noted), so every runner renders every card.
6. **Restyle to the chart and hide the overlays**: an ability edit normalizes off-chart status words to the chart's colour
   and hides the hand-made overlay word layers it replaces (part of S6); the effect is the chart's only on an FX layer it
   makes (since q6, 2026-10-23, on work/fx-rebuild: an existing FX layer keeps its own style).
7. **Reprint six titles** (Clot White's typo, the five PoD trailing dots); **Thunderfury keeps its brackets**: its entry is
   aligned to the print (redundant after 1). The push trigger can go live once that entry is fixed (S5). **Aligned 2026-10-21**
   ("[Thunderfury: Blessed Blade of the Windseeker]", written through `card_bake.dump_edits`). **Reprinted 2026-10-22** (run
   39761732714: Clot White, M.O.A.B., D.D.T., B.F.B., B.A.D.; Z.O.M.G. refused `parity`, baseline mean 1.28 over the then 1.0 bar;
   within the 2.0 bar since 2026-10-22).

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

## 16. Preview (2026-10-23)

**Ask (owner, 2026-10-23 ~18:10):** "press preview image to dry run bake a card, work on another card, and then go back
to the first to see the preview". This section is the server side; the website's UI is a later task, against the
contract below (also in TTSCardMaker `site/NOTES.md` section 19 and the ttscardmaker-companion skill §2.7).

**What runs.** TTSCardMaker `.github/workflows/card-preview.yml` (workflow_dispatch only; inputs `card`, `edit`,
`request`; one concurrency group per request) has two jobs. The **preview** job (`contents: read`, 11 minutes) runs
AutoCroissant `commands/card_preview.py` (uncompiled; import closure `card_preview.SPARSE_FILES`, which its bot checkout
lists; card-edits.yml's pip install, fork pin and all) and hands result.json, preview.webp and before.webp over as an artifact (one
day; the full-size PNG ends with the runner). The **publish** job (`contents: write`, 3 minutes; it runs when the
preview failed or timed out too, not for a bad request) runs `commands/card_preview_branch.py` with the runner's own
python3: the standard library and git only (import closure `card_preview_branch.SPARSE_FILES`), no setup-python or pip.

    python -m commands.card_preview --card <key or PSD path> --edit <json file> --out <dir> [--clone <checkout>] [--request <id>]
    python -m commands.card_preview_branch --out <dir> --request <id> [--git <checkout>] [--remote origin] [--png]

The preview is `card_bake.bake_card` itself on one `Card` (the duplicate, no-PNG and blob gates, the redundancy check,
`psd_edit.classify`, `apply_jobs` with every gate, the stars/types-refused-beside-the-rest rule, the creatures-stats
gate), with the bake's fonts (`card_bake._fonts_dir`: `PSD_TOOLS_FONT_DIRS`, set to the checkout's `MDW/Fonts` from
card-edits.yml's font cache), status chart (by blob sha, `psd_edit.status_fills`, as `run_report` reads it) and icon
vocabulary (`card_bake.vocabulary`), under the same fork pin and Python. Not the bake's: the `recent` gate (it guards a
commit; a preview commits nothing), the cap and budget (one card), and the edits file (the edit is the input). The card
repo is only read: the tree by `git ls-tree` in the blobless checkout, the card's PSD and PNG, the chart, the manifest
and the Types/ PNGs it needs by blob sha through the API (verified), as card-edits.yml reads them. **Adjusted from the
ask:** no sparse checkout of the card's files (that would put a dispatcher's path into a sparse pattern; by sha the
blobs are verified and only what the card needs is fetched); no Card Edits.json (the input is the entry); two jobs, so
the step that parses the dispatched edit and the repo's PSDs with pip-installed code never holds a write token, and the
publisher needs no install (hence a second module: `card_preview` imports PIL, psd-tools and requests through
card_bake); the PNG is not published (1.5 of a preview's 1.6 MB, and the site shows the WebP). No change to card_bake.py
or psd_edit.py (work/fx-rebuild was editing both): the preview calls `bake_card`, `vocabulary`,
`local_snapshot`/`snapshot_at`, `psd_fields` and two private helpers, `_fonts_dir` and `_describe`, so the verdict text
is the bake's.

**Input.** `card`: the card's key as Card Edits.json keys it (the PSD's file name with spaces, the website's `c.key`; a
duplicate's own key, A Snack2, so it refuses `duplicate` as the bake would), or the PSD path of one of the bake's cards
(`Snapshot.psds`: not under `MDW/` or `Markers/`, and of two PSDs with one key the one the bake takes; any other path is
an input error). `edit`: the card's entry as it would be published (the committed entry with the staged ops applied:
every PSD-kind field it would hold, not only the staged one, or the preview misses an unbaked committed edit), Card
Edits.json shapes, at most 16 KB; unknown fields and invalid values are input errors, `_`-keys are ignored, `creatures`
(any non-empty list) gates the stats as in the bake. `request`: `[A-Za-z0-9-]{8,64}` (a `crypto.randomUUID()`), the
folder on the branch; the workflow checks it in bash before anything and a bad one fails the run with nothing published.
The inputs reach the shell only through `env` (never `${{ }}` in a script) and a file; card_preview checks all three
again.

**Hygiene.** The runner reads any log line starting `::` as a workflow command, so every value printed (the job summary,
which `tee` also logs, and the stderr reason) goes through `card_preview_branch.clean`: control characters and line
separators as escapes, one line each. result.json, the index row and the commit message keep the card input clean and
clipped to `MAX_CARD` (200). The repo is public, so a preview is too: the dispatch's inputs in the Actions tab, the
result on the branch; the site says so before an editor's first press.

**Output** (`--out`; the publish job then publishes it): `result.json` always, `preview.png` (the PNG the bake would
commit, losslessly recompressed by `png_export.recompress` where pyoxipng is installed; the workflow leaves
requirements-cards.txt's pyoxipng out (~4 s a preview for nothing) and does not publish the PNG, `--png` publishes it)
and `preview.webp` (720 px wide, WebP quality 75, method 4) when an image was rendered, with `before.webp`: the card's
official PNG made the same way, for the site's Diff (one codec on both sides, so unchanged pixels match; the site's
thumbs are AVIF since 2026-10-24).
result.json (`v` 1):

| Key | Meaning |
|---|---|
| `request`, `card`, `path`, `png` | the request (null when invalid), the card's name, its PSD and PNG repo paths ("" without a PSD) |
| `base` | `{repo, commit, psd_blob, png_blob}`: the commit and the full blob shas the preview was baked from (stale detection) |
| `edit` | the edit exactly as sent, without `_`-keys (nulls and a creatures list as sent): the site's staleness check compares it |
| `state` | `baked` (something would print; an image), `redundant` (the card already prints it all; no image), `refused` (the bake would refuse; an image only when a gate after the save refused: what the render got), `error` (below) |
| `transient` | with `error`: true = try again (a runner or GitHub failure, a blob that failed its hash); false with `reason` "input: ..." = the site sent something the preview does not take; false otherwise = the preview cannot run on this card or repo as it is (a broken manifest, a bot bug): a new press helps once that is fixed |
| `fields` | `{field: baked \| redundant \| refused \| error}` per PSD-kind field (a refused stars edit beside a baked hp) |
| `reason` | the bake's own words: `hp 5 -> 8`, `stat>10: hp 25: above 20`, `already on the card: hp`, `input: not a valid hp: '8'`, `blob: blob 1a2b3c4d hashes to ...` |
| `changes` | `[{field, old, new}]` of a bake |
| `notes` | pixel misses (`miss_notes`), FX layers, FX notes (work/fx-rebuild's `fx_notes` once merged), icon layouts, validator problems of the new card, stars/types not baked, fields already on the card |
| `parity`, `region` | `{field: {bad_pct, mean, max}}`; the edited box `[x0, y0, x1, y1]` on the full-size card |
| `image`, `files`, `size`, `webp_size`, `bytes`, `recompressed` | `baked` / `refused` / null; `{webp, before}` on the branch (`{png, webp, before}` in the out dir); pixel sizes; file sizes |
| `runner`, `run`, `took_s`, `at` | psd-tools version, text engine, story sync, fonts present; the Actions run URL; seconds; UTC time |

Exit 0 for a verdict (baked, redundant, refused), 2 for an input error, 1 for any other error (all write result.json).

**The previews branch.** An orphan branch `previews`: `previews/<request>/{result.json, preview.webp, before.webp}`, `index.json`
(`{"v": 1, "previews": {request: {at, card, path, state, psd_blob, png_blob}}}`) and a README. `card_preview_branch` (a
missing or foreign result.json is published as `error`, `transient`: the preview never finished) makes one commit per
run on the branch head, from the publish job's blobless depth-1 checkout and its credentials:
`git fetch --depth 1 --filter=blob:none` of the head into `refs/card-preview/previews` (index.json by a lazy fetch),
blobs and trees by `hash-object`/`mktree`, `commit-tree`, `push --no-thin` (card-sync.yml's lesson: a thin pack would
want blobs a blobless clone lacks); a refused push (the head moved) is retried from the new head, 5 tries. When the
branch would hold more than `KEEP` (200) previews or one older than `DAYS` (7 days, or of unknown age), the commit is a
fresh orphan with this preview and the newest `KEEP_AFTER` (150) under `DAYS_AFTER` (6 days), pushed with
`--force-with-lease` on the head it read (a run that landed meanwhile breaks the lease and the retry keeps its preview);
the margin makes a rewrite about daily, not every run. So a preview stays at least 6 days unless 150 newer ones came
after it. Size: about 60 KB a preview (Mini Doomer's WebP 59 KB, result.json 1 KB), so about 12 MB at 200, and history
stays one rewrite deep. Nothing triggers on it: every push trigger in TTSCardMaker is `branches: [main]` and none
listens for branch creation, workflow_run or checks (tests/test_card_preview.py checks every workflow), a GITHUB_TOKEN
push starts no workflow anyway, and the full-depth checkouts (site, guards, card sync, TTS save) are blobless, so the
branch adds only its commits and trees to them; the local clones fetch main only.

**Budget.** The preview reads its blobs through the REST API with GITHUB_TOKEN (1,000 requests an hour for the repo,
shared with card-edits and card-sync): the icon manifest, the PSD and the PNG (3 for a stat edit, a test), plus the
status chart for an ability and the added Types/ icons for a types edit. Fine at hobby volume; a burst of a hundred
previews an hour would compete with the schedules.

**Measured (2026-10-23, this Mac at load average ~55-66, the Desktop clone at b7798bb1 then 92b5ae6b, fork
cards-20261023c):** Mini Doomer hp 5 -> 8 baked in 6-14 s (the same pixels as card_bake's dry run, a test); The
Freezer's ability `+1 DEF` -> `+2 DEF` in 26 s, coloured as the original; hp 25 refused in 3 s; hp 5 redundant in 2 s;
stars 0 + hp 8: hp baked, stars refused, 19 s; a publish 1-2 s. On the runner, from card-edits.yml dispatch 40003504861:
queue 3 s, checkouts 4 s, pip 27 s, fonts 2 s; the publish job's start is not measured (about 10-20 s, estimated). So a
stat preview comes back in about 1-2 minutes after the dispatch, an ability or title in about 1.5-3.

**Tested** (`tests/test_card_preview.py`, 37 tests): the outcomes above on a local card repo (and that it is left as it
was), the edit echoed as sent, a blobless clone's three REST reads and a blob failing its hash (before the bake and in
it: try again), fourteen bad inputs (a PSD under MDW/ and a card with a line separator among them), values with newlines
kept off the runner's command lines, the branch's commits, re-runs, both rewrites, a moved head and a moved lease (a
`--force` instead of the lease fails the test), giving up after 5 tries, the card clean in the index and commit, the
publisher on Python without site-packages, the workflow's shape, no other workflow on the branch, the Action-shaped
copies, and both jobs' own steps (Validate, Preview, the hand-over, Publish with python3 = this Python with `-S`) run
here against a bare repo standing in for GitHub: a baked edit, a bad edit, a failed install (published as try again) and
a bad request (nothing published). Also run once by hand (2026-10-23, scratch
`~/Documents/Tools/wt/scratch/preview-1023/`, the single-job version): every `run:` step but pip, in order, against a
stand-in repo of real files (Mini Doomer, The Freezer, the status chart, Types/, the manifest, MDW/Fonts' top level):
The Freezer `[ice]: Blazing` baked with its `Ability FX: Blazing` layer (17 s), the same plus a line refused `overflow`,
Mini Doomer hp 8 + stars 3 baked (`stars: rows of two`, 22 s), a bad request stopped at Validate. Not tested here: the
dispatch itself (forbidden, and workflow_dispatch needs the file on main), the artifact hand-over and job outputs on
GitHub, GitHub's lazy fetch, the persisted credentials, and the REST blob path against GitHub itself (the test serves it
locally).

**First live dispatch (manual test plan).** In this order: (1) merge AutoCroissant work/card-preview to main (the
workflow checks out the bot's main); (2) merge TTSCardMaker work/card-preview to main (the file must be on the default
branch); (3) Actions > Card preview > Run workflow: card `Mini Doomer`, edit `{"hp": 8}`, request `manual-test-0001`:
both jobs green in about 1.5 minutes, the preview job's summary says Baked: hp 5 -> 8 and its "Set up job" log lists
`Contents: read`, the publish job's lists `Contents: write` and it runs no pip; a branch `previews` appears with
`previews/manual-test-0001/` (result.json and preview.webp only) and `index.json`, main does not move, no other workflow
starts; (4) open
`https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/previews/previews/manual-test-0001/preview.webp` in a browser
(it should show the card with 8 hp; confirm raw serves it as `image/webp`, else the site reads it by the API); (5) the
same with edit `{"hp": 25}` (refused, no image), `{"hp": "x"}` (Preview red, publish green, result
`input: not a valid hp`), and request `bad id` (Validate red, the publish job skipped, nothing published); (6) an
ability edit on a card with a status word (fonts from the cache); (7) two dispatches at once with different requests:
both published, two commits; (8) from a browser console with an editor's token: the POST and the poll of the contract,
to learn whether a classic `public_repo` token may dispatch (GitHub documents `repo`); (9) after a week, the first
rewrite: `git ls-remote` shows a new root, older previews gone.
