# Web editor campaign: card as data (2026-10-20)

**Status:** designed 2026-10-20, not scheduled; plan item 22 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-23"). Grows out of item 20 (inline editing) and waits for Task 3 steps 4-7, the standardization batch (item 21, on
hold) and the rendering parity rounds. Numbers are 2026-10-20: `~/Desktop/TTSCardMaker/site/dist/data/cards.json` (801 rows), `site/Card
Edits.json` (56 cards), `~/Documents/Tools/psd-research/RENDER_STATUS.md` (`cards_v1_icc`), the ledger
`docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md`. Anything not checked on disk is marked "to confirm". No PSD or PNG is written by
anything in this file while the hold stands.

## 1. Goal and non-goals

**Goal (the user, 2026-10-20).** "A very in-depth and extensive web editor for cards, to edit more things about cards through the browser
after we extend psd-tools more and standardize the cards more; eventually, ideally, PSD edits and database edits would become the same thing
and be fully in sync." The spine of this design is the end state that makes the two kinds of edits one: **card as data** (section 2).
**Non-goals.** Not a replacement for Photoshop for making art (Photoshop stays optional, never forbidden). No backend, no accounts, no server
we host: the site stays static GitHub Pages built by `site/build/build_site.py`, writes go through the GitHub API with the editor's token
(`site/assets/edit.js` `signIn`, `commitFiles`) and GitHub Actions (`.github/workflows/site.yml`, `tts-save.yml`, the pipeline's
`card-edits.yml`, `docs/EDIT_PIPELINE_DESIGN.md` §8), as the simulator and game-reports campaigns assume. No second source of truth at any
step. No rule or card content invented: the editor changes what the owner types, nothing else.

## 2. The end state: card as data

**The data file.** One JSON per card, the single source of truth. Proposed location: beside the PSD, `<Stem>.json` (`Items/Heal/2 Stars/
A_Snack.json`), found by the stem rule `png_for` already uses (build_site.py:586, case-insensitive) and keyed the way everything keys today
(aliases, authors, edits, `old_stats`, the TTS save all key on the stem: ledger L12); the AI campaign's `<card>.gen.json` sidecar is the
precedent (`AI_CARDS_CAMPAIGN.md` §2). Alternative `site/cards/<stem>.json` (folder does not exist; `site/decks/`, `profiles/`, `games/` are
proposed siblings): one checkout for the site build, but a star edit then changes a folder the file does not live in. Question 1.
**2026-10-21 (question 1 answered, the user):** the data files live under `site/cards/` (`site/decks/` and `site/profiles/` exist since
that day), so `site.yml`'s sparse `/site/` checkout reads them with no blob fetches.
Fields, from three sources: cards.json today (34 keys over 801 rows: `name type subtype series subseries stars types ability author aliases
creatures dup dups problems`, the stats, and build fields such as `img png psd parsed edited`; `hidden note problems_ok allow_high_stats`
are not keys: hidden cards are dropped at build_site.py:869 and `note` is not emitted), `EDIT_FIELDS` (the 19 of build_site.py:71 =
`edit.js FIELDS` = `CardEdits.TYPES`, `commands/card_edits.py:67`; `creatures` is database-only there, `CardEdits.KIND`, PSD-backed only on
Pix, Tainted Lazarus, Twin Emperors), and what only the PSD holds today: `template` (one of `MDW/Templates/`: `Creature_Template`, `2_`/`3_Creature`,
`Item`, `ActiveItem`, `Raid_Boss`, `Status_Effect`), `art` (asset file + placement in the clip layer `images/Clip Main image to this`,
0-1731 on the creature template), `ability_bg` (`Ability bg Exposure`, per-series backgrounds), `star_style` (which `Stars` smart object:
Warpstar, Hk star, Omori star, undertale...), `types` as header badges, inline `[fire]` tokens inside `ability` (stoogedings runs, L1),
the per-creature stat rows of those three cards, `exports` for the numbered duplicate PNGs (hidden title layers today, L12), `stat_txt`
(`Text/Stat Txt`), and for rulebook pages title/text/note (item 15). Not stored: anything derived from the chart or the manifest
(status-word colour and FX layers, badge glyphs, the Single-line composer: L7, L10). Provenance moves to git: `_by`/`_at` become the file's
log; `_psd` becomes a build stamp (below). Author and the constructed ban stay in `site/Card Authors.txt` / `Constructed Ban List.txt`
until question 3. The three field tables collapse to one schema file both repos read (`site/build/card_schema.json`, proposed).
**The escape hatch.** `assets: [{file, anchor, place}]`: an opaque per-card group of pixel layers (an overlay the chart cannot express, a
painted bar, extra art) placed at named template anchors. Tier B cards live here (section 4).
**What the PSD and PNG become.** Build outputs, regenerated from the template by the fork's renderer: text through `set_text` /
`set_span_style` / `rasterize` (fork `docs/text.rst`, branch `feat/text-engine`), stars and types by smart-object visibility, one digit per
stat row, art into the clip layer, assets pasted (the AI campaign's Phase 2 template fill is this same path: generation is card as data from
day one, D5). Both stay committed: Photoshop users, the TTS save and the website read them. The pipeline is their only writer.
**What the parser becomes.** A round-trip verifier: `parse(render(data)) == data` on every PSD-kind field, `CardValidator` problems a build
refusal instead of a database note; `ABILITY_EXCLUSIONS` and `EXCESSIVE_STAT_EXCLUSIONS` go with the cards that needed them. It stays the
reader for tier C cards and for history.
**How `Card Edits.json` dissolves.** Today: 56 cards, `name` 29, `series` 28, `subseries` 28, no stat or ability edit. The pipeline (Task 3
step 4) first drives the PSD-kind fields to zero (bake, then remove). Then the database-only fields move into the data files (`series`,
`subseries` first: 28 cards), `_by`/`_at` are dropped only after plan item 9 (contribution data; items 10-11 build on it) reads recent
edits from the history walk (`commands/repo_history.py`) instead of `Card Edits.json`, `_psd` becomes an entry in a pipeline-written manifest
`site/build/renders.json` (proposed: `stem -> {data: <blob8>, psd, png, fork, at, refused?}`; one writer, so the data file never carries
its own build state), and the file is deleted. `pending` = data blob != `renders[stem].data`, the same computation as today's
`pending`/`refused` in `build_site.apply_edits` (:411-449) with a different input.
**The bot's database.** `stats.pkl` is a `CardInfo` per card (`commands/card_models.py:59-84`: name, card_type, path, timestamp, ability,
stars, subtype, series, types, stats, problems, author, display_name, note, hidden, problems_ok, allow_high_stats, parsed_values, creatures,
duplicate_of, alt_names, blob_sha, subseries): every field has a data-file home. Two ways to fill it. (a) Derived: `/update_stats` reads
`<Stem>.json` by blob sha the way it reads `Card Edits.json` now (`repo_traverser.py:258-260` through the `_repo_text` cache, :120-128),
parses PSDs only for tier C; the pickle is a cache of the data files; `parsed_values` retires; no PSD download for tier A cards (813 PSDs,
16.4 GB). API cost: the first sweep is one blob request per data file (813), later sweeps only the changed blobs. The site build cannot read
files beside the PSDs for free: `site.yml`'s sparse checkout is `/site/`, `/Types/` and the manifest, no card folders, so it reads them from
the pickle or fetches 813 blobs (question 1 weighs this). (b) Still parsed, data file as the override: today's model, two sources forever.
Trade-off: (a) is one truth and cheap but can answer from data whose PNG is stale or refused;
that is exactly what the stamp and the round-trip gate make visible ("pending", "refused" on the site and in `/query`). Recommended: (a) per
card once it is tier A; no new pickle (autocroissant-failure-archaeology Entry 7); `diff_stats.py` before `/push` as today.

## 3. Why it must wait

1. **The edit pipeline** (Task 3 steps 4-7; `commands/card_bake.py`, `psd_edit.py`, `psd_regions.py`, uncommitted; design §13 S0-S8): it is
   the one writer, the gates (`blob parity region bytes reparse idempotence edit-changed`) and the runner this campaign reuses. Done when the
   10 edits of design §12 land.
2. **The standardization batch** (ledger §3-4, on hold): tier A needs template structure. Of 651 bar-eligible cards today: 371 template stat
   rows (L2), 605 single `Text/Ability` (L5), 552 visible `Name` (L6), 0 cards on the stoogedings font and 0 `Ability FX:` layers (L1, L7),
   6 faces missing everywhere plus 2 near-misses (L13; the 10 platform-only faces are done, 136 of 144 names resolve from the repo; 2026-10-21:
   only BabaUI was missing; added 2026-10-23, TTSCardMaker 92b5ae6b, so none is missing now). Plus
   the guard `check_cards.py` (ledger §5), which is how a card gets its tier automatically.
3. **Rendering parity** ("extend psd-tools more", concretely). `cards_v1_icc`: median 0.186% bad pixels, 115 cards > 2%, 243 > 0.5%;
   target "no card > 2%" not met. Remaining classes (RENDER_STATUS "What is still missing" and the critic pass): **A** effects on empty-bbox
   adjustment layers, the styled stat digits (87-101 cards, 41% of all bad pixels); **B** Gaussian Blur smart filters whose mask is not in
   the file (62 cards; not reconstructible: re-save in Photoshop or exclude); **C** full-canvas Perc gradient fills (19 cards; `gs99` vs
   `gradientsInterpolationMethod`, t^1.72 opacity ramp); **D** several instances of one effect kind painted in the wrong order (9 cards / 34
   layers); text edge blending gamma ~1.55 (all 813 cards, 41 of 48 rulebook pages > 0.5% for this alone); overlays on pass-through groups
   (37 cards) and adjustments inside styled groups over semi-transparent content (8 cards); big shadow and glow shapes; the LINE half-pixel
   offset under `force`. Fork branches exist for most (`fix/adjustment-layer-effects`, `fix/effect-instance-order`,
   `fix/gradient-fill-interpolation`, `fix/group-overlay-box`, `fix/shape-raster-offset`, `fix/text-blend-gamma`; merge state to confirm).
   **2026-10-21.** Class B is fixable in the fork, not Photoshop-only: the document's `FEid`/`FMsk` blocks hold each smart filter's
   pre-filter source and painted mask, and blending them fixes the 13 cards with active painted masks (branch `fix/smart-filter-mask`,
   planned; `~/Documents/Tools/psd-research/RENDER_ROUND4_PLAN.md` §2.1). Merge state: `cards` (tag cards-20261020, a6363a04) holds
   rounds 1 and 2 and the text engine; the user's round-2 review fixes sit on the six branches and `review/round2-consolidated`
   (31e1a294), not yet in `cards`; round 3 is integrating; round 4 (class B, pixel-bar rounding, the Cannon's bevel) is planned.
4. **Fork API for template fill**, beyond editing: add a face to a layer's `FontSet` (stoogedings runs, `text_edit.py:109-150`), move and
   rename layers, copy `lfx2` blocks, replace a smart object's content (art), `save(update_preview=True)`; the text engine's limits (warp
   unwarped, platform anti-aliasing as Strong, colour emoji as silhouettes: `text.rst` "What is rendered") must not touch any template layer.
5. **Items 12-15**: stats above 10 as two rows, FX layers from the text, text-first rulebook; the digit effects (12) are open: copy the
   block to all ten digits or let the pipeline move it on toggle is ledger question 5 (design §7 proposes the move). 2026-10-21: answered,
   the pipeline moves it (in code, ledger L4).

## 4. Migration in tiers, with gates

Extractor (new, bot side): parser values + the readers the parser lacks (`Text/Name`, star smart-object visibility, header badges, template
variant from the layer tree, art from the clip layer) -> data file; then regenerate PSD + PNG from the template and compare.
- **Tier A: template-conformant.** Gate A1 *regenerate-and-compare*: the regenerated PNG against the official PNG, per region (`score_cards.py`
  regions, |diff| > 24): 0 bad pixels and mae <= 1.0 in every region admits the source change. Byte-identical regions are counted (M5), not
  required: text edges blend at gamma ~1.55 (RENDER_STATUS critic 4), so no text region is byte-identical until that fix lands; the art
  region is identical by construction when the asset is the official render's art pixels (question 2). Gate A2 *parser round trip*:
  `parse(new PSD).to_dict()` == the data file on every PSD-kind field, validator problems none, `alt_names` unchanged. Gate A3 *Photoshop*:
  opens with no missing-font or "update text layers" prompt, runs intact; the user checks 10 cards per batch. Gate A4 *consumers*:
  `diff_stats.py` lists exactly the batch, the TTS save pins the new blobs, the version viewer shows the regeneration as a version whose
  image diff is empty.
- **Tier B: custom layers.** Overlays the chart cannot express, drawn bars kept as a variant (ledger question 3), extra art: data file +
  `assets`. Same gates; the assets are pasted pixels, so A1 holds by construction on them.
- **Tier C: hand-crafted.** No template structure (Sonic's pixel title, Grimora's painted bars, the 11 cards without a `Text` group, the 5
  rasterized titles; rulebook pages until item 15): PSD-first. The data file holds the database-only fields and read-only mirrors of the
  parsed values (`source: "psd"`), written by the extractor when the guard classifies a pushed PSD as tier C (the pipeline, never the
  editor: one writer); the editor offers only database-only fields and says "edit in Photoshop" for the rest, until the owner
  redoes the card on a template (question 4). Field cards (50) have no template in `MDW/Templates/`: tier C until one exists (question 7).
  2026-10-21: ledger questions 3 and 6 are answered (drawn bars rebuilt as template rows, the six raster titles retyped), so those cards
  can reach tier A after the batch.
- **Order.** The goldens first (Mini Doomer, The Freezer, The Lich King; `tests/goldens/`), then one series, then a class at a time; every
  batch re-scores against `cards_v1_icc` (no region worse) and records its share of byte-identical regions (section 8, M5).
- **Harness bar.** The pipeline's parity bar (bad <= 0.5%, mae <= 2.0 since 2026-10-22) admits an *edit*; A1 admits a *source change* and is stricter on
  purpose (0 bad pixels, every region): a card that fails A1 in any region keeps its PSD as source.

## 5. Browser preview

| Option | What | Cost | Verdict |
|---|---|---|---|
| (a) Exact: the fork in the page via Pyodide | psd-tools core (attrs, Pillow, numpy) + `composite` (aggdraw, scipy) + `text` (uharfbuzz >= 0.56, freetype-py >= 2.5; `/tmp/wt-text-engine/pyproject.toml`) + the Cython `_rle` as wasm; template + art + fonts over the wire (87 faces used, psd-research `REPORT.md:46`; `MDW/Fonts` top level 225 MB over 272 files, a subset to confirm) | Pyodide ships numpy, scipy and Pillow (to confirm, offline); no Pyodide build of aggdraw or uharfbuzz is known and freetype-py loads libfreetype through ctypes (to confirm): weeks of research for an unknown | research |
| (b) Approximate: a JavaScript template renderer | Canvas 2D: pre-rendered template layers per variant, digit cells, icons the site has, text in the card fonts via `@font-face` (the site ships none today; `MDW/Fonts/stoogedings.ttf` exists); browser line breaking is not the Single-line composer | a second renderer that drifts unless the harness scores it against the fork on every card in CI (headless browser: new dependency) | text and bars only, labelled approximate |
| (c) Server renders from the pipeline's dry run | `card-edits.yml` `workflow_dispatch` (design §8) on the staged values; the browser polls the run with the editor's token and downloads the before/after/diff crops the pipeline already uploads as an artifact. Needs one extension: today's inputs are only `card` + `dry_run` and the job bakes `site/Card Edits.json` at head, so unpublished staged values reach it only through a new `values` input (JSON) or a scratch branch; the artifact comes as the API's zip, and whether the editor's classic `public_repo` token may read Actions on the public repo is to confirm | minutes (job setup + ~2 min per card, to confirm idle); Actions minutes are free on a public repo; API budget per poll | first |

**Recommendation.** (c) first: exact and cheap once the `values` input lands (a small change to design §8). Then (b) for the ability box
and stat bars only, instant, never a gate. (a) as a one-day spike that answers "which of the five native dependencies build for wasm, and
how big is the download"; go or no-go on the answer.

## 6. The editor surface

| Card part | Today (`edit.js` form, `formHtml`) | Item 20 | This campaign |
|---|---|---|---|
| Name | input; 29 live edits; `Text/Name` becomes the parser's `display_name` (design Q1) | double-click | title render preview; a reprint is a PSD job |
| Stats | number inputs 0-99; "Split into creatures" table | click a bar cell | two-row encoding above 10 (L3); per-creature rows; digit effects follow the toggle (12) |
| Ability | textarea, token bar inserting `[type]`, live HTML preview `#ef-preview` | inline; `[` token picker, `(` tag completion | chart colours and FX in the preview; "does not fit" from the engine's `overflow` before publish (via 5c); FX layers regenerated (14) |
| Stars | number | proposed inline | star style picker; the pipeline moves the folder (design Q4) |
| Types | icon toggles | type-row palette | header badges stay images (L1 scope, the user 2026-10-21); inline icons as font tokens |
| Art | none | none | upload to the asset group; placement and crop in the clip layer; ability background variant |
| Template | none | none | pick among the templates; a switch is a rebuild |
| Status words | none (colour is derived) | none | styling shown from the chart; never typed by hand |
| Rulebook pages | title, text, note form | title and text proposed | text-first pages (15) with pagination preview |
| Database-only | kind, item type, series, subseries, note, aliases, author, duplicate of, checks, ban, hidden | drag-to-assign facets, inline comboboxes | unchanged; the same fields in the data file |
| Batch | one batch bar, Review and publish, `describe(op)` sentences | per-item revert | mass operations: series, star style, template, "re-render this series" |
| Duplicates | `dup_of`; numbered PNGs from hidden title layers | - | `exports` in the data file; the pipeline exports every PNG |
| History | image version viewer (`app.js:396-434`: every commit that changed the PNG, raw with jsdelivr fallback, side by side, ↑/↓); text history | - | pixel diff of two versions in-page (canvas), data-file diff per version, rollback = a publish that restores the data file at that sha and lets the pipeline re-render |

**Reuse.** Sign-in and permissions exactly as today (classic token with `public_repo`, push permission on the repo, admins from
`site/Website Admins.txt`; `signIn`). `commitFiles` (git data API, one commit, 422 retry up to 3), the stage and `K.sent` preview until
`S.data.generated` is newer, `renderBar`, `reviewHtml`, `describe`; `editsFile` becomes a `dataFile(op)` writer for `<Stem>.json`. Everything
below the returned object is private and must be exported first (`site/NOTES.md` §3).
**Conflicts and locks.** No locks (static site). Optimistic concurrency: an op records the data file's blob it was staged on; at publish the
re-read at head (`readFile(path, head)`) with a different blob opens a per-field mine / theirs / base view; non-overlapping fields merge,
overlapping ones are chosen by hand. A Photoshop save on a tier A card is detected by the guard on push: if the PSD still round-trips to the
data file, nothing; if not, a conflict report (Discord, and a "PSD diverged: hp, ability" badge on the card) and the card drops to tier C until
someone publishes the parsed values into the data file or re-renders from data. The pipeline never overwrites a human's PSD on its own.

## 7. Sync guarantees and failure handling

1. One writer for PSD and PNG: the pipeline (Actions; the jetson as fallback, design §8). Humans write PSDs on tier C cards or to resolve a
   conflict; the `recent` rule (30 min, `card_bake.RECENT`) and the guard keep the two apart.
2. The data file is the truth for tiers A and B; `renders.json` stamps what each PSD/PNG was built from, so "pending" and "stale" are
   computed, never remembered.
3. Optimistic concurrency on blob shas for every writer: the editor (above), the pipeline (`edit-changed`, `head-moved`, 3 tries), the TTS
   save (`COMMIT_TRIES`).
4. The standardization check runs on every push and classifies every changed PSD; a diverged tier A PSD is a report, never a silent re-render.
5. Photoshop optional, never forbidden: art and tier C cards are Photoshop work; a tier A PSD opened to retouch art means "update the asset".
6. Round trip on every build (`reparse` generalised): the build refuses rather than commit a PSD that reads back differently.
7. No new pickle; the `/update_stats` -> `diff_stats.py` -> `/push` flow unchanged, its internals not: under option (a) `/update_stats`
   reads data-file blobs instead of PSDs for tier A cards (section 2, API cost there).
**Failures.** Refusals are data with a code: the pipeline's closed list plus `tier`, `diverged`, `asset-missing`, `template-missing`; per card
all-or-nothing; the site shows the reason as it shows `refused` today; the manifest and the Discord report carry it. Nothing partial is
committed. Undo is a revert of the card's commit (PSDs do not merge). A bad data file (schema check at build and at publish) is refused
before it reaches the pipeline.

## 8. Milestones

- **M1 Pipeline accepted.** The 10 edits of design §12 land; `Card Edits.json` carries 0 PSD-kind fields for a week. (Task 3, not this campaign.)
- **M2 Standardized.** The ledger's entries carry commit shas; `check_cards.py` reports 0 deviations in its classes on a push.
- **M3 Parity.** Successor of `cards_v1_icc`: no card > 2% outside class B, rulebook median < 0.1%, class A rows within the parity bar.
- **M4 First data-native card.** Mini Doomer regenerated from `Mini_Doomer.json`: gates A1-A4 green; opens in Photoshop clean.
- **M5 Share of data-native cards (A1-A4 green).** 10%, 50%, 90% of the 651, with the share of byte-identical regions beside it; the
  campaign's scoreboard, re-measured per batch.
- **M6 Minutes from publish to PSD landing.** Measured on the Action for one card (estimate under 5, to confirm idle); no "pending" after `site.yml`.
- **M7 `Card Edits.json` shrinks.** PSD-kind fields to 0 (M1), database-only fields migrated, file deleted, one schema file for both repos.
- **M8 Editor.** Art placement, a template switch and one rollback done on real cards by someone other than the user, through the batch bar.
- **M9 Preview.** (c) shows a dry-run crop in the editor; (b) answers in under 100 ms; (a) has its go/no-go.

## 9. Order and rough sizes

S = a session, M = a few, L = many. Everything after 1 waits for the hold to lift.
1. Edit pipeline, Task 3 steps 4-7 (running): M. 2. Standardization batch (ledger §4): L. 3. Parity rounds (classes A, C, D, text gamma,
groups; B by PSD re-save, 2026-10-21: B in the fork instead, round 4): L, as fork PRs. 4. Fork template-fill API (section 3.4): M. 5. Schema, extractor, tiering in `check_cards.py`: M.
6. Card-as-data migration, tier A batches (M4, M5): L. 7. Preview (c), then (b), the (a) spike: S + M + S. 8. Extended editor (item 20 first,
then art, template, history): L. 9. `Card Edits.json` retirement (after item 9 moves to the history walk) and the derived `stats.pkl`: M.

## 10. Open questions for the user

1. Data file beside the PSD (`<Stem>.json`) or under `site/cards/`?
   2026-10-21: under `site/cards/` (the user).
2. Tier A art: a flat asset cut from the official render (A1 identical by construction; the committed PSD loses its art layers) or the
   original smart object kept (A1 then holds only outside the art region; Photoshop art editing survives in the PSD)?
3. Authors and the constructed ban list: into the data files, or the two text files stay?
4. Tier C cards: redo them on a template (look changes, per card) or PSD-first forever?
5. Preview: is (c)'s minutes-long latency enough to start, or is (b) wanted from day one?
6. Who may re-render a diverged tier A card: anyone with push, or admins only?
7. Field cards and status effects: in scope (a field template to make; `Status_Effect_Template.psd` exists)? Rulebook pages only through item 15?
8. `stats.pkl` derived from the data files for tier A cards (section 2, option a)?
9. Duplicates: one data file with `exports`, or one per PNG referencing the original?

## 11. Known wrong paths (fenced off)

- A backend, accounts or a render server: state outside GitHub, hosting, a second place truth lives.
- A second source of truth: a hand-patched `stats.pkl`, a data file and `Card Edits.json` both authoritative for the same field, build state
  stored inside the data file (circular staleness: the stamp lives in `renders.json`).
- Regex-reading free text into structure: status words, tags and types are derived at build from the chart and the manifest, never stored
  twice; the ability stays text with tokens. Collapsing gap whitespace or gaps (archaeology Entry 2).
- Editing PSDs during the hold, 0-pixel renames included; regenerating a PSD a human changed since the data file did (`recent`, `diverged`).
- Two renderers without a shared harness: a JS preview or a Pyodide build nobody scores against the fork drifts silently.
- Renaming PSD or PNG files to fix stems (L12); removing `name` and expecting the stem; a new pickle (Entry 7).
- `force=True` baselines, `layer.composite()` deltas, fill-only recolouring of a styled status word (design §15).
- Letting the editor write a PSD or PNG directly, or any writer besides the pipeline: the one-writer rule is the sync guarantee.
