# Smaller PNG pushes

**Status:** designed 2026-10-25 from the research below, rulings in; not built. Queue item 3. Research:
`~/Documents/Tools/wt/scratch/png-research/REPORT.md` (lossless options, A1-A10) and
`~/Documents/Tools/wt/scratch/repo-size/workflow-result.json` (what grows the repo).

**Goal.** Every PNG that enters TTSCardMaker from now on is as small as lossless allows, so the repo (31 GB on GitHub,
the owner's main storage concern) grows more slowly.

**Rulings that bind it** (owner 10-24, memory `owner-decisions-2026-10-23-night`)
- Keep PNGs for TTS (no JPG copies; TTS caches by commit SHA, download size is not the concern).
- New pushes: the oxipng CLI, XMP strip, fully opaque cards saved as RGB.
- Never rewrite existing PNGs (a rewrite adds history). No batch recompression; `commands/png_recompress.py` goes in
  the test cut.

## What the research found

- One card edit adds about 2.4 MB packed to the repo; 92% of it is PNG. Photoshop's PNG is about 1.36 MB of that, and
  the guards' lossless recompression after the push adds another 0.82 MB (the old PNG is already in history).
- Today's call (pyoxipng 9.1.1, level 2, reductions off, every chunk kept) reaches about 95% of any lossless tool. Left:
  XMP strip 1.0% overall (up to 17% on small cards); opaque RGBA as RGB 2.2% overall, 12.6-13.5% on the 201 opaque
  cards; `-o4` 0.9% for 2.2x CPU (not chosen).
- The oxipng 10.2.1 CLI with the same settings writes byte-identical files (60/60, plus the 48 committed ones), runs on
  any Python, and blocks the bot's other threads for 19 ms instead of 3.2 s.
- PNGs the bot writes today: guard re-exports (`png_export`, recompressed), previews (recompressed, never committed),
  and **bakes (`psd_edit.save_png`, PIL's default deflate, not recompressed)**.

## Design

1. **CLI instead of pyoxipng** (`commands/png_export.py` `optimize`, about 20 lines): run
   `oxipng -q --force -o 2 --nb --nc --np --ng --timeout 120 --out <dst> <src>` (today's bytes; flags from the research's
   `pyoxipng-fork/scripts/bench.py`). Binary: the Actions download the 10.2.1 release and check a pinned sha256 (card-sync, card-guards, card-preview); the Mac `brew install oxipng`;
   the jetson needs none. pyoxipng leaves `requirements-cards.txt`; the `oxipng` skip markers in the tests check
   `shutil.which("oxipng")`.
2. **XMP strip:** `--strip iTXt,tEXt,zTXt` (keeps iCCP, sRGB, gAMA, cHRM, pHYs; `--strip safe` would drop the profile).
   `same_png` ignores those three chunk types.
3. **Opaque as RGB:** when every alpha is 255, write colour type 2 (drop `--nc` from step 1 so oxipng
   may reduce the colour type; keep any result that is RGB or RGBA, else the step 1 file). `same_png` compares pixels as
   RGBA instead of the header. The bake and guard 1 read RGB and RGBA already; prove it with one bake of Mini Doomer
   saved as RGB before landing.
4. **Where it runs, before the one commit (no added history):** the bake's PNG (`psd_edit.save_png` output through
   `png_export.recompress`), guard re-exports (already), the stars-folder move's redraw. Card makers' pushed PNGs: see
   the open question.

Expected: steps 2-3 save 1-13% per file on top of today's call; the bake's gain over PIL's default deflate is not
measured yet (Photoshop's exports shrink 69-71% under today's call). Existing files are untouched.

## Verification (lean)

- The CLI against pyoxipng on the 60-card research sample: identical bytes with step 1's flags.
- Steps 2-3 dry over the 731 current card PNGs (nothing written): every result decodes to the same RGBA pixels.
- One live bake of an opaque card; TTS shows the RGB PNG (the next live TTS check).
- Tests: change the existing recompression tests, add none.

## Card makers' pushed PNGs

The guards keep recompressing them after the push (owner, 10-23; confirmed 10-25 ~06:30): a second PNG in history
(+0.82 MB per card) for a smaller current file.
